"""avion shell-verb tests — ``avion run`` completes a recorded flow from
the terminal; ``avion launch`` owns the browser start; ``avion repl
--record`` authors a flow by driving it.

Unit layer: the Event dict round-trip, flow-file loading (including
malformed input failing LOUDLY), and parser wiring — no browser.

Journey layer (real browser, real subprocesses — how a user runs it):

1. record a flow on the mini-site (reusing the ``test_flow_complete``
   fixture and its ``_record`` helper), save it with ``EventRecorder.save``,
   then ``python -m avion run flow.json --profile …`` completes it in a
   child process of its own;
2. the attach path: ``run --cdp`` against a browser we own must complete
   the flow WITHOUT killing a browser it does not own — and its honest
   failure probe (no --base-url, relative flow) must exit 1, never count
   a page that never moved as done;
3. ``avion launch`` waits for Enter and really stops the browser (pid
   gone), and ``--no-wait`` really detaches (browser outlives the CLI —
   the test reaps it by scanning /proc for its profile marker);
4. the authoring loop: ``avion repl --record`` captures what was driven,
   then ``avion run`` completes that authored flow on a fresh profile.

Degrades with a stated skip when no Chromium is available.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from avion.cli import _load_flow_events, build_parser, main
from avion.events.models import Event, EventType
from avion.events.recorder import EventRecorder
from avion.launch import Profile, launch_chromium
from test_flow_complete import _LAUNCH_ARGS, _chromium_or_skip, _record, _SiteHandler

_CHROMIUM_ARGS = ("--arg=--no-sandbox", "--arg=--disable-gpu")


@pytest.fixture
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _child_env(tmp_path: Path, binary: str) -> dict[str, str]:
    """Sandboxed environment for `python -m avion` subprocesses."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env["PYTHONNOUSERSITE"] = "1"
    env["AVION_PROFILES_DIR"] = str(tmp_path / "profiles")
    env["AVION_STATE_DIR"] = str(tmp_path / "state")
    env["AVION_CHROMIUM"] = binary
    return env


def _pids_with(substr: str) -> list[int]:
    """PIDs whose cmdline contains ``substr`` (fixed string, no regex)."""
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if substr in cmdline:
            pids.append(int(entry.name))
    return pids


def _reap(marker: str, timeout: float = 8.0) -> None:
    """SIGTERM, escalate to SIGKILL, until no PID carries ``marker``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = _pids_with(marker)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(0.25)
    for pid in _pids_with(marker):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.time() + 3.0
    while time.time() < deadline and _pids_with(marker):
        time.sleep(0.25)
    assert not _pids_with(marker), f"could not reap processes carrying {marker}"


class TestEventRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        e = Event(
            type=EventType.CLICK,
            name="#go",
            data={"selector": "#go"},
            success=False,
            error="boom",
            duration_ms=12.5,
            timestamp=1000.5,
            seq=7,
        )
        back = Event.from_dict(e.to_dict())
        assert back.type is EventType.CLICK
        assert back.name == "#go"
        assert back.data == {"selector": "#go"}
        assert back.success is False
        assert back.error == "boom"
        assert back.duration_ms == 12.5
        assert back.timestamp == 1000.5
        assert back.seq == 7

    def test_screenshot_flag_is_dropped_never_faked(self):
        """to_dict stores a presence flag; from_dict must not turn it into data."""
        e = Event(
            type=EventType.SCREENSHOT,
            name="p.png",
            data={"screenshot": b"\x89PNG", "path": "p.png"},
        )
        d = e.to_dict()
        assert d["has_screenshot"] is True
        assert "screenshot" not in d["data"]
        back = Event.from_dict(d)
        assert "screenshot" not in back.data
        assert "has_screenshot" not in back.data

    def test_optional_fields_default_when_absent(self):
        back = Event.from_dict({"type": "navigate", "name": "/"})
        assert back.type is EventType.NAVIGATE
        assert back.success is True
        assert back.error == ""
        assert back.duration_ms == 0.0
        assert back.seq == 0

    def test_unknown_type_raises_honestly(self):
        with pytest.raises(ValueError, match="nope"):
            Event.from_dict({"type": "nope"})


class TestLoadFlowEvents:
    def test_recorder_save_loads_back(self, tmp_path):
        rec = EventRecorder(session_name="cli_test")
        rec.record(EventType.NAVIGATE, name="/")
        rec.record(EventType.FILL, name="q", data={"value": "x"})
        path = tmp_path / "flow.json"
        rec.save(str(path))
        events = _load_flow_events(str(path))
        assert [e.type for e in events] == [EventType.NAVIGATE, EventType.FILL]
        assert events[1].data == {"value": "x"}
        # the logger assigned distinct, monotonic seqs — they survive the file
        assert events[0].seq < events[1].seq

    def test_bare_list_payload_also_loads(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps([{"type": "navigate", "name": "/", "seq": 1}]))
        events = _load_flow_events(str(path))
        assert len(events) == 1 and events[0].type is EventType.NAVIGATE

    def test_dict_without_events_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"session": "x"}))
        with pytest.raises(ValueError, match="no events"):
            _load_flow_events(str(path))

    def test_malformed_event_raises_clean_value_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text(json.dumps({"events": [{"name": "typeless"}]}))
        with pytest.raises(ValueError, match="malformed event"):
            _load_flow_events(str(path))

    def test_missing_file_raises_os_error(self, tmp_path):
        with pytest.raises(OSError):
            _load_flow_events(str(tmp_path / "nope.json"))


class TestParserWiring:
    def test_run_defaults_to_attach_headless_automation(self):
        a = build_parser().parse_args(["run", "f.json"])
        assert a.cmd == "run"
        assert a.flow == "f.json"
        assert a.profile is None  # attach unless told to own a launch
        assert a.headed is False
        assert a.dry_run is False
        assert a.json is False
        assert a.base_url == ""
        assert a.arg == []

    def test_run_full_flags(self):
        a = build_parser().parse_args(
            [
                "run",
                "f.json",
                "--profile",
                "x",
                "--base-url",
                "http://h",
                "--headed",
                "--dry-run",
                "--json",
                "--arg=--no-sandbox",
            ]
        )
        assert a.profile == "x"
        assert a.base_url == "http://h"
        assert a.headed and a.dry_run and a.json
        assert a.arg == ["--no-sandbox"]

    def test_launch_flags(self):
        a = build_parser().parse_args(["launch", "--profile", "p", "--headless", "--no-wait"])
        assert a.cmd == "launch"
        assert a.profile == "p"
        assert a.headless and a.no_wait

    def test_repl_record_flag(self):
        a = build_parser().parse_args(["repl", "--record", "f.json"])
        assert a.cmd == "repl"
        assert a.record == "f.json"
        b = build_parser().parse_args(["repl"])
        assert b.record is None


class TestRunExitCodes:
    """The no-browser failure paths: loud message, exit 1, no traceback."""

    def test_missing_flow_file_fails_loudly(self, capsys, tmp_path):
        code = main(["run", str(tmp_path / "nope.json")])
        assert code == 1
        assert "avion:" in capsys.readouterr().err

    def test_flow_without_events_fails_loudly(self, capsys, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"session": "x"}))
        code = main(["run", str(path)])
        assert code == 1
        assert "no events" in capsys.readouterr().err


class TestLaunchJourney:
    """`avion launch` — the browser's lifecycle belongs to the verb."""

    def test_launch_waits_then_stops(self, tmp_path):
        binary = _chromium_or_skip()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "avion",
                "launch",
                "--profile",
                "life",
                "--headless",
                *_CHROMIUM_ARGS,
            ],
            input=b"\n",
            env=_child_env(tmp_path, binary),
            capture_output=True,
            timeout=90,
        )
        out = proc.stdout.decode()
        assert proc.returncode == 0, proc.stderr.decode()
        assert "launched profile=life" in out
        assert "endpoint=ws://" in out
        assert "stopped." in out
        m = re.search(r"pid=(\d+)", out)
        assert m, out
        with pytest.raises(ProcessLookupError):
            os.kill(int(m.group(1)), 0)

    def test_launch_no_wait_detaches(self, tmp_path):
        binary = _chromium_or_skip()
        marker = f"--user-data-dir={tmp_path / 'profiles' / 'det'}"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "avion",
                "launch",
                "--profile",
                "det",
                "--headless",
                "--no-wait",
                *_CHROMIUM_ARGS,
            ],
            env=_child_env(tmp_path, binary),
            capture_output=True,
            timeout=90,
        )
        out = proc.stdout.decode()
        assert proc.returncode == 0, proc.stderr.decode()
        assert "endpoint=ws://" in out
        assert "detach: export AVION_CDP_URL=" in out
        try:
            # The whole point of --no-wait: the browser outlives the CLI.
            assert _pids_with(marker), "detached browser should still be running"
        finally:
            _reap(marker)


class TestRunJourney:
    """`avion run` — record once, complete again, from a child process."""

    async def test_run_profile_completes_recorded_flow(self, site, tmp_path):
        binary = _chromium_or_skip()
        flow = tmp_path / "flow.json"
        launched = launch_chromium(
            Profile("rec", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            await _record(site, launched, login=True, save_to=flow)
        finally:
            launched.stop()
        assert flow.exists(), "recording was not saved"

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "avion",
                "run",
                str(flow),
                "--profile",
                "e2e",
                "--base-url",
                site,
                *_CHROMIUM_ARGS,
            ],
            env=_child_env(tmp_path, binary),
            capture_output=True,
            timeout=120,
        )
        out = proc.stdout.decode()
        assert proc.returncode == 0, f"stdout={out}\nstderr={proc.stderr.decode()}"
        assert out.startswith("completed:"), out
        assert "/app?q=avion" in out  # final URL — reached by form submission
        # the profile sandbox was honored: the child's browser lived HERE
        assert (tmp_path / "profiles" / "e2e").exists()

    async def test_run_cdp_attach_completes_and_spares_browser(self, site, tmp_path):
        binary = _chromium_or_skip()
        flow = tmp_path / "flow.json"
        launched = launch_chromium(
            Profile("att", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            await _record(site, launched, login=True, save_to=flow)
            endpoint = launched.endpoint.base_ws_url
            env = _child_env(tmp_path, binary)

            # Honest-failure probe: the flow carries relative paths ("/"),
            # and without --base-url the goto CANNOT land — the report must
            # say so and exit 1, never count a page that never moved as done.
            probe = subprocess.run(
                # --cdp is the ROOT flag, before the subcommand — every
                # avion verb attaches that way.
                [sys.executable, "-m", "avion", "--cdp", endpoint, "run", str(flow)],
                env=env,
                capture_output=True,
                timeout=120,
            )
            probe_out = probe.stdout.decode()
            assert probe.returncode == 1, probe_out
            assert "FAILED:" in probe_out
            assert "did not land" in probe_out

            # The attach journey proper: --base-url gives the relative
            # navigation its origin — the flow completes on our browser.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "avion",
                    "--cdp",
                    endpoint,
                    "run",
                    str(flow),
                    "--base-url",
                    site,
                ],
                env=env,
                capture_output=True,
                timeout=120,
            )
            out = proc.stdout.decode()
            assert proc.returncode == 0, f"stdout={out}\nstderr={proc.stderr.decode()}"
            assert out.startswith("completed:"), out
            # attach closes its ws connection, never the browser itself
            assert launched.alive, "attach must not kill a browser it does not own"
        finally:
            launched.stop()


class TestRecordJourney:
    """`avion repl --record` — author a flow by driving, then `avion run`
    completes it: the full shell-native create-then-run loop."""

    async def test_record_in_repl_then_run_completes(self, site, tmp_path):
        binary = _chromium_or_skip()
        flow = tmp_path / "authored.json"
        launched = launch_chromium(
            Profile("author", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            endpoint = launched.endpoint.base_ws_url
            script = "\n".join(
                [
                    f"open {site}/login",
                    "fill input[name=user] ada",
                    "fill input[name=pass] lovelace",
                    "click #go",
                    "wait Dashboard",
                    'fill input[name=q] "avion flows"',
                    "click #search",
                    "quit",
                ]
            ).encode()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "avion",
                    "--cdp",
                    endpoint,
                    "repl",
                    "--record",
                    str(flow),
                ],
                input=script,
                env=_child_env(tmp_path, binary),
                capture_output=True,
                timeout=120,
            )
            out = proc.stdout.decode()
            assert proc.returncode == 0, proc.stderr.decode()
            assert "RECORDING" in out
            assert "recorded " in out and str(flow) in out

            events = _load_flow_events(str(flow))
            kinds = [e.type for e in events]
            assert EventType.NAVIGATE in kinds
            assert EventType.CLICK in kinds
            assert EventType.FILL in kinds
            # one barrier per click (2) and fill (3), one for the wait
            assert kinds.count(EventType.ELEMENT_FOUND) >= 5
            # the fill values survive for replay
            fills = [e.data.get("value") for e in events if e.type is EventType.FILL]
            assert "ada" in fills and "avion flows" in fills
        finally:
            launched.stop()

        # The authored flow completes on a FRESH profile. Absolute URLs
        # were driven, so no --base-url is needed.
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "avion",
                "run",
                str(flow),
                "--profile",
                "ran",
                *_CHROMIUM_ARGS,
            ],
            env=_child_env(tmp_path, binary),
            capture_output=True,
            timeout=120,
        )
        out = run.stdout.decode()
        assert run.returncode == 0, f"stdout={out}\nstderr={run.stderr.decode()}"
        assert out.startswith("completed:"), out
