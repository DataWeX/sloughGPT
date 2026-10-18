"""Agent-verb + input-primitive tests — our own backend and model drive the loop.

Two seams are proved here, both against a real browser and real fixtures
(no mocks anywhere):

1. ``CdpBackend``'s input primitives (the methods the agent loop's
   Mouse/Keyboard controllers call — click, double-click, type, press,
   scroll) actually drive a page: the Playwright backend always had
   these; this is our own, over CDP's Input domain.
2. The ``avion agent`` verb closes the loop end to end: a real local
   .soul checkpoint decides actions from the accessibility tree and the
   loop executes them. A tiny untrained model will not *solve* a task,
   and the tests do not pretend otherwise — they assert the wiring is
   live (model consulted, at least one step executed, honest
   stop_reason, honest exit code).

Degrades explicitly (stated skips) when no Chromium or no checkpoint is
on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from avion.launch import Profile, find_chromium, launch_chromium

_PY = sys.executable
_PKG = Path(__file__).resolve().parents[1]  # packages/avion
_PACKAGES = _PKG.parent  # packages


class _SiteHandler(BaseHTTPRequestHandler):
    """Counter button + a form + enough height to scroll."""

    def log_message(self, *args):  # silence the server thread
        pass

    def do_GET(self):
        if self.path.startswith("/done"):
            body = b"<h1>Submitted</h1>"
        else:
            body = (
                b"<h1>Agent site</h1>"
                b"<button id='go' onclick=\"var n=document.getElementById('n');"
                b'n.textContent=Number(n.textContent)+1">Press</button>'
                b"<span id='n'>0</span>"
                b"<form action='/done' method='get'>"
                b"<input name='q' id='q'><button type='submit' id='sub'>Go</button>"
                b"</form><div style='height:3000px'></div>"
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def _chromium_or_skip() -> str:
    try:
        return find_chromium()
    except RuntimeError:
        pytest.skip("no Chromium available — set AVION_CHROMIUM or install chromium")


def _soul_or_skip() -> str:
    candidates: list[Path] = []
    env = os.environ.get("AVION_TEST_SOUL")
    if env:
        candidates.append(Path(env))
    home = Path.home() / ".local" / "share" / "sloughgpt" / "souls"
    if home.is_dir():
        candidates.extend(sorted(home.glob("*.soul")))
    repo = _PACKAGES.parent / "checkpoints"
    if repo.is_dir():
        candidates.extend(sorted(repo.glob("*.soul"), key=lambda p: p.stat().st_size))
    for path in candidates:
        if path.is_file():
            return str(path)
    pytest.skip("no .soul checkpoint on disk — set AVION_TEST_SOUL to run the live agent")


_LAUNCH_ARGS = ("--no-sandbox", "--disable-gpu")


class TestCdpInput:
    """The primitives the agent loop needs — on a real page, driven by ours."""

    async def test_click_type_press_scroll(self, site, tmp_path):
        binary = _chromium_or_skip()
        launched = launch_chromium(
            Profile("cdpinput", root=tmp_path / "profiles"),
            headless=True,
            binary=binary,
            extra_args=_LAUNCH_ARGS,
            timeout=30.0,
        )
        try:
            backend = launched.backend
            await backend.start()
            try:
                await backend.navigate(site)
                assert await backend.wait_for_text("Agent site", timeout=10.0)

                # Click the button by its real on-screen box.
                box = json.loads(
                    await backend.evaluate(
                        "JSON.stringify((() => {"
                        " const r = document.getElementById('go').getBoundingClientRect();"
                        " return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })())"
                    )
                )
                await backend.mouse_click(box["x"], box["y"])
                assert await backend.evaluate("document.getElementById('n').textContent") == "1", (
                    "mouse_click did not hit the button"
                )

                # Double-click counts twice — the page must see one dblclick.
                await backend.mouse_double_click(box["x"], box["y"])
                assert await backend.evaluate("document.getElementById('n').textContent") == "3", (
                    "mouse_double_click did not register as a dblclick"
                )

                # Scrolling moves the page — while still on the tall one.
                await backend.mouse_scroll(200, 400, 0, 600)
                await asyncio.sleep(0.3)
                assert await backend.evaluate("window.scrollY") > 0, (
                    "mouse_scroll did not move the page"
                )

                # Typing goes into the focused field, key by key.
                await backend.evaluate("document.getElementById('q').focus()")
                await backend.keyboard_type("avion agent", delay_ms=1)
                assert await backend.evaluate("document.getElementById('q').value") == (
                    "avion agent"
                ), "keyboard_type did not reach the focused input"

                # Enter submits the form the same way a human's would.
                await backend.keyboard_press("Enter")
                assert await backend.wait_for_text("Submitted", timeout=10.0), (
                    "keyboard_press('Enter') did not submit the form"
                )
            finally:
                await backend.stop()
        finally:
            launched.stop()


def _avion_env(profiles_dir: Path | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(_PKG / "src"), str(_PACKAGES / "core-py"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    env.pop("AVION_SOUL", None)  # the verb's env fallback must not leak into tests
    if profiles_dir is not None:
        env["AVION_PROFILES_DIR"] = str(profiles_dir)
    return env


def _avion(*argv: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_PY, "-m", "avion", *argv], capture_output=True, text=True, env=env, timeout=240
    )


class TestAgentVerb:
    def test_refuses_to_run_without_a_model(self):
        proc = _avion("--cdp", "ws://127.0.0.1:1", "agent", "press the button", env=_avion_env())
        assert proc.returncode != 0, "the verb ran with no model configured"
        assert "model" in (proc.stderr + proc.stdout).lower()

    def test_real_browser_real_soul_drives_the_loop(self, tmp_path):
        _chromium_or_skip()
        soul = _soul_or_skip()
        proc = _avion(
            "agent",
            "open the page and press the button",
            "--soul",
            soul,
            "--profile",
            "agentcli",
            "--max-steps",
            "2",
            "--json",
            "--arg=--no-sandbox",
            "--arg=--disable-gpu",
            env=_avion_env(tmp_path / "profiles"),
        )
        assert proc.returncode in (0, 1), f"unexpected exit: {proc.stderr[-600:]}"
        report = json.loads(proc.stdout)
        assert report["model_name"] == soul, "the run did not use the requested checkpoint"
        assert report["stop_reason"] in (
            "done",
            "fail",
            "max_steps",
            "deadline",
            "no_progress",
            "unverified",
            "aborted",
        )
        assert len(report["trajectory"]["steps"]) >= 1, "the loop never took a step"
