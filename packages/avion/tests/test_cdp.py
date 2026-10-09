"""CDP backend + CLI tests — real Chromium over the DevTools protocol.

Degrades explicitly: skips with a stated reason when Chromium or
websockets is unavailable (never a silent pass, never a crash).
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from avion.core.element import ElementLocator


def _find_chromium() -> str | None:
    env = os.environ.get("AVION_CHROMIUM")
    if env and Path(env).exists():
        return env
    cache = Path.home() / ".cache" / "ms-playwright"
    if cache.is_dir():
        for pattern in (
            "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
            "chromium-*/chrome-linux/chrome",
        ):
            hits = sorted(cache.glob(pattern))
            if hits:
                return str(hits[-1])
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        from shutil import which

        found = which(name)
        if found:
            return found
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


HAS_WEBSOCKETS = importlib.util.find_spec("websockets") is not None


@pytest.fixture(scope="module")
def chromium_ws(request):
    """Launch a debuggable headless Chromium; yield its CDP endpoint."""
    if not HAS_WEBSOCKETS:
        pytest.skip("websockets not installed — CDP unavailable (pip install websockets)")
    binary = _find_chromium()
    if binary is None:
        pytest.skip("no Chromium binary found (AVION_CHROMIUM or ms-playwright cache)")
    port = _free_port()
    user_dir = tempfile.mkdtemp(prefix="avion-cdp-test-")
    args = [binary]
    if "headless-shell" not in binary:
        args += ["--headless=new"]
    args += [
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-sandbox",          # unprivileged userns is AppArmor-blocked here
        "--password-store=basic",  # keyring DBus lookups stall Page.navigate
        "--disable-gpu",
        "about:blank",
    ]
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(base + "/json/version", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)
    if not ready:
        proc.terminate()
        pytest.skip(f"Chromium did not expose /json/version on :{port}")
    yield f"ws://127.0.0.1:{port}"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


PAGE = """<!doctype html>
<html><head><title>Avion CDP Fixture</title></head>
<body>
  <h1>Hello Journey</h1>
  <button id="go" onclick="console.error('go-err'); window.__clicked = true">Go</button>
  <input name="q" />
  <input type="checkbox" id="agree" onchange="window.__checked = this.checked" />
  <select id="pick" onchange="window.__picked = this.value">
    <option value="a">a</option>
    <option value="b">b</option>
  </select>
</body></html>
"""


@pytest.fixture(scope="module")
def http_url():
    """Serve PAGE on a local port; yield the base URL."""
    root = tempfile.mkdtemp(prefix="avion-http-")
    (Path(root) / "index.html").write_text(PAGE)
    handler = lambda *a, **kw: SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=root, **kw
    )
    server = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()


@pytest.fixture(autouse=True)
def _isolated_cli_state(tmp_path, monkeypatch):
    """Keep the CLI capture store out of the shared $TMPDIR."""
    monkeypatch.setenv("AVION_STATE_DIR", str(tmp_path))


def _backend(chromium_ws):
    from avion.backends.cdp import CdpBackend

    return CdpBackend(cdp_url=chromium_ws)


async def _poll(fn, timeout=5.0):
    """Poll ``fn`` until truthy. Yields to the loop so the CDP reader can run."""
    import asyncio

    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        await asyncio.sleep(0.05)
    return fn()


# ── backend ─────────────────────────────────────────────────────────────


async def test_navigate_evaluate_title(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        assert await b.wait_for_text("Hello Journey", timeout=5)
        assert (await b.get_title()) == "Avion CDP Fixture"
        assert await b.evaluate("1 + 1") == 2
    finally:
        await b.stop()


async def test_console_capture_and_errors_filter(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        await b.evaluate("console.error('boom-err'); console.log('info-line')")
        entries = await _poll(
            lambda: b.console_entries() if len(b.console_entries()) >= 2 else None
        )
        assert entries, "no console entries captured"
        texts = [e["text"] for e in entries]
        assert any("boom-err" in t for t in texts), texts
        errors = b.console_entries(errors_only=True)
        assert all(e["level"] in ("error", "assert") for e in errors)
        assert any("boom-err" in e["text"] for e in errors), errors
        b.clear_events()
        assert b.console_entries() == []
    finally:
        await b.stop()


async def test_exception_captured_as_console_error(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        # evaluate-time throw: raises, and is recorded for console --errors
        with pytest.raises(RuntimeError, match="kaboom"):
            await b.evaluate("(function(){ throw new Error('kaboom'); })()")
        entries = await _poll(
            lambda: [
                e
                for e in b.console_entries()
                if "kaboom" in e["text"] and e["level"] == "error"
            ]
        )
        assert entries, b.console_entries()
        # async throw: arrives via Runtime.exceptionThrown (no evaluate involved)
        await b.evaluate("setTimeout(() => { throw new Error('later-boom'); }, 0)")
        entries = await _poll(
            lambda: [
                e for e in b.console_entries() if "later-boom" in e["text"]
            ]
        )
        assert entries, b.console_entries()
        assert all(e["level"] in ("error", "assert") for e in entries)
    finally:
        await b.stop()


async def test_wait_for_text_times_out(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        assert await b.wait_for_text("Hello Journey", timeout=2) is True
        assert await b.wait_for_text("no-such-string", timeout=0.4) is False
    finally:
        await b.stop()


async def test_network_capture(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        entries = await _poll(
            lambda: [
                e for e in b.network_entries() if "index.html" in e.get("url", "")
            ]
        )
        assert entries, b.network_entries()
        assert entries[0]["status"] == 200
    finally:
        await b.stop()


async def test_click_and_fill_over_cdp(chromium_ws, http_url):
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        # navigate returns on commit — the target may not be parsed yet
        button = await b.wait_for(ElementLocator.css("#go"), timeout=5)
        await b.click(button)
        assert await b.evaluate("window.__clicked === true")
        field = await b.find_element(ElementLocator.css("input[name=q]"))
        assert field is not None
        await b.fill(field, "typed-value")
        assert await b.evaluate("document.querySelector('input[name=q]').value") == "typed-value"
    finally:
        await b.stop()


async def test_check_uncheck_select_over_cdp(chromium_ws, http_url):
    """The JS these build once silently failed — evaluate now surfaces it."""
    b = _backend(chromium_ws)
    await b.start()
    try:
        await b.navigate(http_url)
        box = await b.wait_for(ElementLocator.css("#agree"), timeout=5)
        await b.check(box)
        assert await b.evaluate("document.querySelector('#agree').checked") is True
        await b.uncheck(box)
        assert await b.evaluate("document.querySelector('#agree').checked") is False

        pick = await b.wait_for(ElementLocator.css("#pick"), timeout=5)
        await b.select_option(pick, "b")
        assert await b.evaluate("document.querySelector('#pick').value") == "b"
    finally:
        await b.stop()


def test_missing_websockets_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "websockets", None)
    from avion.backends.cdp import _websockets

    with pytest.raises(ImportError) as exc:
        _websockets()
    assert "pip install websockets" in str(exc.value)


# ── CLI ─────────────────────────────────────────────────────────────────


def test_cli_help_and_no_args(capsys):
    from avion.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "snapshot" in out and "console" in out


def test_cli_connect_failure_is_actionable(capsys):
    from avion.cli import main

    code = main(["--cdp", "ws://127.0.0.1:1", "open", "http://example.invalid"])
    assert code == 2
    err = capsys.readouterr().err
    assert "remote-debugging-port" in err


def test_cli_journey_one_shots(chromium_ws, http_url, capsys):
    from avion.cli import main

    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    capsys.readouterr()
    assert main(["--cdp", chromium_ws, "wait", "Hello Journey", "--timeout", "5"]) == 0
    assert main(["--cdp", chromium_ws, "click", "#go"]) == 0
    assert main(["--cdp", chromium_ws, "snapshot"]) == 0
    out = capsys.readouterr().out
    assert "Avion CDP Fixture" in out
    assert main(["--cdp", chromium_ws, "click", "#does-not-exist"]) == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert main(["--cdp", chromium_ws, "wait", "nope-text", "--timeout", "0.3"]) == 1


def test_cli_click_state_persists_across_invocations(chromium_ws, http_url, capsys):
    from avion.cli import main

    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    capsys.readouterr()
    assert main(["--cdp", chromium_ws, "click", "--by", "text", "Go"]) == 0
    capsys.readouterr()
    # separate session, same browser: the click stuck
    from avion.backends.cdp import CdpBackend

    async def check():
        b = CdpBackend(cdp_url=chromium_ws)
        await b.start()
        try:
            return await b.evaluate("window.__clicked === true")
        finally:
            await b.stop()

    import asyncio

    assert asyncio.run(check()) is True


def test_cli_module_invocation():
    """python -m avion --help works as a subprocess (packaging check)."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "avion", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": src},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "snapshot" in proc.stdout


def test_cli_console_and_network_survive_separate_invocations(
    chromium_ws, http_url, capsys
):
    """Each CLI run is a new process — history must outlive the session."""
    from avion.cli import main

    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    assert main(["--cdp", chromium_ws, "click", "#go"]) == 0
    capsys.readouterr()
    # a *later* process still sees what the earlier sessions captured
    assert main(["--cdp", chromium_ws, "console", "--errors"]) == 0
    assert "go-err" in capsys.readouterr().out
    assert main(["--cdp", chromium_ws, "network", "--json"]) == 0
    assert "index.html" in capsys.readouterr().out
    # `open` starts a fresh capture: repeated opens do not pile up history.
    # (Network has no replay, so it shows the reset exactly.)
    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    capsys.readouterr()
    assert main(["--cdp", chromium_ws, "network", "--json"]) == 0
    out = capsys.readouterr().out
    assert out.count("index.html") == 1, out
    # the reload replaced the document, so the old error is gone from both
    # Chrome's buffer and the journey's capture — a genuinely fresh start
    assert main(["--cdp", chromium_ws, "console", "--errors"]) == 0
    assert "go-err" not in capsys.readouterr().out


def test_cli_repeated_reads_do_not_grow_history(chromium_ws, http_url, capsys):
    """Chrome replays its console buffer to every attach — reads must not
    re-append it, or history grows by one copy per invocation."""
    from avion.cli import main

    assert main(["--cdp", chromium_ws, "open", http_url]) == 0
    assert main(["--cdp", chromium_ws, "click", "#go"]) == 0
    counts = []
    for _ in range(4):
        capsys.readouterr()
        assert main(["--cdp", chromium_ws, "console", "--errors"]) == 0
        counts.append(capsys.readouterr().out.count("go-err"))
    assert counts == [1, 1, 1, 1], counts
