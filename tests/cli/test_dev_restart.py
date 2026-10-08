"""The dev monitor loop must restart the web server from the same cwd it started it in.

Regression guard for the Aug-31 TUI/dev note: the restart path once used a
different base directory than the initial spawn, so a crashed Vite server came
back up outside ``apps/web`` (wrong ``vite.config``/``node_modules`` resolution)
and immediately died again.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _FakeThread:
    """Thread stand-in: the test drives the loop synchronously."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class _ThreadingStub:
    Thread = _FakeThread

    def __getattr__(self, name):
        return getattr(threading, name)


@pytest.fixture
def run_dev(monkeypatch, tmp_path):
    """Run ``_cmd_api_and_web`` with a web server that crashes once."""
    import commands.dev as dev_mod

    monkeypatch.setattr(dev_mod, "log", MagicMock())

    args = MagicMock()
    args.host = "localhost"
    args.port = 8000
    args.web_port = 3000

    popen_calls: list[dict] = []

    def fake_popen(cmd, **kwargs):
        popen_calls.append(kwargs)
        proc = MagicMock()
        if "uvicorn" in " ".join(str(part) for part in cmd):
            proc.poll.return_value = None  # API stays up
        else:
            # The readiness loop never polls (the port looks open at once), so
            # the first poll() is the monitor loop's — report the crash there.
            proc.poll.side_effect = [0]
        return proc

    api_checks = {"seen": 0}

    def api_ready(port):
        # First call decides "reuse an existing API?" → no; later calls are the
        # startup wait loop, which must succeed immediately.
        api_checks["seen"] += 1
        return api_checks["seen"] > 1

    monitor_ticks = {"seen": 0}

    def fake_sleep(seconds):
        if seconds == 2:  # monitor cadence; readiness waits use 1s / 1.5s
            monitor_ticks["seen"] += 1
            if monitor_ticks["seen"] >= 2:
                raise KeyboardInterrupt
        return None

    fake_time = MagicMock()
    fake_time.sleep.side_effect = fake_sleep

    monkeypatch.setattr("commands.dev._repo_root", lambda: Path(tmp_path))
    monkeypatch.setattr("commands.dev._check_api_ready", api_ready)
    monkeypatch.setattr("commands.dev._check_web_ready", lambda port: False)
    monkeypatch.setattr("commands.dev._check_port", lambda port: True)
    monkeypatch.setattr("commands.dev._is_port_bound", lambda port: False)
    monkeypatch.setattr("commands.dev._is_eaddrinuse", lambda lines: False)
    monkeypatch.setattr("commands.dev.find_server_python", lambda root: "/usr/bin/python3")
    monkeypatch.setattr("commands.dev._cleanup", MagicMock())
    monkeypatch.setattr("commands.dev.signal", MagicMock())
    monkeypatch.setattr("commands.dev.webbrowser", MagicMock())
    monkeypatch.setattr("commands.dev.time", fake_time)
    monkeypatch.setattr("commands.dev.threading", _ThreadingStub())
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    from commands.dev import _cmd_api_and_web

    _cmd_api_and_web(args)
    return popen_calls, tmp_path


def test_monitor_restart_reuses_the_initial_web_cwd(run_dev):
    popen_calls, root = run_dev
    web_calls = [call for call in popen_calls if Path(call["cwd"]).name == "web"]
    expected = str((root / "apps" / "web").resolve())

    assert len(web_calls) == 2, "expected the initial spawn plus one restart"
    assert web_calls[0]["cwd"] == expected
    assert web_calls[1]["cwd"] == expected, "restart used a different working directory"


def test_api_spawns_from_the_repo_root(run_dev):
    popen_calls, root = run_dev
    api_calls = [call for call in popen_calls if "cwd" in call and Path(call["cwd"]).name != "web"]

    assert len(api_calls) == 1, "the API should have been spawned, not reused"
    assert api_calls[0]["cwd"] == str(root)
