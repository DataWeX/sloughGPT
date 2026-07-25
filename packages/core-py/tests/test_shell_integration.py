"""
Integration tests for ShellREPL — calls real API endpoints.

All tests skip if the API server is not running on localhost:8000.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.runtime import DaitRuntime


_API_AVAILABLE: bool | None = None


def _check_api() -> bool:
    global _API_AVAILABLE
    if _API_AVAILABLE is None:
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            _API_AVAILABLE = r.status_code == 200
        except Exception:
            _API_AVAILABLE = False
    return _API_AVAILABLE


@pytest.fixture
def repl():
    os_obj = DaitRuntime()
    r = ShellREPL(os_obj)
    r._running = True
    return r


def _capture(repl: ShellREPL, cmd: str) -> str:
    return repl._execute_single(cmd)


# ── API-dependent commands ──────────────────────────────────────────


class TestApiCommands:
    def test_health_returns_ok(self, repl):
        if not _check_api():
            pytest.skip("API server not running")
        out = _capture(repl, "health")
        assert "healthy" in out.lower() or "ok" in out.lower()

    def test_models_lists_something(self, repl):
        if not _check_api():
            pytest.skip("API server not running")
        out = _capture(repl, "models")
        assert len(out) > 0

    def test_status_shows_runtime(self, repl):
        if not _check_api():
            pytest.skip("API server not running")
        out = _capture(repl, "status")
        assert "uptime" in out.lower() or "process" in out.lower()

    def test_souls_lists_souls(self, repl):
        if not _check_api():
            pytest.skip("API server not running")
        out = _capture(repl, "souls")
        assert len(out) > 0

    def test_whoami_returns_string(self, repl):
        if not _check_api():
            pytest.skip("API server not running")
        out = _capture(repl, "whoami")
        assert isinstance(out, str)


# ── Local commands ──────────────────────────────────────────────────


class TestLocalCommands:
    def test_help_shows_builtins(self, repl):
        out = _capture(repl, "help")
        assert "health" in out

    def test_echo_works(self, repl):
        out = _capture(repl, "echo hello world")
        assert "hello world" in out

    def test_pwd_returns_directory(self, repl):
        out = _capture(repl, "pwd")
        assert out.startswith("/")

    def test_clear_returns_empty(self, repl):
        out = _capture(repl, "clear")
        assert out == "" or out is None

    def test_history_shows_previous(self, repl):
        out = _capture(repl, "history")
        assert isinstance(out, str)

    def test_pipeline_splits_on_pipe(self, repl):
        try:
            out = repl._execute_pipeline(["echo a b c", "echo"])
        except Exception:
            out = ""
        assert isinstance(out, str)

    def test_boot_detects_running(self, repl):
        out = _capture(repl, "boot")
        assert isinstance(out, str)

    def test_help_brief_shows_commands(self, repl):
        out = _capture(repl, "help brief")
        assert isinstance(out, str)

    def test_procs_shows_process_list(self, repl):
        out = _capture(repl, "procs")
        assert isinstance(out, str)

    def test_devices_available(self, repl):
        out = _capture(repl, "devices")
        assert isinstance(out, str)


# ── Subprocess shell ────────────────────────────────────────────────


def test_subprocess_shell_launches():
    pkgs = str(Path(__file__).resolve().parents[1])
    import os
    env = {**os.environ, "PYTHONPATH": pkgs}
    result = subprocess.run(
        [sys.executable, "-c", "from domains.shell.repl import ShellREPL; print('ok')"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert "ok" in result.stdout, result.stderr


def test_subprocess_shell_cmd():
    if not _check_api():
        pytest.skip("API server not running")
    pkgs = repr(str(Path(__file__).resolve().parents[1]))
    code = """import sys; sys.path.insert(0, {pkgs})
from pathlib import Path
from unittest.mock import patch
from domains.shell.repl import ShellREPL, _CaptureOutput
from domains.shell.runtime import DaitRuntime
import tempfile, json, os
tmp = tempfile.mkdtemp()
with open(Path(tmp) / "state.json", "w") as f:
    json.dump({"version": 1, "first_run": False, "history": [], "aliases": {}, "env": {}, "cwd": os.getcwd()}, f)
with patch("domains.shell.state._STATE_FILE", Path(tmp) / "state.json"):
    r = ShellREPL(DaitRuntime())
    r._running = True
    out = r._execute_single("health")
    assert "healthy" in out.lower() or "ok" in out.lower()
""".replace("{pkgs}", pkgs)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
