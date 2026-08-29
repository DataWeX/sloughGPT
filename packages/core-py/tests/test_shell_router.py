"""Tests for the shell API router (routers/shell.py).

Covers: exec, exec/stream SSE. ShellREPL is mocked throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app():
    from routers.shell import router
    app = FastAPI()
    app.include_router(router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


def _mock_repl(output="hello\nworld", exit_code=0):
    repl = MagicMock()
    repl.execute.return_value = (output, exit_code)
    return repl


class TestShellExecRequest:
    def test_valid(self):
        from routers.shell import ShellExecRequest
        req = ShellExecRequest(command="echo hi")
        assert req.command == "echo hi"
        assert req.timeout_ms == 30000

    def test_defaults(self):
        from routers.shell import ShellExecRequest
        req = ShellExecRequest(command="ls")
        assert req.timeout_ms == 30000

    def test_custom_timeout(self):
        from routers.shell import ShellExecRequest
        req = ShellExecRequest(command="ls", timeout_ms=5000)
        assert req.timeout_ms == 5000


class TestShellExecResponse:
    def test_valid(self):
        from routers.shell import ShellExecResponse
        resp = ShellExecResponse(output="ok", exit_code=0, elapsed_ms=12.5)
        assert resp.output == "ok"
        assert resp.exit_code == 0
        assert resp.elapsed_ms == 12.5


class TestExecCommand:
    @patch("routers.shell._get_repl")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_repl("hello", 0)
        client = TestClient(_app())
        resp = client.post("/shell/exec", json={"command": "echo hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "hello"
        assert body["exit_code"] == 0
        assert "elapsed_ms" in body

    @patch("routers.shell._get_repl")
    def test_nonzero_exit(self, mock_get):
        mock_get.return_value = _mock_repl("error", 1)
        client = TestClient(_app())
        resp = client.post("/shell/exec", json={"command": "bad"})
        assert resp.status_code == 200
        assert resp.json()["exit_code"] == 1

    @patch("routers.shell._get_repl")
    def test_exception_returns_500(self, mock_get):
        repl = MagicMock()
        repl.execute.side_effect = RuntimeError("boom")
        mock_get.return_value = repl
        client = TestClient(_app())
        resp = client.post("/shell/exec", json={"command": "crash"})
        assert resp.status_code == 500

    @patch("routers.shell._get_repl")
    def test_empty_output(self, mock_get):
        mock_get.return_value = _mock_repl("", 0)
        client = TestClient(_app())
        resp = client.post("/shell/exec", json={"command": "true"})
        assert resp.status_code == 200
        assert resp.json()["output"] == ""

    @patch("routers.shell._get_repl")
    def test_multiline_output(self, mock_get):
        mock_get.return_value = _mock_repl("line1\nline2\nline3", 0)
        client = TestClient(_app())
        resp = client.post("/shell/exec", json={"command": "cat file"})
        assert resp.status_code == 200
        assert "line1\nline2\nline3" in resp.json()["output"]


class TestExecStream:
    @patch("routers.shell._get_repl")
    def test_stream_complete(self, mock_get):
        mock_get.return_value = _mock_repl("a\nb", 0)
        client = TestClient(_app())
        resp = client.post("/shell/exec/stream", json={"command": "echo a; echo b"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 2
        import json
        last = json.loads(data_lines[-1].removeprefix("data: ").strip())
        assert last["status"] == "complete"
        assert last["data"]["exit_code"] == 0

    @patch("routers.shell._get_repl")
    def test_stream_error(self, mock_get):
        repl = MagicMock()
        repl.execute.side_effect = RuntimeError("fail")
        mock_get.return_value = repl
        client = TestClient(_app())
        resp = client.post("/shell/exec/stream", json={"command": "bad"})
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 1
        import json
        first = json.loads(data_lines[0].removeprefix("data: ").strip())
        assert first["status"] == "error"

    @patch("routers.shell._get_repl")
    def test_stream_yields_all_lines(self, mock_get):
        mock_get.return_value = _mock_repl("x\ny\nz", 0)
        client = TestClient(_app())
        resp = client.post("/shell/exec/stream", json={"command": "multi"})
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        import json
        streaming = [json.loads(l.removeprefix("data: ").strip()) for l in data_lines if "working" in l]
        assert len(streaming) == 3
