"""
Tests for the Shell router endpoints.

Tests the /shell/exec endpoint.
"""

import json
import os
import sys

# Ensure the server directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from routers.shell import _get_repl
from routers.shell import router as shell_router

from apps.api.server.main import app

app.include_router(shell_router)
client = TestClient(app)


class TestShellExecEndpoint:
    """POST /shell/exec — basic command execution."""

    def test_exec_echo(self):
        resp = client.post("/shell/exec", json={"command": "echo hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "hello world" in data["output"]
        assert data["exit_code"] == 0
        assert data["elapsed_ms"] >= 0

    def test_exec_returns_elapsed_ms(self):
        resp = client.post("/shell/exec", json={"command": "echo fast"})
        data = resp.json()
        assert "elapsed_ms" in data
        assert isinstance(data["elapsed_ms"], (int, float))

    def test_exec_empty_command_rejected(self):
        resp = client.post("/shell/exec", json={"command": ""})
        assert resp.status_code == 422

    def test_exec_missing_command(self):
        resp = client.post("/shell/exec", json={})
        assert resp.status_code == 422

    def test_exec_invalid_timeout_rejected(self):
        resp = client.post("/shell/exec", json={"command": "echo x", "timeout_ms": 10})
        assert resp.status_code == 422

    def test_exec_returns_output_string(self):
        resp = client.post("/shell/exec", json={"command": "echo xyz"})
        data = resp.json()
        assert isinstance(data["output"], str)
        assert "xyz" in data["output"]

    def test_exec_exit_code_zero(self):
        resp = client.post("/shell/exec", json={"command": "echo ok"})
        assert resp.json()["exit_code"] == 0

    def test_get_repl_singleton(self):
        repl1 = _get_repl()
        repl2 = _get_repl()
        assert repl1 is repl2


class TestShellStreamEndpoint:
    """POST /shell/exec/stream — SSE streaming command execution."""

    def test_stream_returns_sse(self):
        resp = client.post("/shell/exec/stream", json={"command": "echo stream_test"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_complete_event(self):
        resp = client.post("/shell/exec/stream", json={"command": "echo hi"})
        text = resp.text
        assert "complete" in text
        assert "exit_code" in text

    def test_stream_emits_working_events(self):
        resp = client.post("/shell/exec/stream", json={"command": "echo line1"})
        text = resp.text
        assert "working" in text
        assert "line1" in text

    def test_stream_empty_command_rejected(self):
        resp = client.post("/shell/exec/stream", json={"command": ""})
        assert resp.status_code == 422

    def test_stream_data_format(self):
        resp = client.post("/shell/exec/stream", json={"command": "echo ok"})
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        for line in lines:
            payload = json.loads(line[5:].strip())
            assert "stream" in payload
            assert "phase" in payload
            assert "status" in payload
            assert "data" in payload
