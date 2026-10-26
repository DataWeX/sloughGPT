"""
Tests for the Shell router endpoints.

Tests the /shell/exec endpoint.
"""

import json
import os
import sys
import time

import pytest

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


# ── B3: wire hygiene at the seam ────────────────────────────────────────────


class _FakeRepl:
    """Scriptable repl for endpoint-level tests (sanitization + streaming)."""

    def __init__(self, output="", exit_code=0, chunks=(), delay=0.0):
        self.output = output
        self.exit_code = exit_code
        self.chunks = list(chunks)
        self.delay = delay
        self.executed = False

    def execute(self, line, *, on_write=None):
        self.executed = True
        for i, chunk in enumerate(self.chunks):
            if i and self.delay:
                time.sleep(self.delay)
            if on_write:
                on_write(chunk)
        return self.output, self.exit_code


@pytest.fixture
def fake_repl():
    import routers.shell as m

    old = m._repl
    repl = _FakeRepl()
    m._repl = repl
    yield repl
    m._repl = old


class TestSanitizeLine:
    def test_exists(self):
        from routers import shell as m

        assert getattr(m, "_sanitize_line", None) is not None, (
            "routers.shell._sanitize_line missing"
        )

    def test_strips_csi_sequences(self):
        from routers import shell as m

        assert m._sanitize_line("\x1b[32mgreen\x1b[0m text") == "green text"

    def test_carriage_return_keeps_last_segment(self):
        from routers import shell as m

        line = "\r  ⠋ Fetching models\x1b[K\r\x1b[K  Model  Type  Status"
        assert m._sanitize_line(line) == "  Model  Type  Status"

    def test_drops_other_control_chars_keeps_tab(self):
        from routers import shell as m

        assert m._sanitize_line("a\x07b\tc\x00d") == "ab\tcd"

    def test_plain_text_untouched(self):
        from routers import shell as m

        assert m._sanitize_line("  plain line  ") == "  plain line  "


class TestExecSanitized:
    def test_exec_output_sanitized(self, fake_repl):
        fake_repl.output = "\r\x1b[31mred\x1b[0m\nok\n"
        resp = client.post("/shell/exec", json={"command": "x"})
        assert resp.status_code == 200
        assert resp.json()["output"] == "red\nok\n"


class TestStreamSanitized:
    def test_stream_lines_sanitized(self, fake_repl):
        fake_repl.chunks = [
            "\r  ⠋ Fetching models\x1b[K\r\x1b[K  Model  Type  Status\n",
            "\x1b[32mQwen  cpu  loaded\x1b[0m\n",
        ]
        resp = client.post("/shell/exec/stream", json={"command": "models"})
        assert resp.status_code == 200
        working = []
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                if ev.get("status") == "working":
                    working.append(ev["data"].get("line"))
        assert "  Model  Type  Status" in working
        assert "Qwen  cpu  loaded" in working
        assert not any("\x1b" in (x or "") for x in working)


class TestRawFlag:
    """raw=true passes escape sequences through for the terminal engine.

    Default stays sanitized (TestStreamSanitized pins that); raw is the
    explicit opt-out consumed by useShell once the engine interprets SGR,
    cursor and clear-screen sequences client-side.
    """

    @staticmethod
    def _working_lines(text: str) -> list:
        out = []
        for line in text.split("\n"):
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                if ev.get("status") == "working":
                    out.append(ev["data"].get("line"))
        return out

    def test_stream_raw_preserves_escapes(self, fake_repl):
        fake_repl.chunks = ["\x1b[2J\x1b[1;32mready\x1b[0m\n"]
        resp = client.post("/shell/exec/stream", json={"command": "x", "raw": True})
        assert resp.status_code == 200
        working = self._working_lines(resp.text)
        assert any("\x1b[2J" in (x or "") for x in working), (
            "raw=true stripped the clear-screen sequence"
        )

    def test_stream_raw_false_stays_sanitized(self, fake_repl):
        fake_repl.chunks = ["\x1b[31mred\x1b[0m\n"]
        resp = client.post("/shell/exec/stream", json={"command": "x", "raw": False})
        assert resp.status_code == 200
        working = self._working_lines(resp.text)
        assert working and not any("\x1b" in (x or "") for x in working)

    def test_exec_raw_preserves_escapes(self, fake_repl):
        fake_repl.output = "\x1b[2Jdone\x1b[0m\n"
        resp = client.post("/shell/exec", json={"command": "x", "raw": True})
        assert resp.status_code == 200
        assert "\x1b[2J" in resp.json()["output"]


# ── B4: genuinely incremental streaming ─────────────────────────────────────


class TestStreamIncremental:
    def test_lines_arrive_during_execution(self, fake_repl):
        """Unit-drive the generator: the HTTP TestClient buffers whole
        bodies, so incremental delivery must be proven at the source."""
        from routers.shell import _stream_events

        fake_repl.chunks = ["early line\n", "late line\n"]
        fake_repl.delay = 0.8
        t0 = time.monotonic()
        t_early = None
        seen_complete = False
        for payload in _stream_events(fake_repl, "slow", t0):
            ev = json.loads(payload[5:])
            if ev.get("status") == "working" and ev["data"].get("line") == "early line":
                t_early = time.monotonic() - t0
            if ev.get("status") == "complete":
                seen_complete = True
        assert t_early is not None, "early line never streamed"
        assert t_early < 0.6, (
            f"line arrived after execute finished ({t_early:.2f}s) — batch, not streaming"
        )
        assert seen_complete

    def test_endpoint_event_order(self, fake_repl):
        """The endpoint wires the generator through: lines before complete."""
        fake_repl.chunks = ["one\n", "two\n"]
        resp = client.post("/shell/exec/stream", json={"command": "x"})
        assert resp.status_code == 200
        working, last = [], None
        for line in resp.text.split("\n"):
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                if ev.get("status") == "working":
                    working.append(ev["data"].get("line"))
                last = ev.get("status")
        assert working == ["one", "two"]
        assert last == "complete"
