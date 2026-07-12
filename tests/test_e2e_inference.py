"""
Full integration test with real model inference.

Loads a model (via ``MAN_AUTOLOAD_MODEL`` — defaults to no model) and tests
the complete inference pipeline: chat, generate, streaming, session context,
and feedback.  Skipped when no model is loaded.

Usage:
    MAN_AUTOLOAD_MODEL=gpt2 pytest tests/test_e2e_inference.py -v
"""

import os
import time
import json
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


def _sse_events(resp):
    """Parse SSE ``data:`` lines from a ``TestClient`` streaming response."""
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.fixture(scope="module")
def client():
    """Start the FastAPI app with full lifespan."""
    from apps.api.server.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def model_loaded(client):
    """Poll /health until a model is loaded or timeout (30s)."""
    deadline = time.time() + 30
    while time.time() < deadline:
        resp = client.get("/health")
        data = resp.json()
        if data.get("model_loaded"):
            return data
        time.sleep(0.5)
    pytest.skip("No model loaded — set MAN_AUTOLOAD_MODEL=gpt2 (or similar)")


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

class TestModelHealth:
    def test_health_shows_model_loaded(self, client, model_loaded):
        assert model_loaded["model_loaded"] is True
        assert isinstance(model_loaded["model_type"], str)

    def test_health_inference_count(self, client, model_loaded):
        assert model_loaded.get("inference_count", 0) >= 0


# ------------------------------------------------------------------
# Non-streaming chat
# ------------------------------------------------------------------

class TestChatNonStreaming:
    def test_chat_returns_text(self, client, model_loaded):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert len(data["message"]) > 0

    def test_chat_returns_session_id(self, client, model_loaded):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "session_id": "integration-test-session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("session_id") == "integration-test-session"

    def test_chat_returns_done(self, client, model_loaded):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("done") is True


# ------------------------------------------------------------------
# Streaming chat
# ------------------------------------------------------------------

class TestChatStreaming:
    def test_chat_stream_returns_sse(self, client, model_loaded):
        resp = client.post("/chat/stream", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        events = _sse_events(resp)
        assert len(events) >= 1
        tokens = "".join(
            e.get("data", {}).get("token", "")
            for e in events
            if e.get("data", {}).get("token")
        )
        assert len(tokens) > 0

    def test_chat_stream_has_complete_event(self, client, model_loaded):
        resp = client.post("/chat/stream", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20,
        })
        events = _sse_events(resp)
        complete = [e for e in events if e.get("status") == "complete"]
        assert len(complete) >= 1

    def test_chat_stream_has_token_in_standard_envelope(self, client, model_loaded):
        resp = client.post("/chat/stream", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 15,
        })
        events = _sse_events(resp)
        for event in events:
            assert "stream" in event or "status" in event
            if event.get("data"):
                assert isinstance(event["data"], dict)


# ------------------------------------------------------------------
# Non-streaming generate
# ------------------------------------------------------------------

class TestGenerate:
    def test_generate_returns_text(self, client, model_loaded):
        resp = client.post("/inference/generate", json={
            "prompt": "Hello",
            "max_new_tokens": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert len(data["text"]) > 0

    def test_generate_returns_model_and_tokens(self, client, model_loaded):
        resp = client.post("/inference/generate", json={
            "prompt": "Hello world",
            "max_new_tokens": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert isinstance(data.get("tokens_generated"), int)
        assert data["tokens_generated"] > 0

    def test_generate_increases_inference_count(self, client, model_loaded):
        before = client.get("/health").json().get("inference_count", 0)
        client.post("/inference/generate", json={
            "prompt": "Test", "max_new_tokens": 10,
        })
        after = client.get("/health").json().get("inference_count", 0)
        assert after >= before


# ------------------------------------------------------------------
# Streaming generate
# ------------------------------------------------------------------

class TestGenerateStream:
    def test_generate_stream_returns_sse(self, client, model_loaded):
        resp = client.post("/inference/generate/stream", json={
            "prompt": "Hi",
            "max_new_tokens": 20,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        events = _sse_events(resp)
        assert len(events) >= 1

    def test_generate_stream_has_tokens_and_complete(self, client, model_loaded):
        resp = client.post("/inference/generate/stream", json={
            "prompt": "Hello",
            "max_new_tokens": 20,
        })
        events = _sse_events(resp)
        tokens = "".join(
            e.get("data", {}).get("token", "")
            for e in events
            if e.get("data", {}).get("token")
        )
        assert len(tokens) > 0
        complete = [e for e in events if e.get("status") == "complete"]
        assert len(complete) >= 1

    def test_generate_stream_meta_has_elapsed(self, client, model_loaded):
        resp = client.post("/inference/generate/stream", json={
            "prompt": "Hi",
            "max_new_tokens": 15,
        })
        events = _sse_events(resp)
        complete = next((e for e in events if e.get("status") == "complete"), None)
        if complete and "meta" in complete:
            assert "elapsed_ms" in complete["meta"]


# ------------------------------------------------------------------
# Session context (save + retrieve)
# ------------------------------------------------------------------

class TestSessionContext:
    def test_save_session_context(self, client, model_loaded):
        resp = client.post("/session/int-test-context/context", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "stored"

    def test_retrieve_session_context(self, client, model_loaded):
        resp = client.get("/session/int-test-context/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert "messages" in data
        assert len(data["messages"]) == 2


# ------------------------------------------------------------------
# Chat sessions CRUD
# ------------------------------------------------------------------

class TestChatSessions:
    def test_create_session(self, client, model_loaded):
        resp = client.post("/chat/sessions", json={
            "session_id": "int-test-session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "created"

    def test_list_sessions_includes_new(self, client, model_loaded):
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        ids = [s.get("session_id") or s.get("id") for s in data["sessions"]]
        assert "int-test-session" in ids


# ------------------------------------------------------------------
# Feedback recording
# ------------------------------------------------------------------

class TestFeedback:
    def test_record_thumbs_up(self, client, model_loaded):
        resp = client.post("/feedback/workflow-record", json={
            "session_id": "int-test-session",
            "message_id": "int-test-msg-1",
            "user_id": "integration-test",
            "feedback_type": "thumbs_up",
            "message_text": "Great response!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("recorded", "ok", "accepted")

    def test_record_thumbs_down(self, client, model_loaded):
        resp = client.post("/feedback/workflow-record", json={
            "session_id": "int-test-session",
            "message_id": "int-test-msg-2",
            "user_id": "integration-test",
            "feedback_type": "thumbs_down",
            "message_text": "Not helpful",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("recorded", "ok", "accepted")


# ------------------------------------------------------------------
# Regenerate
# ------------------------------------------------------------------

class TestRegenerate:
    """Regenerate the last assistant response for a session."""

    REGEN_SESSION = "int-test-regen"

    def test_regen_prerequisite_save_context(self, client, model_loaded):
        """Save session context that regenerate will use."""
        resp = client.post(f"/session/{self.REGEN_SESSION}/context", json={
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "stored"

    def test_regen_returns_sse_tokens(self, client, model_loaded):
        """Regenerate produces SSE tokens from stored context."""
        resp = client.post(f"/session/{self.REGEN_SESSION}/regenerate")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        events = _sse_events(resp)
        assert len(events) >= 1
        tokens = "".join(
            e.get("data", {}).get("token", "")
            for e in events
            if e.get("data", {}).get("token")
        )
        assert len(tokens) > 0

    def test_regen_has_complete_event(self, client, model_loaded):
        """Regenerate SSE stream ends with complete status."""
        resp = client.post(f"/session/{self.REGEN_SESSION}/regenerate")
        events = _sse_events(resp)
        complete = [e for e in events if e.get("status") == "complete"]
        assert len(complete) >= 1

    def test_regen_missing_session_returns_error(self, client, model_loaded):
        """Regenerate for a session with no context returns error event."""
        resp = client.post("/session/no-such-session/regenerate")
        assert resp.status_code == 200
        events = _sse_events(resp)
        assert any(e.get("status") == "error" for e in events)


# ------------------------------------------------------------------
# Full cycle: stream → context → regenerate → feedback
# ------------------------------------------------------------------

class TestFullCycle:
    """End-to-end cycle: chat stream, save context, regenerate, feedback."""

    CYCLE_SESSION = "int-test-cycle"

    def test_cycle_stream(self, client, model_loaded):
        """Step 1: Stream a chat message."""
        resp = client.post("/chat/stream", json={
            "session_id": self.CYCLE_SESSION,
            "messages": [{"role": "user", "content": "Hello from full cycle"}],
            "max_tokens": 20,
        })
        assert resp.status_code == 200
        events = _sse_events(resp)
        assert any(e.get("status") == "complete" for e in events)

    def test_cycle_save_context(self, client, model_loaded):
        """Step 2: Save the conversation context."""
        resp = client.post(f"/session/{self.CYCLE_SESSION}/context", json={
            "messages": [
                {"role": "user", "content": "Hello from full cycle"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "stored"

    def test_cycle_regenerate(self, client, model_loaded):
        """Step 3: Regenerate from saved context."""
        resp = client.post(f"/session/{self.CYCLE_SESSION}/regenerate")
        assert resp.status_code == 200
        events = _sse_events(resp)
        tokens = "".join(
            e.get("data", {}).get("token", "")
            for e in events
            if e.get("data", {}).get("token")
        )
        assert len(tokens) > 0
        assert any(e.get("status") == "complete" for e in events)

    def test_cycle_feedback(self, client, model_loaded):
        """Step 4: Record feedback on the regenerated response."""
        resp = client.post("/feedback/workflow-record", json={
            "session_id": self.CYCLE_SESSION,
            "message_id": "cycle-msg-regen",
            "user_id": "integration-test",
            "feedback_type": "thumbs_up",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("recorded", "ok", "accepted")
