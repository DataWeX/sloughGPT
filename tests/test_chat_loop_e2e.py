"""
E2E test for the full chat loop — session, stream, regenerate, feedback.

Uses the real FastAPI app with targeted mocks for the provider pipeline.
Only the regenerate endpoint requires real model inference, so it's tested
separately with the "model not loaded" error path.
"""
import os
os.environ.setdefault("MAN_AUTOLOAD_MODEL", "")
os.environ.setdefault("MAN_AUTO_WORKFLOW", "false")
os.environ.setdefault("MAN_HEALTH_MONITOR", "false")
os.environ.setdefault("MAN_WATCHDOG", "false")

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


class AsyncIteratorMock:
    """Async iterator yielding SSE tokens on-demand."""
    def __init__(self, tokens):
        self._tokens = list(tokens)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        return self._tokens.pop(0)


@pytest.fixture(scope="module")
def client():
    from apps.api.server.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_chat_deps():
    """Mock provider pipeline + learner + session to avoid HF model loading."""
    import startup_progress
    startup_progress.STARTUP_PHASE.update(phase="ready", message="Server ready (test mock)")
    in_memory_sessions: dict = {}

    def fake_store_context(sid, msgs):
        in_memory_sessions.setdefault(sid, {})["messages"] = list(msgs)
        return {"status": "stored", "session_id": sid, "message_count": len(msgs)}

    def fake_get_messages(sid):
        return (in_memory_sessions.get(sid) or {}).get("messages", [])

    provider = MagicMock()
    provider.chat = AsyncMock(return_value="Hello! How can I help?")
    provider.chat_stream = MagicMock(
        return_value=AsyncIteratorMock(["Hello!", " How", " can", " I", " help", "?"])
    )
    provider.model_id = "mock-model"

    model_ctrl = MagicMock()
    model_ctrl._hf_model = MagicMock()
    model_ctrl._tokenizer = MagicMock()

    with patch("domains.models.provider.get_provider", return_value=provider), \
         patch("routers.inference._enrich_knowledge",
               return_value={"source": "none", "facts": [], "topics": []}), \
         patch("domains.infrastructure.session_core.SessionCore.store_context",
               side_effect=fake_store_context), \
         patch("domains.infrastructure.session_core.SessionCore.get_messages",
               side_effect=fake_get_messages), \
         patch("controllers.feedback.get_feedback_controller") as mock_fb_ctrl, \
         patch("controllers.models.get_models_controller", return_value=model_ctrl), \
         patch("domains.learner.get_learner"):
        mock_fb = MagicMock()
        mock_fb.record_feedback = MagicMock(return_value={
            "status": "recorded", "feedback_id": "mock-fb-1"
        })
        mock_fb_ctrl.return_value = mock_fb
        yield {
            "provider": provider,
            "sessions": in_memory_sessions,
            "model_ctrl": model_ctrl,
        }


class TestChatLoopE2E:
    """Full chat loop: session -> stream -> context -> regenerate -> feedback."""

    def test_create_session(self, client):
        """Create a chat session."""
        resp = client.post("/chat/sessions", json={"name": "test-session"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "created" or "session_id" in body

    def test_chat_stream_returns_sse(self, client, mock_chat_deps):
        """Send a message via /chat/stream and collect SSE events."""
        resp = client.post(
            "/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hi there"}],
                "max_tokens": 50,
                "use_context_core": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) >= 2
        tokens = "".join(
            e["data"]["token"] for e in events
            if e.get("data", {}).get("token")
        )
        assert "Hello! How can I help?" in tokens
        assert any(e["status"] == "complete" for e in events), "Missing complete event"

        provider = mock_chat_deps["provider"]
        provider.chat_stream.assert_called_once()

    def test_session_context(self, client, mock_chat_deps):
        """Save and retrieve session context for regeneration."""
        session_id = "e2e-test-session"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        resp = client.post(
            f"/session/{session_id}/context",
            json={"messages": messages},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "stored"

        resp = client.get(f"/session/{session_id}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Hello"

    def test_regenerate_model_not_loaded(self, client, mock_chat_deps):
        """Regenerate with model set to None returns error."""
        with patch("controllers.models.get_models_controller") as mc:
            ctrl = MagicMock()
            ctrl._hf_model = None
            ctrl._tokenizer = None
            mc.return_value = ctrl

            resp = client.post(
                "/session/no-model-test/regenerate",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
            assert resp.status_code == 200
            events = []
            for line in resp.text.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
            assert any(e.get("status") == "error" for e in events)

    def test_record_feedback(self, client, mock_chat_deps):
        """Record feedback via workflow-record endpoint."""
        resp = client.post(
            "/feedback/workflow-record",
            json={
                "session_id": "fb-test-session",
                "message_id": "msg-1",
                "user_id": "test-user",
                "feedback_type": "thumbs_up",
                "message_text": "Great response!",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("recorded", "ok", "accepted")

    def test_full_cycle(self, client, mock_chat_deps):
        """Complete chat cycle: stream -> context -> feedback."""
        session_id = "full-cycle-test"

        resp = client.post(
            "/chat/stream",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Hello world"}],
                "use_context_core": False,
            },
        )
        assert resp.status_code == 200
        stream_events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                stream_events.append(json.loads(line[6:]))
        assert any(e["status"] == "complete" for e in stream_events)

        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        resp = client.post(f"/session/{session_id}/context", json={"messages": messages})
        assert resp.status_code == 200

        resp = client.post(
            "/feedback/workflow-record",
            json={
                "session_id": session_id,
                "message_id": "regen-msg-1",
                "user_id": "test-user",
                "feedback_type": "thumbs_up",
            },
        )
        assert resp.status_code == 200
