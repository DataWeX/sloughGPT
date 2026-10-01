"""
Tests for the Chat router endpoints (apps/api/server/routers/chat.py).

Covers all 7 routes: POST /chat, /chat/stream, /chat/{session_id}/regenerate,
/chat/{session_id}/cancel, GET /chat/active, /chat/health, /chat/addons —
happy path + 404/405/422 + auth edge.

NOTE: POST /chat and POST /chat/stream are ALSO claimed by InferenceRouter
(prefix "") which is registered before ChatRouter in get_all_routers(), so on
the shared app the inference handler wins (first route match). These tests
mount ONLY the ChatRouter on an isolated app (house pattern:
test_tokens_router.py) so its handlers are exercised directly.

The chat manager is faked: these tests exercise the HTTP contract (envelope,
SSE framing, validation, auth), not model inference.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from routers.chat import router as chat_router

app = FastAPI()
register_app_error_handler(app)
app.include_router(chat_router)
client = TestClient(app)


def _data(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


class _FakeManager:
    """Stands in for domain.chat.get_chat_manager()."""

    async def respond(self, **kwargs):
        return SimpleNamespace(
            text="hello from fake",
            session_id=kwargs.get("session_id", "default"),
            tokens_generated=2,
            duration_ms=5,
            usage_tokens=7,
        )

    async def stream(self, **kwargs):
        yield "Hello"
        yield " world"

    def cancel_session(self, session_id):
        return True

    def active_sessions(self):
        return ["s1"]

    def health(self):
        return {"provider": "fake", "ok": True}

    def addons(self):
        return {"processors": []}

    def last_usage(self):
        return {"total": 5}


@pytest.fixture(autouse=True)
def _fake_chat_manager(monkeypatch):
    monkeypatch.setattr("domain.chat.get_chat_manager", lambda: _FakeManager())


def _msg_body():
    return {"messages": [{"role": "user", "content": "hi"}]}


class TestChatRespond:
    """POST /chat — non-streaming response."""

    def test_respond_happy_path(self):
        resp = client.post("/chat", json=_msg_body())
        assert resp.status_code == 200
        data = _data(resp)
        assert data["text"] == "hello from fake"
        assert data["session_id"] == "default"
        assert data["tokens_generated"] == 2

    def test_respond_forwards_session_id(self):
        resp = client.post("/chat", json={**_msg_body(), "session_id": "s42"})
        assert resp.status_code == 200
        assert _data(resp)["session_id"] == "s42"

    def test_respond_missing_messages_is_422(self):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_unknown_path_is_404(self):
        resp = client.get("/chat/definitely-not-a-route")
        assert resp.status_code == 404

    def test_wrong_method_is_405(self):
        # /chat/health exists as GET only
        resp = client.post("/chat/health")
        assert resp.status_code == 405


class TestChatStream:
    """POST /chat/stream — SSE streaming response."""

    def test_stream_returns_sse(self):
        resp = client.post("/chat/stream", json=_msg_body())
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_tokens(self):
        resp = client.post("/chat/stream", json=_msg_body())
        assert "Hello" in resp.text
        assert "data:" in resp.text

    def test_stream_missing_messages_is_422(self):
        resp = client.post("/chat/stream", json={})
        assert resp.status_code == 422


class TestChatSessionOps:
    """POST /chat/{session_id}/regenerate, POST /chat/{session_id}/cancel."""

    def test_regenerate_returns_sse(self):
        resp = client.post("/chat/s1/regenerate")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "REGENERATE" in resp.text

    def test_cancel_happy_path(self):
        resp = client.post("/chat/s1/cancel")
        assert resp.status_code == 200
        assert _data(resp)["cancelled"] is True


class TestChatReadModels:
    """GET /chat/active, /chat/health, /chat/addons."""

    def test_active_sessions(self):
        resp = client.get("/chat/active")
        assert resp.status_code == 200
        assert _data(resp)["sessions"] == ["s1"]

    def test_health(self):
        resp = client.get("/chat/health")
        assert resp.status_code == 200
        assert _data(resp)["provider"] == "fake"

    def test_addons(self):
        resp = client.get("/chat/addons")
        assert resp.status_code == 200
        assert _data(resp)["processors"] == []

    def test_unknown_path_is_404(self):
        resp = client.get("/chat/nope/nope")
        assert resp.status_code == 404


class TestChatAuth:
    """Auth edge: enforced only when SLO_AUTH_REQUIRED=true."""

    def test_auth_disabled_allows_anonymous(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "false")
        resp = client.get("/chat/active")
        assert resp.status_code == 200

    def test_auth_enabled_rejects_missing_token(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
        resp = client.get("/chat/active")
        assert resp.status_code == 401

    def test_auth_enabled_rejects_respond_without_token(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
        resp = client.post("/chat", json=_msg_body())
        assert resp.status_code == 401
