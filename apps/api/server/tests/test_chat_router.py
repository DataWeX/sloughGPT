"""
Tests for the Chat router endpoints (apps/api/server/routers/chat.py).

Covers all 7 routes: POST /chat, /chat/stream, /chat/{session_id}/regenerate,
/chat/{session_id}/cancel, GET /chat/active, /chat/health, /chat/addons —
happy path + 404/405/422 + auth edge.

POST /chat + POST /chat/stream are owned by THIS router (InferenceRouter's
duplicate registrations were removed; scripts/check_docs_api_parity.py fails
on any duplicate path+method). Both delegate to the inference kernel
(`routers.inference._instance`), which tests/server/test_inference_router.py
exercises end-to-end — here the kernel seam is faked so these stay a fast
HTTP-contract check (validation, SSE framing, auth).

The chat manager is faked for the session routes (regenerate, cancel, read
models).
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from routers.chat import router as chat_router
from routers.inference import ChatResponse

app = FastAPI()
register_app_error_handler(app)
app.include_router(chat_router)
client = TestClient(app)


def _data(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


class _FakeManager:
    """Stands in for domain.chat.get_chat_manager() — session routes only."""

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


class _FakeKernel:
    """Stands in for routers.inference._instance (the chat kernel)."""

    async def chat(self, req, auth_user=None):
        return ChatResponse(
            message="hello from fake",
            session_id=req.session_id or "default",
            done=True,
        )

    async def chat_stream(self, req, http_request, auth_user=None):
        frames = (
            'data: {"stream":"chat","phase":"STREAMING","status":"working",'
            '"data":{"token":"Hello"}}\n\n'
            'data: {"stream":"chat","phase":"STREAMING","status":"working",'
            '"data":{"token":" world"}}\n\n'
            'data: {"stream":"chat","phase":"STREAMING","status":"complete","data":{}}\n\n'
        )
        return StreamingResponse(iter([frames]), media_type="text/event-stream")


@pytest.fixture(autouse=True)
def _fake_chat_manager(monkeypatch):
    monkeypatch.setattr("domain.chat.get_chat_manager", lambda: _FakeManager())
    monkeypatch.setattr("routers.chat._chat_kernel", _FakeKernel())


def _msg_body():
    return {"messages": [{"role": "user", "content": "hi"}]}


class TestChatRespond:
    """POST /chat — non-streaming response (delegates to the inference kernel)."""

    def test_respond_happy_path(self):
        resp = client.post("/chat", json=_msg_body())
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "hello from fake"
        assert body["session_id"] == "default"
        assert body["done"] is True

    def test_respond_forwards_session_id(self):
        resp = client.post("/chat", json={**_msg_body(), "session_id": "s42"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "s42"

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
