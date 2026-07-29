"""
Tests for the inference router — chat, generate, sessions.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.inference import _instance as _inference_router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(_inference_router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


MOCK_STATE = MagicMock()
MOCK_STATE.model = MagicMock()
MOCK_STATE.model_type = "gpt2"
MOCK_STATE.model_request_logger = None
MOCK_STATE.checkpoint = None
MOCK_STATE.soul_engine = None
MOCK_STATE.current_soul = None

MOCK_STARTUP = {"phase": "ready", "step": 8, "total": 8, "message": "Ready"}


class TestGenerate:
    """POST /inference/generate"""

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.models.provider.get_provider")
    def test_generate_success(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(return_value="Hello world")
        mock_get_provider.return_value = mock_prov

        resp = client.post("/inference/generate", json={"prompt": "Hi", "max_new_tokens": 16})
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "Hello world"
        assert body["model"] == "gpt2"
        assert body["tokens_generated"] == 2

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.models.provider.get_provider")
    def test_generate_provider_failure(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(side_effect=RuntimeError("inference failed"))
        mock_get_provider.return_value = mock_prov

        resp = client.post("/inference/generate", json={"prompt": "Hi"})
        assert resp.status_code == 500

    def test_generate_no_model(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hi"})
        assert resp.status_code == 503


class TestGenerateStream:
    """POST /inference/generate/stream"""

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.models.provider.get_provider")
    def test_generate_stream_success(self, mock_get_provider, client):
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "Hello"
            yield " world"

        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        resp = client.post("/inference/generate/stream", json={"prompt": "Hi", "max_new_tokens": 16})
        assert resp.status_code == 200
        text = resp.text
        assert "Hello" in text
        assert "world" in text

    def test_generate_stream_no_model(self, client):
        resp = client.post("/inference/generate/stream", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert "error" in resp.text


class TestChat:
    """POST /chat"""

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.get_chat_domain")
    def test_chat_success(self, mock_get_domain, client):
        mock_domain = MagicMock()
        mock_domain.respond = AsyncMock(return_value=MagicMock(
            text="Hello back", session_id="sess-1", done=True,
        ))
        mock_get_domain.return_value = mock_domain

        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "Hello back"
        assert body["session_id"] == "sess-1"

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    def test_chat_no_message(self, client):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 400

    def test_chat_no_model(self, client):
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
        assert resp.status_code == 503


class TestListSessions:
    """GET /chat/sessions"""

    @patch.object(_inference_router, "_build_session_cache")
    def test_list_sessions(self, mock_build, client):
        mock_build.return_value = [
            {"id": "s1", "name": "Chat 1", "messages": [], "updated_at": "2026-01-01"},
            {"id": "s2", "name": "Chat 2", "messages": [], "updated_at": "2026-01-02"},
        ]
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2

    @patch.object(_inference_router, "_build_session_cache")
    def test_list_sessions_empty(self, mock_build, client):
        mock_build.return_value = []
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        assert resp.json()["data"] == []
