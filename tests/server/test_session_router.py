"""
Tests for the session router — context store, messages, regenerate.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.session import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestSetSessionContext:
    """POST /session/{session_id}/context"""

    @patch("domains.infrastructure.session_core.SessionCore.store_context")
    def test_set_context_with_messages(self, mock_store, client):
        mock_store.return_value = {
            "status": "stored", "session_id": "sess-1", "message_count": 2,
        }
        resp = client.post("/session/sess-1/context", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["message_count"] == 2

    def test_set_context_empty(self, client):
        resp = client.post("/session/sess-1/context", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"


class TestGetSessionMessages:
    """GET /session/{session_id}/messages"""

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_get_messages(self, mock_get, client):
        mock_get.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        resp = client.get("/session/sess-1/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]["messages"]) == 2

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_get_messages_empty(self, mock_get, client):
        mock_get.return_value = []
        resp = client.get("/session/sess-1/messages")
        assert resp.status_code == 200
        assert resp.json()["data"]["messages"] == []

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_get_messages_error(self, mock_get, client):
        mock_get.side_effect = RuntimeError("storage error")
        resp = client.get("/session/sess-1/messages")
        assert resp.status_code == 500


class TestRegenerateSession:
    """POST /session/{session_id}/regenerate"""

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_regenerate_no_context(self, mock_get, client):
        mock_get.return_value = []
        resp = client.post("/session/sess-1/regenerate")
        assert resp.status_code == 200
        assert "No session context found" in resp.text

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    @patch("domains.models.provider.get_provider")
    def test_regenerate_success(self, mock_get_provider, mock_get, client):
        mock_get.return_value = [
            {"role": "user", "content": "Hello"},
        ]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "Regenerated"
            yield " response"

        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        resp = client.post("/session/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Regenerated" in resp.text

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    @patch("domains.models.provider.get_provider")
    def test_regenerate_no_provider(self, mock_get_provider, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_get_provider.return_value = None
        resp = client.post("/session/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Model not loaded" in resp.text
