"""
Tests for the session router — context store, messages, inspector, regenerate.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.session import router


def _make_app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def app():
    return _make_app()


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

    @patch("domains.infrastructure.session_core.SessionCore.store_context")
    def test_set_context_with_system_prompt(self, mock_store, client):
        mock_store.return_value = {
            "status": "stored", "session_id": "sess-2", "message_count": 1,
        }
        resp = client.post("/session/sess-2/context", json={
            "system_prompt": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["session_id"] == "sess-2"

    @patch("domains.infrastructure.session_core.SessionCore.store_context")
    def test_set_context_with_knowledge(self, mock_store, client):
        mock_store.return_value = {
            "status": "stored", "session_id": "sess-3", "message_count": 1,
        }
        resp = client.post("/session/sess-3/context", json={
            "knowledge": ["fact1", "fact2"],
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["session_id"] == "sess-3"

    @patch("domains.infrastructure.session_core.SessionCore.store_context")
    def test_set_context_store_via_kwargs_passes_messages(self, mock_store, client):
        mock_store.return_value = {"status": "stored", "message_count": 1}
        client.post("/session/sess-x/context", json={
            "messages": [{"role": "assistant", "content": "ok"}],
        })
        args, kwargs = mock_store.call_args
        assert args[0] == "sess-x"
        assert args[1][0]["role"] == "assistant"

    @patch("domains.infrastructure.session_core.SessionCore.store_context")
    def test_set_context_store_propagates_error(self, mock_store):
        from fastapi.testclient import TestClient
        client = TestClient(_make_app(), raise_server_exceptions=False)
        mock_store.side_effect = RuntimeError("disk full")
        resp = client.post("/session/sess-x/context", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 500


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

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_get_messages_returns_session_id(self, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hi"}]
        resp = client.get("/session/my-session-42/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["session_id"] == "my-session-42"

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_get_messages_long_history(self, mock_get, client):
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(50)]
        mock_get.return_value = messages
        resp = client.get("/session/sess-long/messages")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["messages"]) == 50

    def test_get_messages_wrong_method_is_405(self, client):
        resp = client.post("/session/sess-1/messages")
        assert resp.status_code == 405


class TestSessionInspector:
    """GET /session/{session_id}/inspector"""

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_inspector_returns_structure(self, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hi"}]
        resp = client.get("/session/sess-1/inspector")
        assert resp.status_code == 200
        body = resp.json()
        assert "session" in body
        assert "knowledge" in body
        assert "traits" in body
        assert "feedback" in body
        assert body["session"]["id"] == "sess-1"
        assert body["session"]["message_count"] == 1

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_inspector_empty_session(self, mock_get, client):
        mock_get.return_value = []
        resp = client.get("/session/sess-empty/inspector")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session"]["message_count"] == 0
        assert body["session"]["messages"] == []

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_inspector_respects_last_ten(self, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": f"m{i}"} for i in range(25)]
        resp = client.get("/session/sess-25/inspector")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session"]["message_count"] == 25
        assert len(body["session"]["messages"]) == 10

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    def test_inspector_has_workspace_and_modes(self, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hi"}]
        resp = client.get("/session/sess-1/inspector")
        body = resp.json()
        assert "workspace" in body
        assert "modes" in body
        assert set(("personality", "memory", "style", "task")) <= set(body["modes"])


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

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    @patch("domains.models.provider.get_provider")
    def test_regenerate_streams_multiple_tokens(self, mock_get_provider, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            for token in ["Hello", " ", "world", "!"]:
                yield token

        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        resp = client.post("/session/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Hello" in resp.text
        assert "world" in resp.text

    @patch("domains.infrastructure.session_core.SessionCore.get_messages")
    @patch("domains.models.provider.get_provider")
    def test_regenerate_stream_emits_errors_as_sse(self, mock_get_provider, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            raise RuntimeError("tokenizer broke")
            yield  # pragma: no cover — makes _stream an async generator

        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        resp = client.post("/session/sess-1/regenerate")
        assert resp.status_code == 200
        assert "error" in resp.text
        assert "tokenizer broke" in resp.text
