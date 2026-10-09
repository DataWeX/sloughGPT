"""
Tests for the chat router regenerate endpoint — `POST /chat/{session_id}/regenerate`.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.chat import router
from domain.chat import reset_chat_manager


def _make_app():
    _app = FastAPI()
    _app.include_router(router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(_app)
    return _app


@pytest.fixture(autouse=True)
def _fresh_chat_manager():
    reset_chat_manager()
    yield
    reset_chat_manager()


@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestRegenerateChat:
    """POST /chat/{session_id}/regenerate"""

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    def test_regenerate_no_context(self, mock_get, client):
        mock_get.return_value = []
        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "No session context found" in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models._internal.provider.get_provider")
    @patch("domain.models.get_provider")
    def test_regenerate_success(self, mock_facade, mock_internal, mock_get, client):
        mock_get.return_value = [
            {"role": "user", "content": "Hello"},
        ]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "Regenerated"
            yield " response"

        mock_prov.chat_stream = _stream
        mock_facade.return_value = mock_prov
        mock_internal.return_value = mock_prov

        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Regenerated" in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models.get_provider")
    def test_regenerate_no_provider(self, mock_get_provider, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_get_provider.return_value = None
        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Model not loaded" in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models._internal.provider.get_provider")
    @patch("domain.models.get_provider")
    def test_regenerate_streams_multiple_tokens(self, mock_facade, mock_internal, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            for token in ["Hello", " ", "world", "!"]:
                yield token

        mock_prov.chat_stream = _stream
        mock_facade.return_value = mock_prov
        mock_internal.return_value = mock_prov

        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Hello" in resp.text
        assert "world" in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models._internal.provider.get_provider")
    @patch("domain.models.get_provider")
    def test_regenerate_stream_emits_errors_as_sse(self, mock_facade, mock_internal, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            raise RuntimeError("tokenizer broke")
            yield  # pragma: no cover — makes _stream an async generator

        mock_prov.chat_stream = _stream
        mock_facade.return_value = mock_prov
        mock_internal.return_value = mock_prov

        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "error" in resp.text
        assert "tokenizer broke" in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models._internal.provider.get_provider")
    @patch("domain.models.get_provider")
    @patch("fastapi.Request.is_disconnected")
    def test_regenerate_stops_on_disconnect(
        self, mock_disc, mock_facade, mock_internal, mock_get, client
    ):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "First"
            yield "Second"

        mock_prov.chat_stream = _stream
        mock_facade.return_value = mock_prov
        mock_internal.return_value = mock_prov
        mock_disc.side_effect = [False, True]

        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "Regenerating" in resp.text
        assert "First" in resp.text
        assert "Second" not in resp.text

    @patch("domain.infrastructure._internal.session_core.SessionCore.get_messages")
    @patch("domain.models._internal.provider.get_provider")
    @patch("domain.models.get_provider")
    def test_regenerate_emits_thinking_event(self, mock_facade, mock_internal, mock_get, client):
        mock_get.return_value = [{"role": "user", "content": "Hello"}]
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "Token"

        mock_prov.chat_stream = _stream
        mock_facade.return_value = mock_prov
        mock_internal.return_value = mock_prov
        resp = client.post("/chat/sess-1/regenerate")
        assert resp.status_code == 200
        assert "thinking" in resp.text
        assert "Token" in resp.text

    def test_regenerate_get_405(self, client):
        resp = client.get("/chat/sess-1/regenerate")
        assert resp.status_code == 405
