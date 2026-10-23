"""
Tests for the chat router regenerate endpoint — `POST /chat/{session_id}/regenerate`.
"""

import json
from types import SimpleNamespace
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
    def test_regenerate_stream_emits_errors_as_sse(
        self, mock_facade, mock_internal, mock_get, client
    ):
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


class TestStreamReconnectReplay:
    """POST /chat reconnect replay — session + message-hash keyed cache (card 8125c86e).

    Cache isolation is global (tests/conftest.py autouse fixture).
    """

    @staticmethod
    def _body(content="Hello", session_id="sess-1"):
        return {
            "session_id": session_id,
            "messages": [{"role": "user", "content": content}],
        }

    @staticmethod
    def _fake_manager(tokens, *, result=None, blow_up=False):
        """ChatManager stand-in: streams tokens, then the final TurnResult."""
        mgr = MagicMock()

        async def _gen(session_id, messages, stream, text_of):
            if blow_up:
                raise AssertionError("engine must not run (replay expected)")
            for t in tokens:
                yield t
            if result is not None:
                yield result

        mgr.generate_stream_tracked = MagicMock(side_effect=_gen)
        mgr.cancel_session = MagicMock(return_value=1)
        return mgr

    @staticmethod
    def _events(resp):
        out = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    out.append(json.loads(line[6:]))
                except ValueError:
                    pass
        return out

    @classmethod
    def _text(cls, resp):
        """Concatenated streamed tokens — frames split tokens, prose does not."""
        return "".join(e.get("data", {}).get("token", "") for e in cls._events(resp))

    @patch("domain.chat.get_chat_manager")
    def test_completed_turn_replays_from_cache(self, mock_gcm, client):
        from infrastructure.chat_response_cache import (
            hash_messages,
            set_chat_response_cache,
        )

        body = self._body()
        set_chat_response_cache(
            "sess-1",
            {
                "tokens": ["The ans", "wer."],
                "complete": True,
                "message_hash": hash_messages(body["messages"]),
            },
        )
        mgr = self._fake_manager([], blow_up=True)
        mock_gcm.return_value = mgr

        resp = client.post("/chat", json=body)
        assert resp.status_code == 200
        assert "The answer." in resp.text
        mgr.generate_stream_tracked.assert_not_called()
        mgr.cancel_session.assert_not_called()

        done = [e for e in self._events(resp) if e.get("status") == "complete"]
        assert done, "replay must close with the done frame"
        assert done[-1]["meta"].get("cached") is True
        assert done[-1]["meta"].get("usage_tokens") == 2

    @patch("domain.chat.get_chat_manager")
    def test_completed_turn_populates_cache(self, mock_gcm, client):
        from infrastructure.chat_response_cache import get_chat_response_cache

        body = self._body()
        mgr = self._fake_manager(
            ["The ", "answer."],
            result=SimpleNamespace(usage=3, model="m", finish_reason="stop", elapsed_ms=5.0),
        )
        mock_gcm.return_value = mgr

        resp = client.post("/chat", json=body)
        assert resp.status_code == 200
        assert self._text(resp) == "The answer."
        done = [e for e in self._events(resp) if e.get("status") == "complete"]
        assert done, "normal completion must emit the done frame"
        assert done[-1]["meta"].get("cached") is None

        entry = get_chat_response_cache("sess-1")
        assert entry is not None
        assert entry["complete"] is True
        assert "".join(entry["tokens"]) == "The answer."

    @patch("domain.chat.get_chat_manager")
    def test_new_turn_bypasses_stale_cache(self, mock_gcm, client):
        from infrastructure.chat_response_cache import (
            get_chat_response_cache,
            hash_messages,
            set_chat_response_cache,
        )

        stale = self._body(content="Old question")
        set_chat_response_cache(
            "sess-1",
            {
                "tokens": ["Old answer."],
                "complete": True,
                "message_hash": hash_messages(stale["messages"]),
            },
        )
        body = self._body(content="New question")
        mgr = self._fake_manager(
            ["Fresh ", "reply."],
            result=SimpleNamespace(usage=3, model="m", finish_reason="stop", elapsed_ms=5.0),
        )
        mock_gcm.return_value = mgr

        resp = client.post("/chat", json=body)
        assert resp.status_code == 200
        streamed = self._text(resp)
        assert "Old answer." not in streamed
        assert streamed == "Fresh reply."
        mgr.generate_stream_tracked.assert_called_once()

        entry = get_chat_response_cache("sess-1")
        assert entry["complete"] is True
        assert entry["message_hash"] == hash_messages(body["messages"])
        assert "".join(entry["tokens"]) == "Fresh reply."

    @patch("fastapi.Request.is_disconnected")
    @patch("domain.chat.get_chat_manager")
    def test_disconnect_caches_partial_and_cancels(self, mock_gcm, mock_disc, client):
        from infrastructure.chat_response_cache import get_chat_response_cache

        body = self._body()
        mgr = self._fake_manager(
            ["First", "Second"],
            result=SimpleNamespace(usage=3, model="m", finish_reason="stop", elapsed_ms=5.0),
        )
        mock_gcm.return_value = mgr
        mock_disc.side_effect = [False, True]

        resp = client.post("/chat", json=body)
        assert resp.status_code == 200
        assert "First" in resp.text
        assert "Second" not in resp.text
        mgr.cancel_session.assert_called_once_with("sess-1")

        entry = get_chat_response_cache("sess-1")
        assert entry is not None
        assert entry["complete"] is False
        assert entry["tokens"] == ["First"]

    @patch("domain.chat.get_chat_manager")
    def test_partial_cache_regenerates_instead_of_replaying(self, mock_gcm, client):
        from infrastructure.chat_response_cache import (
            get_chat_response_cache,
            hash_messages,
            set_chat_response_cache,
        )

        body = self._body()
        set_chat_response_cache(
            "sess-1",
            {
                "tokens": ["Half of it."],
                "complete": False,
                "message_hash": hash_messages(body["messages"]),
            },
        )
        mgr = self._fake_manager(
            ["Full ", "answer."],
            result=SimpleNamespace(usage=3, model="m", finish_reason="stop", elapsed_ms=5.0),
        )
        mock_gcm.return_value = mgr

        resp = client.post("/chat", json=body)
        assert resp.status_code == 200
        # The partial cache must not be replayed (infinite retry loop) — the
        # turn regenerates and the retry gets a real done frame.
        assert self._text(resp) == "Full answer."
        mgr.generate_stream_tracked.assert_called_once()
        done = [e for e in self._events(resp) if e.get("status") == "complete"]
        assert done, "regenerated turn must close with the done frame"

        entry = get_chat_response_cache("sess-1")
        assert entry["complete"] is True
        assert "".join(entry["tokens"]) == "Full answer."
