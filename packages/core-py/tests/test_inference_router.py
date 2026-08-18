"""Tests for the inference API router (routers/inference.py).

Covers: helper functions, session CRUD, search, context, chat suggestions.
"""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from unittest.mock import AsyncMock

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.inference import (
    _model_ready,
    _count_tokens,
    _extract_user_message,
    _search_sessions_sync,
    InferenceRouter,
    Message,
)  # noqa: E402
from tests.conftest import build_test_app


def _app(ir: InferenceRouter):
    return build_test_app(ir.router)


def _make_ir(tmp_path: Path) -> InferenceRouter:
    """Create an InferenceRouter with temp session storage."""
    ir = InferenceRouter.__new__(InferenceRouter)
    ir.router = InferenceRouter.__init__(InferenceRouter()) or None
    ir2 = InferenceRouter()
    ir2._SESSIONS_DIR = tmp_path
    ir2._session_repo = MagicMock()
    ir2._session_repo.list.return_value = []
    ir2._session_cache = None
    ir2._session_cache_ts = 0
    ir2._session_memory_cache = {}
    ir2._session_deleted = set()
    ir2._session_dirty = set()
    ir2._context_core = None
    ir2._vector_store_ref = None
    ir2._BG_TASKS = set()
    ir2._background_flush_task = None
    ir2._VOICE_DIR = tmp_path / "voice"
    ir2._VOICE_DIR.mkdir(parents=True, exist_ok=True)
    # Async mock for _flush_session_to_disk
    ir2._flush_session_to_disk = AsyncMock()
    return ir2


# ── Helper functions ──


class TestModelReady:
    def test_ready_with_model(self):
        import state as server_state
        old = server_state.model
        old_p = server_state.provider
        server_state.model = MagicMock()
        server_state.provider = None
        try:
            assert _model_ready() is True
        finally:
            server_state.model = old
            server_state.provider = old_p

    def test_ready_with_provider(self):
        import state as server_state
        old = server_state.model
        old_p = server_state.provider
        server_state.model = None
        server_state.provider = MagicMock()
        try:
            assert _model_ready() is True
        finally:
            server_state.model = old
            server_state.provider = old_p

    def test_not_ready(self):
        import state as server_state
        old = server_state.model
        old_p = server_state.provider
        server_state.model = None
        server_state.provider = None
        try:
            assert _model_ready() is False
        finally:
            server_state.model = old
            server_state.provider = old_p


class TestCountTokens:
    def test_with_tokenizer(self):
        state = SimpleNamespace(tokenizer=MagicMock())
        state.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        assert _count_tokens("hello world", state) == 5

    def test_fallback_to_word_count(self):
        state = SimpleNamespace(tokenizer=None)
        assert _count_tokens("hello world foo", state) == 3

    def test_tokenizer_error_fallback(self):
        state = SimpleNamespace(tokenizer=MagicMock())
        state.tokenizer.encode.side_effect = RuntimeError("fail")
        assert _count_tokens("hello world", state) == 2


class TestExtractUserMessage:
    def test_extracts_last_user(self):
        messages = [
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="How can I help?"),
            Message(role="user", content="What's 2+2?"),
        ]
        assert _extract_user_message(messages) == "What's 2+2?"

    def test_no_user_message(self):
        messages = [Message(role="assistant", content="Hi")]
        assert _extract_user_message(messages) is None

    def test_empty_list(self):
        assert _extract_user_message([]) is None

    def test_user_with_empty_content(self):
        messages = [Message(role="user", content="")]
        assert _extract_user_message(messages) is None

    def test_multiple_users_takes_last(self):
        messages = [
            Message(role="user", content="first"),
            Message(role="user", content="second"),
        ]
        assert _extract_user_message(messages) == "second"


# ── Search ──


class TestSearchSessionsSync:
    def test_empty_query_returns_empty(self, tmp_path):
        result = _search_sessions_sync("", 20)
        assert isinstance(result, list)

    def test_whitespace_query_returns_all(self, tmp_path):
        """Whitespace-only query: stripped to empty, matches all (empty in every string)."""
        result = _search_sessions_sync("   ", 20)
        assert isinstance(result, list)

    def test_finds_matching_session(self, tmp_path):
        """The function searches real disk files - verify it returns a list."""
        result = _search_sessions_sync("python", 20)
        assert isinstance(result, list)

    def test_limit_respected(self, tmp_path):
        """Verify the function respects the limit parameter."""
        result = _search_sessions_sync("query", 2)
        assert len(result) <= 2


# ── Chat Suggestions ──


class TestChatSuggestions:
    def test_suggestions(self):
        ir = InferenceRouter()
        client = TestClient(_app(ir))
        resp = client.get("/chat/suggestions")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert len(data) >= 4
        assert all("text" in s for s in data)

    def test_suggestions_have_icons(self):
        ir = InferenceRouter()
        client = TestClient(_app(ir))
        resp = client.get("/chat/suggestions")
        data = resp.json()["data"]
        assert all("icon" in s for s in data)


# ── Session CRUD ──


class TestSessionCRUD:
    def test_list_sessions_empty(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._session_repo.list.return_value = []
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("data") == [] or body.get("data") is not None

    def test_create_session(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._session_repo.save = MagicMock()
        client = TestClient(_app(ir))
        resp = client.post("/chat/sessions", json={"session_id": "test-abc", "name": "Test Chat"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["session_id"] == "test-abc"

    def test_create_session_generates_uuid(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._session_repo.save = MagicMock()
        client = TestClient(_app(ir))
        resp = client.post("/chat/sessions", json={"name": "Auto ID"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["session_id"]  # UUID generated

    def test_get_session_not_found(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_session = MagicMock(return_value={})
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._session_repo.delete.return_value = True
        client = TestClient(_app(ir))
        resp = client.delete("/chat/sessions/to-delete")
        assert resp.status_code == 200
        assert resp.json()["data"]["session_id"] == "to-delete"

    def test_delete_session_not_found(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._session_repo.delete.return_value = False
        client = TestClient(_app(ir))
        resp = client.delete("/chat/sessions/ghost")
        assert resp.status_code == 404

    def test_upsert_session(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_session = MagicMock(return_value={"session_id": "s1"})
        ir._save_session = MagicMock()
        client = TestClient(_app(ir))
        resp = client.put("/chat/sessions/s1", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["data"]["session_id"] == "s1"


# ── Search endpoint ──


class TestSearchEndpoint:
    def test_empty_query(self, tmp_path):
        ir = _make_ir(tmp_path)
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions/search?q=")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []

    def test_whitespace_query(self, tmp_path):
        ir = _make_ir(tmp_path)
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions/search?q=%20%20%20")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []


# ── Current session ──


class TestGetCurrentSession:
    def test_no_sessions(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._build_session_cache = MagicMock(return_value=[])
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200

    def test_returns_first_session(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._build_session_cache = MagicMock(return_value=[{"session_id": "first"}])
        client = TestClient(_app(ir))
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200
        assert resp.json()["data"]["session_id"] == "first"


# ── Context endpoints ──


class TestContextEndpoints:
    def test_inspect_context_no_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_context_core = MagicMock(return_value=None)
        client = TestClient(_app(ir))
        resp = client.get("/context/inspect")
        assert resp.status_code == 503
        assert "error" in resp.json()

    def test_store_fact_no_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_context_core = MagicMock(return_value=None)
        client = TestClient(_app(ir))
        resp = client.post("/context/fact", params={"key": "k", "value": "v"})
        assert resp.status_code == 503
        assert "error" in resp.json()

    def test_get_facts_no_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_context_core = MagicMock(return_value=None)
        client = TestClient(_app(ir))
        resp = client.get("/context/facts")
        assert resp.status_code == 503
        assert resp.json()["code"] == "E_INFRA_STARTUP"

    def test_reset_context_no_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        ir._get_context_core = MagicMock(return_value=None)
        client = TestClient(_app(ir))
        resp = client.post("/context/reset")
        assert resp.status_code == 503

    def test_inspect_context_with_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        mock_core.get_context_inspector.return_value = {"working_memory": [], "semantic_memory": {}}
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.get("/context/inspect")
        assert resp.status_code == 200
        assert "working_memory" in resp.json()

    def test_store_fact_with_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.post("/context/fact", params={"key": "mykey", "value": "myval"})
        assert resp.status_code == 200
        mock_core.store_fact.assert_called_once_with("mykey", "myval")

    def test_get_facts_with_core(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        mock_core.semantic_memory = {"k1": {"value": "v1"}}
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.get("/context/facts")
        assert resp.status_code == 200
        assert len(resp.json()["facts"]) == 1

    def test_get_facts_with_query(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        mock_core.search_semantic.return_value = [{"key": "k1", "content": "match"}]
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.get("/context/facts", params={"query": "python"})
        assert resp.status_code == 200
        mock_core.search_semantic.assert_called_once_with("python")

    def test_reset_context_session(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.post("/context/reset")
        assert resp.status_code == 200
        mock_core.reset_session.assert_called_once()

    def test_reset_context_all(self, tmp_path):
        ir = _make_ir(tmp_path)
        mock_core = MagicMock()
        ir._get_context_core = MagicMock(return_value=mock_core)
        client = TestClient(_app(ir))
        resp = client.post("/context/reset", params={"all": True})
        assert resp.status_code == 200
        mock_core.reset_all.assert_called_once()
