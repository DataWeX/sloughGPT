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
MOCK_STATE.tokenizer = None
MOCK_STATE.model_request_logger = None
MOCK_STATE.checkpoint = None
MOCK_STATE.soul_engine = None
MOCK_STATE.current_soul = None

MOCK_STARTUP = {"phase": "ready", "step": 8, "total": 8, "message": "Ready"}


@pytest.fixture
def no_model_state():
    """Programmatically force the real no-model condition, then restore.

    The no-model handlers read the real ``state`` module and
    ``startup_progress.STARTUP_PHASE``. A prior test in the same process may
    leave ``state.model``/``state.provider`` set and the phase ``"ready"``,
    which would bypass the 503 path. Snapshooting the real globals makes these
    tests order-independent without mocks.
    """
    import state as server_state
    from startup_progress import STARTUP_PHASE

    saved_model = server_state.model
    saved_provider = server_state.provider
    saved_phase = STARTUP_PHASE["phase"]

    server_state.model = None
    server_state.provider = None
    STARTUP_PHASE["phase"] = "initializing"
    try:
        yield
    finally:
        server_state.model = saved_model
        server_state.provider = saved_provider
        STARTUP_PHASE["phase"] = saved_phase


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

    def test_generate_no_model(self, client, no_model_state):
        resp = client.post("/inference/generate", json={"prompt": "Hi"})
        assert resp.status_code == 503

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.models.provider.get_provider")
    def test_generate_custom_params(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(return_value="Response")
        mock_get_provider.return_value = mock_prov

        resp = client.post("/inference/generate", json={
            "prompt": "Test",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
        })
        assert resp.status_code == 200
        assert resp.json()["text"] == "Response"


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

    def test_generate_stream_no_model(self, client, no_model_state):
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

    def test_chat_no_model(self, client, no_model_state):
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


class TestInfo:
    """GET /info"""

    def test_get_info(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_version" in data or "data" in data

    def test_get_info_soul(self, client):
        resp = client.get("/info/soul")
        assert resp.status_code == 200


class TestRoot:
    """GET /"""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


class TestChatTools:
    """GET /chat/tools"""

    def test_list_chat_tools(self, client):
        resp = client.get("/chat/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data or "data" in data


class TestContextEndpoints:
    """Context management endpoints"""

    @patch.dict("sys.modules", {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)})
    @patch("domains.get_chat_domain")
    def test_inspect_context(self, mock_get_domain, client):
        resp = client.get("/context/inspect")
        assert resp.status_code == 200

    def test_get_facts(self, client):
        resp = client.get("/context/facts")
        assert resp.status_code == 200


class TestSearchSessions:
    """GET /chat/sessions/search"""

    @patch.object(_inference_router, "_build_session_cache")
    def test_search_sessions(self, mock_build, client):
        mock_build.return_value = [
            {"id": "s1", "name": "Test Chat", "messages": [{"role": "user", "content": "hello"}], "updated_at": "2026-01-01"},
        ]
        resp = client.get("/chat/sessions/search?q=Test")
        assert resp.status_code == 200


class TestSuggestions:
    """GET /suggestions"""

    def test_chat_suggestions(self, client):
        resp = client.get("/suggestions")
        assert resp.status_code == 200

    def test_chat_suggestions_alt_path(self, client):
        resp = client.get("/chat/suggestions")
        assert resp.status_code == 200


class TestProviders:
    """GET /providers"""

    def test_list_model_providers(self, client):
        resp = client.get("/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data


class TestCurrentSession:
    """GET /chat/sessions/current"""

    def test_get_current_session(self, client):
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200


class TestCreateSession:
    """POST /chat/sessions"""

    def test_create_session(self, client):
        resp = client.post("/chat/sessions", json={"session_id": "test-session"})
        assert resp.status_code == 200

    def test_create_session_auto_id(self, client):
        resp = client.post("/chat/sessions", json={})
        assert resp.status_code == 200


class TestGetSession:
    """GET /chat/sessions/{session_id}"""

    def test_get_session_not_found(self, client):
        resp = client.get("/chat/sessions/nonexistent")
        assert resp.status_code in (200, 404)


class TestDeleteSession:
    """DELETE /chat/sessions/{session_id}"""

    def test_delete_session(self, client):
        resp = client.delete("/chat/sessions/nonexistent")
        assert resp.status_code in (200, 404)


class TestUpsertSession:
    """PUT /chat/sessions/{session_id}"""

    def test_upsert_session(self, client):
        resp = client.put("/chat/sessions/test-session", json={
            "name": "Updated",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code in (200, 201)
