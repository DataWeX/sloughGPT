"""
Tests for the inference router — chat, generate, sessions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.inference import _instance as _inference_router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_generate_success(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(return_value="Hello world")
        mock_get_provider.return_value = mock_prov
        MOCK_STATE.provider = mock_prov

        resp = client.post("/inference/generate", json={"prompt": "Hi", "max_new_tokens": 16})
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "Hello world"
        assert body["model"] == "gpt2"
        assert body["tokens_generated"] == 2

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_generate_provider_failure(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(side_effect=RuntimeError("inference failed"))
        mock_get_provider.return_value = mock_prov
        MOCK_STATE.provider = mock_prov

        resp = client.post("/inference/generate", json={"prompt": "Hi"})
        assert resp.status_code == 500

    def test_generate_no_model(self, client, no_model_state):
        resp = client.post("/inference/generate", json={"prompt": "Hi"})
        assert resp.status_code == 503

    def test_generate_missing_prompt_422(self, client):
        resp = client.post("/inference/generate", json={"max_new_tokens": 16})
        assert resp.status_code == 422

    def test_generate_max_new_tokens_zero_422(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hi", "max_new_tokens": 0})
        assert resp.status_code == 422

    def test_generate_temperature_too_high_422(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hi", "temperature": 2.5})
        assert resp.status_code == 422

    def test_generate_temperature_negative_422(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hi", "temperature": -1})
        assert resp.status_code == 422

    def test_generate_top_k_too_large_422(self, client):
        resp = client.post("/inference/generate", json={"prompt": "Hi", "top_k": 9999})
        assert resp.status_code == 422

    def test_generate_wrong_method_405(self, client):
        assert client.get("/inference/generate").status_code == 405

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_generate_custom_params(self, mock_get_provider, client):
        mock_prov = MagicMock()
        mock_prov.chat = AsyncMock(return_value="Response")
        mock_get_provider.return_value = mock_prov

        resp = client.post(
            "/inference/generate",
            json={
                "prompt": "Test",
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["text"] == "Response"


class TestGenerateStream:
    """POST /inference/generate/stream"""

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_generate_stream_success(self, mock_get_provider, client):
        mock_prov = MagicMock()

        async def _stream(*a, **kw):
            yield "Hello"
            yield " world"

        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        resp = client.post(
            "/inference/generate/stream", json={"prompt": "Hi", "max_new_tokens": 16}
        )
        assert resp.status_code == 200
        text = resp.text
        assert "Hello" in text
        assert "world" in text

    def test_generate_stream_no_model(self, client, no_model_state):
        resp = client.post("/inference/generate/stream", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert "error" in resp.text

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider", return_value=None)
    def test_generate_stream_no_provider(self, mock_get_provider, client):
        resp = client.post("/inference/generate/stream", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert "No provider available" in resp.text


class TestChatStream:
    """POST /chat/stream"""

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    def test_chat_stream_no_user_message(self, client):
        resp = client.post(
            "/chat/stream",
            json={
                "messages": [{"role": "system", "content": "You are helpful."}],
            },
        )
        assert resp.status_code == 200
        assert "No user message" in resp.text

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_memory_service")
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider", return_value=None)
    def test_chat_stream_no_provider(
        self, mock_get_provider, mock_rag, mock_mem_svc, mock_enrich, client
    ):
        mock_mem_svc.return_value.stats.return_value = {"total_facts": 0}
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}
        resp = client.post(
            "/chat/stream",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200


class TestChat:
    """POST /chat"""

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    def test_chat_success(self, mock_get_domain, client):
        mock_domain = MagicMock()
        mock_domain.respond = AsyncMock(
            return_value=MagicMock(
                text="Hello back",
                session_id="sess-1",
                done=True,
            )
        )
        mock_get_domain.return_value = mock_domain

        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "Hello back"
        assert body["session_id"] == "sess-1"

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    def test_chat_no_message(self, client):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 400

    def test_chat_no_model(self, client, no_model_state):
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
        assert resp.status_code == 503

    def test_chat_temperature_too_high_422(self, client):
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 3.0,
            },
        )
        assert resp.status_code == 422

    def test_chat_max_tokens_zero_422(self, client):
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 0,
            },
        )
        assert resp.status_code == 422

    def test_chat_repetition_penalty_out_of_range_422(self, client):
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "repetition_penalty": 3.0,
            },
        )
        assert resp.status_code == 422

    def test_chat_wrong_method_405(self, client):
        assert client.get("/chat").status_code == 405


class TestListSessions:
    """GET /chat/sessions"""

    @patch.object(_inference_router, "_build_session_metadata_index")
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

    @patch.object(_inference_router, "_build_session_metadata_index")
    def test_list_sessions_empty(self, mock_build, client):
        mock_build.return_value = []
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @patch.object(_inference_router, "_build_session_metadata_index")
    def test_list_sessions_archived_filter(self, mock_build, client):
        mock_build.return_value = [
            {"id": "s1", "archived": True},
            {"id": "s2", "archived": False},
            {"id": "s3", "archived": True},
        ]
        resp = client.get("/chat/sessions?archived=true")
        data = resp.json()["data"]
        assert [s["id"] for s in data] == ["s1", "s3"]
        resp = client.get("/chat/sessions?archived=false")
        assert [s["id"] for s in resp.json()["data"]] == ["s2"]


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

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    def test_inspect_context(self, mock_get_domain, client):
        resp = client.get("/context/inspect")
        assert resp.status_code == 200

    def test_get_facts(self, client):
        resp = client.get("/context/facts")
        assert resp.status_code == 200


class TestContextStoreFact:
    """POST /context/fact"""

    @patch.object(_inference_router, "_get_context_core")
    def test_store_fact(self, mock_get_core, client):
        core = MagicMock()
        mock_get_core.return_value = core
        resp = client.post("/context/fact?key=name&value=alice")
        assert resp.status_code == 200
        assert resp.json()["data"]["stored"] == "name"
        core.store_fact.assert_called_once_with("name", "alice")

    @patch.object(_inference_router, "_get_context_core")
    def test_store_fact_no_core(self, mock_get_core, client):
        mock_get_core.return_value = None
        resp = client.post("/context/fact?key=k&value=v")
        assert resp.status_code == 503

    @patch.object(_inference_router, "_get_context_core")
    def test_store_fact_wrong_method_405(self, mock_get_core, client):
        assert client.get("/context/fact").status_code == 405


class TestContextReset:
    """POST /context/reset"""

    @patch.object(_inference_router, "_get_context_core")
    def test_reset_session(self, mock_get_core, client):
        core = MagicMock()
        mock_get_core.return_value = core
        resp = client.post("/context/reset")
        assert resp.status_code == 200
        assert resp.json()["data"]["reset"] == "session"
        core.reset_session.assert_called_once()

    @patch.object(_inference_router, "_get_context_core")
    def test_reset_all(self, mock_get_core, client):
        core = MagicMock()
        mock_get_core.return_value = core
        resp = client.post("/context/reset?all=true")
        assert resp.status_code == 200
        assert resp.json()["data"]["reset"] == "all"
        core.reset_all.assert_called_once()

    @patch.object(_inference_router, "_get_context_core")
    def test_reset_no_core(self, mock_get_core, client):
        mock_get_core.return_value = None
        resp = client.post("/context/reset")
        assert resp.json()["error"] == "ContextCore not available"


class TestContextFactsQuery:
    """GET /context/facts?query=..."""

    @patch.object(_inference_router, "_get_context_core")
    def test_get_facts_with_query(self, mock_get_core, client):
        core = MagicMock()
        core.search_semantic.return_value = [{"fact": "1"}]
        mock_get_core.return_value = core
        resp = client.get("/context/facts?query=ai")
        assert resp.json()["data"]["facts"] == [{"fact": "1"}]
        core.search_semantic.assert_called_once_with("ai")

    @patch.object(_inference_router, "_get_context_core")
    def test_get_facts_no_core(self, mock_get_core, client):
        mock_get_core.return_value = None
        resp = client.get("/context/facts")
        assert resp.status_code == 503


class TestSearchSessions:
    """GET /chat/sessions/search"""

    @patch("apps.api.server.routers.inference._search_sessions_sync")
    def test_search_sessions(self, mock_search, client):
        mock_search.return_value = [
            {
                "id": "s1",
                "name": "Test Chat",
                "messages": [{"role": "user", "content": "hello"}],
                "updated_at": "2026-01-01",
            },
        ]
        resp = client.get("/chat/sessions/search?q=Test")
        assert resp.status_code == 200

    def test_search_sessions_empty_query(self, client):
        resp = client.get("/chat/sessions/search")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["meta"]["query"] == ""
        assert body["meta"]["total"] == 0

    def test_search_sessions_whitespace_query(self, client):
        resp = client.get("/chat/sessions/search?q=%20%20")
        assert resp.json()["data"] == []


class TestSuggestions:
    """GET /suggestions"""

    def test_chat_suggestions(self, client):
        resp = client.get("/chat/suggestions")
        assert resp.status_code == 200

    def test_chat_suggestions_alt_path(self, client):
        resp = client.get("/chat/suggestions")
        assert resp.status_code == 200

    def test_suggestions_data_shape(self, client):
        resp = client.get("/chat/suggestions")
        data = resp.json()["data"]
        assert isinstance(data, list)
        assert len(data) >= 1
        for item in data:
            assert "text" in item
            assert "icon" in item


class TestVoice:
    """Voice message upload + audio retrieval."""

    def test_voice_rejects_non_audio(self, client):
        resp = client.post(
            "/chat/voice/voice-test-sess",
            files={"file": ("note.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400

    def test_voice_rejects_no_file(self, client):
        resp = client.post("/chat/voice/voice-test-sess")
        assert resp.status_code in (400, 422)

    def test_audio_traversal_rejected(self, client):
        resp = client.get("/chat/audio/safe-sess/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (403, 404)

    def test_audio_traversal_guard_direct(self):
        import asyncio

        from domain.infrastructure._internal.errors import AuthError

        with pytest.raises(AuthError):
            asyncio.run(_inference_router.get_voice_audio("../evil", "msg"))

    def test_audio_not_found(self, client):
        resp = client.get("/chat/audio/missing-sess/does-not-exist")
        assert resp.status_code == 404


class TestProviders:
    """GET /providers"""

    def test_list_model_providers(self, client):
        resp = client.get("/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    @staticmethod
    def _register_both_shapes():
        """Register one provider of each metadata shape.

        The Provider protocol declares ``metadata`` a ``@property``, but
        ct_provider / slonet_provider / slonet_server implement it as a plain
        method while api_provider / router / slo_transformer / multimodal use
        ``@property``. The route read it property-style, so a method-shaped
        provider put a bound method in the payload and 500'd with
        E_SERIALIZATION — the live bug behind card 2460373b. Every other test
        here registers no provider at all, which is why they passed while
        production failed.
        """
        from domain.models._internal.provider import registry as _reg

        class _Caps:
            chat = streaming = embedding = vision = True

        class MethodShaped:
            model_id = "probe-method"

            @property
            def capabilities(self):
                return _Caps()

            def metadata(self):  # plain method, like slonet_provider
                return {"n_layer": 4, "shaped": "method"}

        class PropertyShaped:
            model_id = "probe-property"

            @property
            def capabilities(self):
                return _Caps()

            @property
            def metadata(self):  # @property, like api_provider
                return {"n_layer": 8, "shaped": "property"}

        saved = {n: _reg.get_provider(n) for n in _reg.list_providers()}
        _reg.register_provider("probe-method", MethodShaped())
        _reg.register_provider("probe-property", PropertyShaped())
        return _reg, saved

    @staticmethod
    def _restore(_reg, saved):
        _reg.clear_providers()
        for name, provider in saved.items():
            _reg.register_provider(name, provider)

    def test_method_shaped_metadata_serialises(self, client):
        """A plain-method `metadata` must come back as a dict, not 500."""
        _reg, saved = self._register_both_shapes()
        try:
            resp = client.get("/providers")
            assert resp.status_code == 200, resp.text
            meta = resp.json()["data"]["probe-method"]["metadata"]
            assert meta == {"n_layer": 4, "shaped": "method"}
            assert not callable(meta)
        finally:
            self._restore(_reg, saved)

    def test_property_shaped_metadata_still_works(self, client):
        """The @property half must not regress — both shapes are accepted."""
        _reg, saved = self._register_both_shapes()
        try:
            resp = client.get("/providers")
            assert resp.status_code == 200, resp.text
            payload = resp.json()["data"]
            assert payload["probe-property"]["metadata"] == {"n_layer": 8, "shaped": "property"}
            assert payload["probe-method"]["capabilities"]["chat"] is True
        finally:
            self._restore(_reg, saved)


class TestCurrentSession:
    """GET /chat/sessions/current"""

    def test_get_current_session(self, client):
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200

    @patch.object(_inference_router, "_build_session_cache")
    def test_get_current_session_empty(self, mock_build, client):
        mock_build.return_value = []
        resp = client.get("/chat/sessions/current")
        assert resp.status_code == 200
        assert resp.json()["data"] is None


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
        resp = client.put(
            "/chat/sessions/test-session",
            json={
                "name": "Updated",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code in (200, 201)


class TestCachedReplayChunking:
    """Coalescer replay must preserve cached text byte-for-byte (card 053).

    The old replay split on whitespace and re-joined with single spaces,
    flattening newlines / multi-space runs on reconnect.
    """

    def test_concat_is_exact_for_awkward_whitespace(self):
        from apps.api.server.routers.inference import _chunk_replay

        text = "line one\n\nline two    indented\n\tend "
        assert "".join(_chunk_replay(text)) == text

    def test_chunks_are_bounded_and_nonempty(self):
        from apps.api.server.routers.inference import _chunk_replay

        chunks = _chunk_replay("x" * 250)
        assert len(chunks) > 1
        assert all(chunks)
        assert all(len(c) <= 40 for c in chunks)

    def test_empty_text_yields_no_chunks(self):
        from apps.api.server.routers.inference import _chunk_replay

        assert _chunk_replay("") == []

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_coalescer")
    @patch("apps.api.server.routers.inference.get_provider")
    def test_generate_stream_replay_preserves_whitespace(
        self, mock_get_provider, mock_get_coalescer, client
    ):
        import json as _json

        mock_get_provider.return_value = MagicMock()
        cached = "alpha  beta\n\ngamma delta"
        existing = MagicMock()
        existing.error = None
        existing.result = cached
        existing.event.wait = AsyncMock()
        fake = MagicMock()
        fake.hash = MagicMock(return_value="k")
        fake.start = AsyncMock(return_value=existing)
        mock_get_coalescer.return_value = fake

        resp = client.post(
            "/inference/generate/stream", json={"prompt": "Hi", "max_new_tokens": 16}
        )
        assert resp.status_code == 200
        tokens = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                env = _json.loads(line[6:])
                tok = (env.get("data") or {}).get("token")
                if tok:
                    tokens.append(tok)
        assert "".join(tokens) == cached


class TestChatPostGenTasks:
    """POST /chat must run the same post-generation task set as streaming.

    Audit gap G2 (card db19c373): ``_run_post_gen_tasks`` was launched from
    exactly one site — ``chat_stream`` — so a non-streaming turn recorded no
    substrate episode at all. Consciousness, entity extraction, the continual
    learner and response tracking all silently skipped half of all chat.
    """

    def _domain_mock(self):
        mock_domain = MagicMock()
        mock_domain.respond = AsyncMock(
            return_value=MagicMock(text="Hello back", session_id="sess-1", done=True)
        )
        return mock_domain

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_non_stream_chat_runs_post_gen_tasks(self, mock_post_gen, mock_get_domain, client):
        """The gap itself: this call did not exist before G2."""
        mock_post_gen.return_value = {"narrative": "", "level": 0}
        mock_get_domain.return_value = self._domain_mock()

        resp = client.post(
            "/chat", json={"messages": [{"role": "user", "content": "Hello"}]}
        )

        assert resp.status_code == 200
        mock_post_gen.assert_called_once()

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_post_gen_receives_the_turn_it_belongs_to(
        self, mock_post_gen, mock_get_domain, client
    ):
        """Post-gen work must be attributed to the response that produced it."""
        mock_post_gen.return_value = {"narrative": "", "level": 0}
        mock_get_domain.return_value = self._domain_mock()

        client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

        args = mock_post_gen.call_args.args
        assert args[0] == "Hello back", "full_response must be the generated text"
        assert args[1] == "Hello", "user_msg must be the prompt"
        assert args[2] == "sess-1", "session_id must follow the response"
        assert args[3] is not None, "start_time drives the duration metric"

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    @patch("apps.api.server.routers.inference.capture")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_no_inline_capture_alongside_post_gen(
        self, mock_post_gen, mock_capture, mock_get_domain, client
    ):
        """No double-run: post-gen already captures, so chat() must not too.

        Post-gen is spied out here, so any capture reached directly by chat()
        would still fire — asserting it does not is the anti-double-count
        guarantee.
        """
        mock_post_gen.return_value = {"narrative": "", "level": 0}
        mock_get_domain.return_value = self._domain_mock()

        client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

        mock_capture.assert_not_called()

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("domain.get_chat_domain")
    @patch(
        "apps.api.server.routers.inference._run_post_gen_tasks",
        side_effect=RuntimeError("post-gen exploded"),
    )
    def test_post_gen_failure_does_not_fail_the_turn(
        self, mock_post_gen, mock_get_domain, client
    ):
        """These tasks are best-effort: the reply must still reach the user."""
        mock_get_domain.return_value = self._domain_mock()

        resp = client.post(
            "/chat", json={"messages": [{"role": "user", "content": "Hello"}]}
        )

        assert resp.status_code == 200
        assert resp.json()["message"] == "Hello back"
