"""
Tests for the inference router — chat, generate, sessions.
"""

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers import inference as _inf
from apps.api.server.routers.chat import router as _chat_router
from apps.api.server.routers.inference import _instance as _inference_router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(_inference_router.router)
    # POST /chat is owned by ChatRouter (it streams — that is its only state),
    # the inference kernel through its public facades (card 13c37a23) —
    # mount both routers so this app mirrors the deployed surface.
    _app.include_router(_chat_router)
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


class TestChatErrorFrames:
    """POST /chat streams, so a handler error arrives as an SSE frame, not a status.

    This is the shape every error inside the turn takes once the response has
    started: HTTP-level gates (model readiness) are caught earlier by
    ``ReadinessGateMiddleware`` and keep returning a real 503 — see
    ``tests/server/test_readiness_gate.py``. Everything that fails *during* the
    turn surfaces here instead.
    """

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    def test_chat_no_user_message_frame(self, client):
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "system", "content": "You are helpful."}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "No user message" in resp.text

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider", return_value=None)
    def test_chat_no_provider_frame(self, mock_get_provider, mock_rag, mock_enrich, client):
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")


class TestChatStreamCancel:
    """The cancel channel must reach the provider's token loop, end to end.

    ``CancelManager`` sets a ``threading.Event``; the provider's token loop
    polls ``is_set()``. The only way a cancel is real is if those are the SAME
    object, so this walks the whole chain in one request:

      chat manager `_track` mints stop  →  hands it to the stream factory
        →  `_stream_tokens(request, stop)`  →  `to_turn(cancel_event=stop)`
        →  `TurnRequest.cancel_event`  →  `stream_turn`  →  provider kwarg

    Before the fix the factory took zero arguments, so ``_stop`` was discarded
    and the provider always received ``None`` — the API still answered
    ``cancelled: true`` while generation ran to completion.
    """

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider")
    def test_cancel_event_reaches_provider(self, mock_get_provider, mock_rag, mock_enrich, client):
        import threading

        seen: dict = {}

        async def _stream(messages, **kw):
            seen["cancel_event"] = kw.get("cancel_event", "NOT_PASSED")
            yield "hi"

        mock_prov = MagicMock()
        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}

        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "session_id": "cancel-pin",
            },
        )

        assert resp.status_code == 200
        assert "cancel_event" in seen, "provider.chat_stream was never called"
        assert seen["cancel_event"] is not None, (
            "provider received cancel_event=None — the registry's stop event was "
            "dropped before reaching the token loop, so POST /chat/control "
            "cancel would report success while generation continues"
        )
        assert isinstance(seen["cancel_event"], threading.Event)


class TestStreamTurnBareProbeGates:
    """Quick-probe flags must gate the context/consciousness pipeline.

    QuickActions "Test model" posts ``use_consciousness=false,
    use_context_core=false, use_rag=false, session_id='quick-test'``
    (card 20261112_140010). Before the fix ``stream_turn`` ignored the
    first two: the consciousness reflection narrative was prepended
    unconditionally and the 0.5B model answered the narrative instead of
    the probe; the ContextCore frame also ignored the wire flag that the
    non-streaming ``chat()`` honored.
    """

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider")
    @patch(
        "apps.api.server.routers.inference._apply_consciousness_reflection",
        new_callable=AsyncMock,
    )
    @patch("apps.api.server.routers.inference._build_context_frame", new_callable=AsyncMock)
    def test_bare_probe_flags_skip_context_and_consciousness(
        self, mock_frame, mock_reflect, mock_get_provider, mock_rag, mock_enrich, client
    ):
        mock_frame.return_value = (None, {})

        async def _reflect(messages, corr_id):
            return messages

        mock_reflect.side_effect = _reflect
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}

        async def _stream(messages, **kw):
            yield "ok"

        mock_prov = MagicMock()
        mock_prov.chat_stream = _stream
        mock_get_provider.return_value = mock_prov

        # Bare probe: neither pipeline stage may run.
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "session_id": "quick-test",
                "use_consciousness": False,
                "use_context_core": False,
                "use_rag": False,
            },
        )
        assert resp.status_code == 200
        assert not mock_reflect.called, (
            "use_consciousness=false but _apply_consciousness_reflection still ran"
        )
        assert not mock_frame.called, (
            "use_context_core=false but the ContextCore frame was still built"
        )
        assert not mock_enrich.called, "use_context_core=false but knowledge enrichment still ran"

        # Omitted flags: use_consciousness now defaults OFF (card
        # 20261114_092948 — narrative-in-system-head hijacked chat replies),
        # context/RAG keep defaulting ON.
        mock_frame.reset_mock()
        mock_reflect.reset_mock()
        mock_enrich.reset_mock()
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hi"}], "session_id": "probe-default"},
        )
        assert resp.status_code == 200
        assert not mock_reflect.called, "omitted use_consciousness must default to OFF"
        mock_frame.assert_called_once()
        mock_enrich.assert_called_once()

        # Explicit opt-in keeps the reflection pipeline reachable.
        mock_reflect.reset_mock()
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi again"}],
                "session_id": "probe-optin",
                "use_consciousness": True,
            },
        )
        assert resp.status_code == 200
        mock_reflect.assert_called_once()


class TestStreamTurnResponseSequence:
    """Pin the whole turn sequence: flags → stage order → closing frames.

    ``TestStreamTurnBareProbeGates`` asserts each stage ran or was skipped,
    one stage at a time. This class pins WHERE each stage sits in the chain
    and which arguments travel with it, so a refactor cannot silently
    reorder stages, re-enable one behind a flag, or drop the
    ``ctx_core=None`` hand-off and still pass the per-stage assertions
    (card 20261112_140010).
    """

    # The exact stage order ``stream_turn`` executes, per request shape.
    # ``memory`` appears twice because ``remember_facts_async`` has two call
    # sites in ``stream_turn`` (pre-existing) — consolidating them means
    # editing this expectation deliberately, not by accident.
    BARE_SEQUENCE = [
        "model_status",
        "memory_pressure",
        "scope",
        "generate",
        "post_gen",
        "memory",
        "memory",
    ]
    DEFAULT_SEQUENCE = [
        "model_status",
        "memory_pressure",
        "scope",
        "resolve_context_core",  # frame-builder argument (stream_turn)
        "context_frame",
        "enrich",
        "rag",
        "reflect",
        "generate",
        "resolve_context_core",  # post-gen argument (stream_turn)
        "post_gen",
        "memory",
        "memory",
    ]

    # The closing TurnResult frame POST /chat emits (see test_chat_state_seam).
    DONE_FRAME = '"phase": "STREAMING", "status": "complete"'

    @staticmethod
    def _post_sequenced(client, payload):
        """POST ``payload`` with every ``stream_turn`` stage instrumented.

        Returns ``(order, ctx_core_args, resp)``: the order the stages ran
        in, the ``ctx_core`` argument each ``_run_post_gen_tasks`` call
        received, and the SSE response. Behaviour-bearing seams (gates,
        scope resolution, ContextCore resolution, the post-gen task set) are
        WRAPPED — the real thing runs — so only their position in the chain
        is recorded, never their outcome.
        """
        order: list[str] = []
        ctx_core_args: list = []

        real_status = _inf._get_model_status
        real_pressure = _inf._check_memory_pressure
        real_post_gen = _inf._run_post_gen_tasks
        real_scope = _inference_router._resolve_turn_scope
        real_ctx_core = _inference_router._get_context_core

        def _status():
            order.append("model_status")
            return real_status()

        def _pressure():
            order.append("memory_pressure")
            return real_pressure()

        def _scope(session_id, req):
            order.append("scope")
            return real_scope(session_id, req)

        def _ctx_core():
            order.append("resolve_context_core")
            return real_ctx_core()

        async def _frame(*args, **kw):
            order.append("context_frame")
            return (None, {})

        def _enrich(*args, **kw):
            order.append("enrich")
            return {"facts": []}

        def _rag_ready():
            order.append("rag")
            return True

        def _rag_service():
            svc = MagicMock()
            svc.stats.return_value = {"total_chunks": 0}
            return svc

        async def _reflect(messages, corr_id):
            order.append("reflect")
            return messages

        async def _chat_stream(messages, **kw):
            order.append("generate")
            yield "ok"

        async def _remember(*args, **kw):
            order.append("memory")
            return []

        def _memory_service():
            svc = MagicMock()
            svc.remember_facts_async = _remember
            return svc

        def _post_gen(*args, **kw):
            order.append("post_gen")
            ctx_core_args.append(args[5])  # positional ctx_core parameter
            return real_post_gen(*args, **kw)

        provider = MagicMock()
        provider.chat_stream = _chat_stream

        def _get_provider(*args, **kw):
            return provider

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "state": MOCK_STATE,
                        "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP),
                    },
                )
            )
            stack.enter_context(
                patch("apps.api.server.routers.inference._get_model_status", new=_status)
            )
            stack.enter_context(
                patch(
                    "apps.api.server.routers.inference._check_memory_pressure",
                    new=_pressure,
                )
            )
            stack.enter_context(
                patch(
                    "apps.api.server.routers.inference._build_context_frame", new=_frame
                )
            )
            stack.enter_context(
                patch("apps.api.server.routers.inference._enrich_knowledge", new=_enrich)
            )
            stack.enter_context(
                patch(
                    "apps.api.server.routers.inference._apply_consciousness_reflection",
                    new=_reflect,
                )
            )
            stack.enter_context(
                patch("apps.api.server.routers.inference.get_rag_service", new=_rag_service)
            )
            stack.enter_context(
                patch("domain.cognition.is_rag_service_ready", new=_rag_ready)
            )
            stack.enter_context(
                patch("apps.api.server.routers.inference.get_memory_service", new=_memory_service)
            )
            stack.enter_context(
                patch("apps.api.server.routers.inference.get_provider", new=_get_provider)
            )
            stack.enter_context(
                patch(
                    "apps.api.server.routers.inference._run_post_gen_tasks", new=_post_gen
                )
            )
            stack.enter_context(
                patch.object(_inference_router, "_resolve_turn_scope", new=_scope)
            )
            stack.enter_context(
                patch.object(_inference_router, "_get_context_core", new=_ctx_core)
            )
            resp = client.post("/chat", json=payload)
        return order, ctx_core_args, resp

    def test_bare_probe_sequence_is_gates_then_generate_then_close(self, client):
        """The quick-test probe runs the minimal chain and nothing else."""
        order, ctx_core_args, resp = self._post_sequenced(
            client,
            {
                "messages": [{"role": "user", "content": "Hello!"}],
                "session_id": "seq-bare-probe",
                "use_consciousness": False,
                "use_context_core": False,
                "use_rag": False,
            },
        )
        assert resp.status_code == 200
        assert order == self.BARE_SEQUENCE, (
            f"bare probe ran stages outside the pinned contract: {order}"
        )
        assert "resolve_context_core" not in order, (
            "a bare probe resolved ContextCore — the flag gate on "
            "_get_context_core() was dropped"
        )
        assert ctx_core_args == [None], (
            "post-gen must receive the explicit ctx_core=None hand-off "
            f"(got {ctx_core_args!r})"
        )
        # User-visible outcome: tokens, the closing frame, no sidebands.
        assert '"token": "ok"' in resp.text, "the probe's answer never streamed"
        assert self.DONE_FRAME in resp.text, "TurnResult never closed the stream"
        assert '"phase": "CONSCIOUSNESS"' not in resp.text, (
            "bare probe surfaced a consciousness sideband"
        )
        assert '"phase": "MEMORY"' not in resp.text, (
            "bare probe surfaced a memory sideband"
        )

    def test_default_sequence_runs_full_pipeline_in_pinned_order(self, client):
        """Defaults (all flags True) run the full chain, in this order."""
        order, ctx_core_args, resp = self._post_sequenced(
            client,
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "session_id": "seq-default-pipeline",
            },
        )
        assert resp.status_code == 200
        assert order == self.DEFAULT_SEQUENCE, (
            f"default pipeline ran out of pinned order: {order}"
        )
        # ContextCore resolves twice — frame builder first, post-gen second —
        # and the resolved core is what post-gen receives.
        assert order.count("resolve_context_core") == 2
        assert len(ctx_core_args) == 1
        assert ctx_core_args[0] is not None, (
            "opted-in turn passed ctx_core=None to post-gen"
        )
        assert '"token": "ok"' in resp.text
        assert self.DONE_FRAME in resp.text, "TurnResult never closed the stream"


class TestChat:
    """POST /chat"""

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_chat_success(self, mock_get_provider, client):
        # POST /chat streams the engine's raw tokens (streaming is the endpoint's
        # only state), so the seam to fake is the provider's chat_stream.
        mock_provider = MagicMock()

        async def _chat_stream(messages, **kwargs):
            for tok in ["Hello", " back"]:
                yield tok

        mock_provider.chat_stream = _chat_stream
        mock_get_provider.return_value = mock_provider

        resp = client.post(
            "/chat",
            json={
                "session_id": "sess-1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "Hello" in resp.text
        assert " back" in resp.text

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    def test_chat_no_message(self, client):
        resp = client.post("/chat", json={"messages": []})
        # The turn fails after the response has started, so the error arrives as
        # an SSE frame rather than a 400 (see TestChatErrorFrames).
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_chat_no_model(self, client, no_model_state):
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
        # This bare test app has no ReadinessGateMiddleware; in the deployed app
        # the middleware answers 503 before the handler (test_readiness_gate.py).
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

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
    """POST /chat must run the post-generation task set on every turn.

    Audit gap G2 (card db19c373): ``_run_post_gen_tasks`` was launched from
    exactly one site, so half of all chat recorded no substrate episode —
    consciousness, entity extraction, the continual learner and response
    tracking all silently skipped. Stream-first (POST /chat is the one path;
    it frames the engine's token stream) runs the task set in
    ``stream_turn``; these tests pin that contract on the SSE path.
    """

    def _fake_provider(self) -> MagicMock:
        mock_provider = MagicMock()

        async def _chat_stream(messages, **kwargs):
            for tok in ["Hello", " back"]:
                yield tok

        mock_provider.chat_stream = _chat_stream
        return mock_provider

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_chat_runs_post_gen_tasks(
        self, mock_post_gen, mock_get_provider, mock_rag, mock_enrich, client
    ):
        """The gap itself: this call must fire on the SSE turn path."""
        mock_post_gen.return_value = {"narrative": "", "level": 0}
        mock_get_provider.return_value = self._fake_provider()
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}

        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

        assert resp.status_code == 200
        assert "Hello" in resp.text
        mock_post_gen.assert_called_once()

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_post_gen_receives_the_turn_it_belongs_to(
        self, mock_post_gen, mock_get_provider, mock_rag, mock_enrich, client
    ):
        """Post-gen work must be attributed to the response that produced it."""
        mock_post_gen.return_value = {"narrative": "", "level": 0}
        mock_get_provider.return_value = self._fake_provider()
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}

        client.post(
            "/chat",
            json={
                "session_id": "sess-1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        args = mock_post_gen.call_args.args
        assert args[0] == "Hello back", "full_response must be the generated text"
        assert args[1] == "Hello", "user_msg must be the prompt"
        assert args[2] == "sess-1", "session_id must follow the response"
        assert args[3] is not None, "start_time drives the duration metric"

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    @patch("apps.api.server.routers.inference.capture")
    @patch("apps.api.server.routers.inference._run_post_gen_tasks")
    def test_no_inline_capture_alongside_post_gen(
        self, mock_post_gen, mock_capture, mock_get_provider, client
    ):
        """No double-run: post-gen already captures, so the turn must not too.

        Post-gen is spied out here, so any capture reached directly by the
        turn path would still fire — asserting it does not is the
        anti-double-count guarantee.
        """
        mock_post_gen.return_value = {"narrative": "", "level": 0}

        client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

        mock_capture.assert_not_called()

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference._enrich_knowledge", return_value={"facts": []})
    @patch("apps.api.server.routers.inference.get_rag_service")
    @patch("apps.api.server.routers.inference.get_provider")
    @patch(
        "apps.api.server.routers.inference._run_post_gen_tasks",
        side_effect=RuntimeError("post-gen exploded"),
    )
    def test_post_gen_failure_does_not_fail_the_turn(
        self, mock_post_gen, mock_get_provider, mock_rag, mock_enrich, client
    ):
        """These tasks are best-effort: the reply must still reach the user."""
        mock_get_provider.return_value = self._fake_provider()
        mock_rag.return_value.stats.return_value = {"total_chunks": 0}

        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

        assert resp.status_code == 200
        assert "Hello" in resp.text
        assert " back" in resp.text


class TestModelStateReadiness:
    """_get_model_status — a terminal autoload failure must name its cause.

    Before MODEL_LOAD_STATE existed the function had three answers
    (E_NO_MODEL / E_STARTING / E_MODEL_LOADING), so a load that had already
    failed kept getting "Model still loading — please wait." forever.
    """

    @pytest.fixture
    def no_materialized_model(self, monkeypatch):
        import state as server_state
        from startup_progress import MODEL_LOAD_STATE, STARTUP_PHASE

        from domain.infrastructure.server_state import get_server_state

        monkeypatch.setenv("SLO_AUTOLOAD_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

        saved_state = dict(MODEL_LOAD_STATE)
        saved_phase = STARTUP_PHASE["phase"]
        saved_model = getattr(server_state, "model", None)
        saved_provider = getattr(server_state, "provider", None)
        core = get_server_state()
        saved_core_model = core.model.get()

        server_state.model = None
        server_state.provider = None
        core.model.set(None)
        STARTUP_PHASE["phase"] = "ready"
        MODEL_LOAD_STATE.update(status="pending", error=None)
        try:
            yield MODEL_LOAD_STATE
        finally:
            MODEL_LOAD_STATE.clear()
            MODEL_LOAD_STATE.update(saved_state)
            STARTUP_PHASE["phase"] = saved_phase
            server_state.model = saved_model
            server_state.provider = saved_provider
            core.model.set(saved_core_model)

    def test_failed_autoload_reports_the_real_cause(self, no_materialized_model):
        from apps.api.server.routers.inference import _get_model_status

        no_materialized_model.update(
            status="failed",
            error=(
                "No local weights for Qwen/Qwen2.5-0.5B-Instruct (no .slnc/.soul/safetensors) "
                "and SLO_BOOTSTRAP_HF is off — set SLO_BOOTSTRAP_HF=1 to download them, or "
                "point SLO_AUTOLOAD_MODEL at a model you already have."
            ),
        )

        ms = _get_model_status()

        assert ms["ready"] is False
        assert ms["code"] == "E_MODEL_LOAD_FAILED"
        assert ms["status"] == 503
        assert "SLO_BOOTSTRAP_HF" in ms["reason"]

    def test_in_flight_load_still_reports_still_loading(self, no_materialized_model):
        from apps.api.server.routers.inference import _get_model_status

        no_materialized_model.update(status="loading", error=None)

        ms = _get_model_status()

        assert ms["ready"] is False
        assert ms["code"] == "E_MODEL_LOADING"
        assert ms["status"] == 503

    def test_ready_phase_with_no_failure_keeps_the_waiting_answer(self, no_materialized_model):
        """Pending (no load started) + phase ready stays E_MODEL_LOADING."""
        from apps.api.server.routers.inference import _get_model_status

        ms = _get_model_status()

        assert ms["ready"] is False
        assert ms["code"] == "E_MODEL_LOADING"


class TestMemoryPressureMessage:
    """The block message must quote the reading that triggered it.

    Regression (card 20261101_110410, journal 2026-11-01): classification
    fired at 10:49:10 on 96.1% RSS (19074 MB), then the message served at
    10:50:19 said "System memory at 56%" because inference.py re-read
    psutil instead of asking the monitor. A user reading 56% is told the
    machine has plenty of memory while the request is refused — the
    message actively contradicts the refusal.
    """

    def test_message_quotes_the_classifying_reading(self):
        from apps.api.server.routers import inference as inf
        from domain.infrastructure._internal.memory_pressure import PressureLevel

        monitor = MagicMock()
        monitor.check.return_value = PressureLevel.EMERGENCY
        monitor.last_reading = {"percent": 96.1, "level": "emergency"}

        # A later, unrelated reading — this is what the old code printed.
        later = MagicMock()
        later.percent = 56.0

        with (
            patch.object(inf, "_memory_pressure_cache", None),
            patch.object(inf, "_memory_pressure_cache_ts", 0.0),
            patch(
                "domain.infrastructure.memory_pressure.get_memory_pressure_monitor",
                return_value=monitor,
            ),
            patch("psutil.virtual_memory", return_value=later) as fresh_read,
        ):
            message = inf._check_memory_pressure()

        assert message is not None
        assert "96%" in message
        assert "56%" not in message
        # The fix is the removal of the second, contradicting read.
        fresh_read.assert_not_called()


class TestAppendVolatileContext:
    """Per-turn context rides the LAST user turn, never the system prefix.

    KV reuse matches on prefix tokens: ``_resolve_kv_state`` and PrefixCache
    both compare head tokens. Folding knowledge/frame/RAG into the system
    prompt changed the prefix on every request, so each turn re-prefilled the
    whole prompt (measured 43.0s cold vs 1.26s warm on a 482-token prompt).
    Retrieval results are per-turn by nature, so they belong at the tail.
    """

    @staticmethod
    def _fn():
        from apps.api.server.routers import inference as inf

        return inf._append_volatile_context

    def test_block_is_appended_to_the_last_user_turn(self):
        append = self._fn()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]

        out = append(messages, "CONTEXT")

        assert out[-1]["content"] == "second\n\nCONTEXT"
        # Earlier turns are untouched — only the tail grows.
        assert out[0]["content"] == "first"
        assert out[1]["content"] == "answer"

    def test_head_is_identical_across_turns_with_different_context(self):
        """The actual regression: prefix stability, not string plumbing."""
        append = self._fn()
        head = {"role": "system", "content": "You are sloughGPT."}

        turn_one = append([head, {"role": "user", "content": "q1"}], "fact A")
        turn_two = append([head, {"role": "user", "content": "q2"}], "fact B")

        assert turn_one[0] == turn_two[0], "system prefix drifted between turns"
        assert "fact A" not in turn_one[0]["content"]
        assert "fact B" not in turn_two[0]["content"]

    @pytest.mark.parametrize("block", ["", None])
    def test_empty_block_is_a_noop(self, block):
        append = self._fn()
        messages = [{"role": "user", "content": "hi"}]

        assert append(messages, block) == [{"role": "user", "content": "hi"}]

    def test_multipart_content_gets_an_extra_text_part(self):
        append = self._fn()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "what is this?"}],
            }
        ]

        out = append(messages, "CONTEXT")

        assert isinstance(out[-1]["content"], list)
        assert out[-1]["content"][-1] == {"type": "text", "text": "CONTEXT"}
        assert out[-1]["content"][0]["text"] == "what is this?"

    def test_no_user_message_is_a_noop(self):
        append = self._fn()
        messages = [{"role": "assistant", "content": "unprompted"}]

        assert append(messages, "CONTEXT") == [{"role": "assistant", "content": "unprompted"}]

    def test_trim_budget_counts_the_volatile_block(self):
        """Attach before trimming, or the trimmer spends budget it cannot see."""
        from apps.api.server.routers import inference as inf

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u" * 40},
        ]
        block = "C" * 4000
        inf._append_volatile_context(messages, block)

        # Budget that fits the system prompt but not the volatile tail: the
        # trimmer must see the tail and drop it rather than silently overflow.
        trimmed = inf._trim_messages_to_budget(messages, max_context=100, max_new_tokens=10)

        assert all("C" * 4000 not in str(m.get("content")) for m in trimmed)


class TestTraitSteeringReachesThePrompt:
    """feedback -> TraitWeightsConfig -> managers -> ContextFrame -> prompt.

    Wiring the managers was only half the path: ``stream_turn`` used to read
    the frame's retrieval *layers* and discard ``frame.system_prompt``, and
    ``_build_context_frame`` skipped the build entirely while the knowledge
    store held zero facts. Both left the seam decorative on a fresh install.
    """

    STEERING = "\n[PERSONALITY INSTRUCTIONS]\n- Be direct and warm."

    @staticmethod
    def _frame():
        from domain.infrastructure._internal.context_core import ContextFrame

        return ContextFrame(
            id="f1",
            system_prompt="base" + TestTraitSteeringReachesThePrompt.STEERING,
            layers=[],
            total_tokens=10,
            max_tokens=2048,
            created_at="2026-11-03T00:00:00",
            system_extra=TestTraitSteeringReachesThePrompt.STEERING,
        )

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_steering_rides_the_system_turn_not_the_user_turn(self, mock_get_provider, client):
        captured: dict = {}

        async def _chat_stream(messages, **kwargs):
            captured["messages"] = messages
            yield "ok"

        provider = MagicMock()
        provider.chat_stream = _chat_stream
        mock_get_provider.return_value = provider

        with patch(
            "apps.api.server.routers.inference._build_context_frame",
            AsyncMock(return_value=(self._frame(), {})),
        ):
            client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "use_context_core": True},
            )

        messages = captured["messages"]
        system = next(m for m in messages if m["role"] == "system")
        user = next(m for m in messages if m["role"] == "user")
        assert "PERSONALITY INSTRUCTIONS" in system["content"]
        assert "PERSONALITY INSTRUCTIONS" not in user["content"]
        # Head placement is the point: trait steering is stable per feedback
        # profile, so it must stay out of the per-turn volatile tail.
        assert messages[0]["role"] == "system"

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    @patch("apps.api.server.routers.inference.get_provider")
    def test_steering_creates_a_system_turn_when_the_caller_sent_none(
        self, mock_get_provider, client
    ):
        captured: dict = {}

        async def _chat_stream(messages, **kwargs):
            captured["messages"] = messages
            yield "ok"

        provider = MagicMock()
        provider.chat_stream = _chat_stream
        mock_get_provider.return_value = provider

        with patch(
            "apps.api.server.routers.inference._build_context_frame",
            AsyncMock(return_value=(self._frame(), {})),
        ):
            client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

        system = next(m for m in captured["messages"] if m["role"] == "system")
        assert system["content"].strip() == TestTraitSteeringReachesThePrompt.STEERING.strip()


class TestFrameBuildsWithoutFacts:
    """Empty knowledge store must not disable trait steering."""

    def test_empty_memory_still_builds_the_frame_with_layers_off(self):
        from unittest.mock import AsyncMock as _AsyncMock

        from apps.api.server.routers import inference as inf

        core = MagicMock()
        core.build_context_frame = _AsyncMock(return_value=MagicMock(layers=[], system_extra="x"))

        memory_service = MagicMock()
        memory_service.stats.return_value = {"total_facts": 0}

        with patch.object(inf, "get_memory_service", return_value=memory_service):
            frame, info = asyncio.run(
                inf._build_context_frame(core, "sess", "hello", corr_id="test")
            )

        assert frame is not None
        kwargs = core.build_context_frame.call_args.kwargs
        assert kwargs["include_rag"] is False
        assert kwargs["include_memory"] is False
        assert info["system_extra_chars"] == 1
        # No facts means no session message bookkeeping either.
        core.add_message.assert_not_called()

    def test_facts_keep_both_layers_on(self):
        from unittest.mock import AsyncMock as _AsyncMock

        from apps.api.server.routers import inference as inf

        core = MagicMock()
        core.build_context_frame = _AsyncMock(return_value=MagicMock(layers=[], system_extra=""))

        memory_service = MagicMock()
        memory_service.stats.return_value = {"total_facts": 3}

        with patch.object(inf, "get_memory_service", return_value=memory_service):
            asyncio.run(inf._build_context_frame(core, "sess", "hello", corr_id="test"))

        kwargs = core.build_context_frame.call_args.kwargs
        assert kwargs["include_rag"] is True
        assert kwargs["include_memory"] is True
        core.add_message.assert_called_once()
