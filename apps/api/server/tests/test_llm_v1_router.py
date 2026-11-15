"""Tests for the /v1 LLM surface — wire contract + dogfood (card 5959477f).

/v1 serves the raw wire contract every serving stack converged on:
``choices`` / ``data`` at the JSON top level (the project envelope belongs to
our internal routes, not this surface). These tests pin exactly that, plus:

  * the reuse seams (``provider.chat`` / ``provider.chat_stream`` — the same
    generation seam ``/infer`` projects; ``InferRouter.infer_info`` /
    ``InferRouter.infer_embed``);
  * SSE frame protocol (chat.completion.chunk, finish_reason=stop, usage chunk
    on ``stream_options.include_usage``, ``data: [DONE]`` terminator, in-band
    error frames for mid-stream failures);
  * engine guards at /infer parity (memory pressure → 503, model-not-ready →
    503);
  * the ContractRegistry rows for the descriptors ``llm.chat.completions``,
    ``llm.models.list``, ``llm.embeddings.create``;
  * the card's dogfood round-trip: SloNet's own ``ApiProvider`` (the
    standard-wire LLM client SloNet ships) driving SloNet's own /v1 through the ASGI app.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import routers.llm_v1 as shim
import routers.models as infer_mod
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import REGISTRY
from infrastructure.exception_handlers import register_app_error_handler
from routers.models import EmbedResponse

from domain.inference._internal import api_provider as api_provider_mod
from domain.infrastructure.memory_pressure import PressureLevel

_AUTH = {"sub": "u1", "tenant_id": "t1"}
_EMBEDDING = [0.1, 0.2, 0.3]


def _build_app() -> FastAPI:
    app = FastAPI()
    register_app_error_handler(app)
    app.include_router(shim.router)
    app.dependency_overrides[require_auth_if_enabled] = lambda: _AUTH
    return app


def _fake_provider(*, fail_stream: bool = False) -> MagicMock:
    provider = MagicMock()
    provider.calls: list[tuple[list, dict]] = []

    async def _chat(messages, **kwargs):
        provider.calls.append((list(messages), dict(kwargs)))
        return "Hello from SloNet"

    async def _stream(messages, **kwargs):
        provider.calls.append((list(messages), dict(kwargs)))
        for token in ("Hello", " ", "SloNet"):
            yield token
        if fail_stream:
            raise RuntimeError("engine died mid-stream")

    provider.chat = _chat
    provider.chat_stream = _stream
    return provider


@contextmanager
def _engine(
    *,
    provider=None,
    status: dict | None = None,
    pressure: PressureLevel = PressureLevel.NORMAL,
    model_ready: bool = True,
    infer_info=None,
    infer_embed=None,
):
    """Patch the reuse seams to a deterministic engine (parity with /infer)."""
    monitor = MagicMock()
    monitor.check.return_value = pressure or PressureLevel.NORMAL
    with ExitStack() as stack:
        stack.enter_context(
            patch("routers.inference._get_model_status", return_value=status or {"ready": True})
        )
        # /v1/models reads the materialization probe directly; without this the
        # suite's answer would depend on whatever a prior test left loaded.
        stack.enter_context(patch("routers.inference._model_ready", return_value=model_ready))
        stack.enter_context(
            patch(
                "domain.infrastructure.memory_pressure.get_memory_pressure_monitor",
                return_value=monitor,
            )
        )
        if provider is not None:
            stack.enter_context(patch("domain.models.get_provider", return_value=provider))
        if infer_info is not None:
            stack.enter_context(
                patch.object(
                    infer_mod.InferRouter, "infer_info", new=AsyncMock(return_value=infer_info)
                )
            )
        if infer_embed is not None:
            stack.enter_context(
                patch.object(
                    infer_mod.InferRouter, "infer_embed", new=AsyncMock(return_value=infer_embed)
                )
            )
        yield


def _post_chat(app: FastAPI, **overrides):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    body.update(overrides)
    return TestClient(app).post("/v1/chat/completions", json=body)


# ── /v1/chat/completions ────────────────────────────────────────────────────


class TestChatCompletionsWireShape:
    def test_nonstream_is_raw_wire_shape_no_envelope(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app)
        assert resp.status_code == 200
        body = resp.json()
        # Wire contract: NO project envelope at the top level.
        assert "status" not in body
        assert "data" not in body
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert isinstance(body["created"], int)
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"] == {"role": "assistant", "content": "Hello from SloNet"}
        assert choice["finish_reason"] == "stop"
        assert body["usage"] == {
            "prompt_tokens": 1,  # "hi"
            "completion_tokens": 3,  # "Hello from SloNet"
            "total_tokens": 4,
        }

    def test_nonstream_reuses_provider_seam_with_adapted_messages(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(
                app,
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                ],
            )
        messages, kwargs = provider.calls[0]
        assert messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        assert kwargs["max_tokens"] == 16
        assert kwargs["temperature"] == 0.7  # InferRequest-default parity
        assert kwargs["top_p"] == 0.85
        assert kwargs["top_k"] == 40  # top_k=0 → SloNet default

    def test_stream_emits_chunk_frames_usage_and_done(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, stream=True, stream_options={"include_usage": True})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        frames = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        parsed = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
        assert all(c["object"] == "chat.completion.chunk" for c in parsed)
        # first frame announces the role (wire convention)
        assert parsed[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in parsed if c.get("choices")
        )
        assert content == "Hello SloNet"
        finish = [c for c in parsed if c["choices"] and c["choices"][0]["finish_reason"] == "stop"]
        assert finish, "a chunk must carry finish_reason=stop"
        usage_chunks = [c for c in parsed if isinstance(c.get("usage"), dict)]
        assert usage_chunks, "stream_options.include_usage must emit a usage chunk"
        assert usage_chunks[-1]["usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 2,  # "Hello SloNet"
            "total_tokens": 3,
        }

    def test_stream_without_include_usage_has_no_usage_chunk(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, stream=True)
        frames = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
        parsed = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
        assert not any(isinstance(c.get("usage"), dict) for c in parsed)

    def test_midstream_failure_goes_out_as_inband_error_frame(self):
        provider = _fake_provider(fail_stream=True)
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, stream=True)
        frames = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
        error_frames = [f for f in frames if '"error"' in f]
        assert error_frames, "mid-stream failure must emit an in-band error frame"
        err = json.loads(error_frames[0])["error"]
        assert err["type"]  # classified error code
        assert err["message"]

    def test_request_validation_rejects_missing_messages(self):
        app = _build_app()
        with _engine(provider=_fake_provider()):
            resp = TestClient(app).post("/v1/chat/completions", json={"model": "default"})
        assert resp.status_code == 422  # pydantic rejects — the wire schema is the validator


# ── session_id passthrough (cross-turn KV reuse — card 20261029_035756) ──────


class TestSessionIdPassthrough:
    """``session_id`` must reach the provider seam — it keys cross-turn KV reuse.

    SloNetServer._resolve_kv_state: no session_id => fresh state every call,
    i.e. every /v1 multi-turn client re-prefills the whole prefix. The
    machinery below the seam is fully built (SessionKVManager, LRU+TTL) and
    the internal /chat route already feeds it; /v1 must too. Pydantic ignores
    extra body fields, so without this field clients that DO send session_id
    had it silently discarded.
    """

    def test_nonstream_passes_session_id_to_provider(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, session_id="sess-abc")
        assert resp.status_code == 200
        _, kwargs = provider.calls[0]
        assert kwargs["session_id"] == "sess-abc"

    def test_stream_passes_session_id_to_provider(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, stream=True, session_id="sess-abc")
        assert resp.status_code == 200
        _, kwargs = provider.calls[0]
        assert kwargs["session_id"] == "sess-abc"

    def test_absent_session_id_is_not_injected(self):
        # Requests without it stay byte-identical for every adapter provider —
        # no session_id=None key pollution into kwargs.
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app)
        _, kwargs = provider.calls[0]
        assert "session_id" not in kwargs

    def test_descriptor_documents_session_id(self):
        # The descriptor is the contract source of truth — the wire field must
        # appear in CHAT_SPEC parameters (TS contract halves regenerate from it).
        assert any(p.name == "session_id" for p in shim.CHAT_SPEC.parameters)
        assert "session_id" in shim.CHAT_SPEC.params["properties"]  # JSON-Schema view derives it


# ── Stream cancellation (lock hygiene — native /infer parity) ───────────────


class TestStreamCancelEvent:
    """The stream passes a cancel_event the shim sets on any non-complete exit.

    SloNetServer serializes generation behind ``_model_lock``; an abandoned SSE
    client otherwise leaves the pump thread generating (and holding the lock)
    — every later request starves behind it. Native /infer creates the event
    (inference.py); the /v1 shim must do the same.
    """

    def test_stream_passes_event_left_unset_on_completion(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            resp = _post_chat(app, stream=True)
        assert resp.status_code == 200
        _, kwargs = provider.calls[0]
        cancel_event = kwargs["cancel_event"]
        assert isinstance(cancel_event, threading.Event)
        # Completed stream: the caller's event is a request, never a completion
        # side effect (provider contract, test_cancel_event_not_set_when_fast).
        assert not cancel_event.is_set()

    def test_abandoned_stream_sets_event(self):
        provider = _fake_provider()
        request = MagicMock()
        request.is_disconnected = AsyncMock(return_value=False)
        gen = shim._stream_frames(
            req=SimpleNamespace(stream=True, stream_options=None),
            request=request,
            provider=provider,
            messages=[{"role": "user", "content": "hi"}],
            model_id="default",
            kwargs={"max_tokens": 16},
            prompt_text="hi",
        )

        async def _run():
            frames = []
            async for frame in gen:
                frames.append(frame)
                if len(frames) == 2:  # role frame + first token frame
                    break
            await gen.aclose()  # caller walks away mid-stream
            _, kwargs = provider.calls[0]
            return kwargs

        kwargs = asyncio.run(_run())
        assert kwargs["cancel_event"].is_set()


# ── /v1/models ──────────────────────────────────────────────────────────────


class TestModelsList:
    def test_models_is_raw_wire_list_shape(self):
        app = _build_app()
        with _engine(infer_info=SimpleNamespace(model_id="slonet-test")):
            resp = TestClient(app).get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" not in body
        assert body["object"] == "list"
        assert body["data"] == [
            {"id": "slonet-test", "object": "model", "created": 0, "owned_by": "sloughgpt"}
        ]

    def test_models_is_discovery_not_a_generation_gate(self):
        """No materialized model -> 200 with an empty list, never 503.

        Clients poll /v1/models as a health probe; a 503 here (the old
        ``_require_engine()`` gate) spammed two ERROR lines per probe while
        telling the client nothing it could not learn from an empty list.
        """
        app = _build_app()
        status = {
            "ready": False,
            "reason": "No local weights for Qwen/Qwen2.5-0.5B-Instruct — SLO_BOOTSTRAP_HF is off",
            "code": "E_MODEL_LOAD_FAILED",
            "status": 503,
        }
        with _engine(status=status, model_ready=False):
            resp = TestClient(app).get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"object": "list", "data": []}


# ── /v1/embeddings ──────────────────────────────────────────────────────────


class TestEmbeddings:
    def test_batch_shape(self):
        app = _build_app()
        emb = EmbedResponse(embedding=_EMBEDDING, dimensions=3, model="default")
        with _engine(infer_embed=emb):
            resp = TestClient(app).post(
                "/v1/embeddings", json={"model": "default", "input": ["aa", "bb"]}
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "status" not in body
        assert body["object"] == "list"
        assert body["data"] == [
            {"object": "embedding", "embedding": _EMBEDDING, "index": 0},
            {"object": "embedding", "embedding": _EMBEDDING, "index": 1},
        ]
        assert body["model"] == "default"
        assert body["usage"] == {"prompt_tokens": 2, "total_tokens": 2}

    def test_single_string_input(self):
        app = _build_app()
        emb = EmbedResponse(embedding=_EMBEDDING, dimensions=3, model="default")
        with _engine(infer_embed=emb):
            resp = TestClient(app).post("/v1/embeddings", json={"input": "hi"})
        body = resp.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["index"] == 0


# ── Engine guards (parity with /infer) ──────────────────────────────────────


class TestEngineGuards:
    def test_model_not_ready_returns_503(self):
        status = {
            "ready": False,
            "reason": "model loading",
            "code": "E_MODEL_LOADING",
            "status": 503,
        }
        app = _build_app()
        with _engine(status=status):
            resp = _post_chat(app)
        assert resp.status_code == 503

    def test_memory_emergency_returns_503(self):
        app = _build_app()
        with _engine(pressure=PressureLevel.EMERGENCY):
            resp = _post_chat(app)
        assert resp.status_code == 503


# ── ContractRegistry rows ───────────────────────────────────────────────────


class TestRegistryProjection:
    def test_descriptors_registered_with_honest_rows(self):
        routes = {(r["method"], r["path"]): r for r in REGISTRY.routes(prefix="/v1")}
        for name, method, path in (
            ("llm.chat.completions", "POST", "/v1/chat/completions"),
            ("llm.models.list", "GET", "/v1/models"),
            ("llm.embeddings.create", "POST", "/v1/embeddings"),
        ):
            entry = routes[(method, path)]
            assert entry["name"] == name
            assert entry["auth_scope"] == "authenticated"
            assert entry["module"] == "routers.llm_v1"


# ── Dogfood: SloNet's own ApiProvider drives SloNet's own /v1 ───────────────


class TestApiProviderDogfood:
    def _patched_httpx(self, app: FastAPI):
        """Route ApiProvider's clients through the ASGI app.

        Async paths use ``httpx.AsyncClient`` + ``ASGITransport``; the sync
        path (``test_connection``) uses Starlette's ``TestClient`` — itself an
        ``httpx.Client`` over the same ASGI app.
        """
        ns = MagicMock()
        ns.AsyncClient = lambda **kw: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), **kw
        )
        ns.Client = lambda **kw: TestClient(app)  # TestClient takes no timeout kwarg
        return patch.object(api_provider_mod, "httpx", ns)

    def test_chat_stream_models_round_trip(self):
        app = _build_app()
        provider = _fake_provider()
        with (
            _engine(
                provider=provider,
                infer_info=SimpleNamespace(model_id="default"),
                infer_embed=EmbedResponse(embedding=_EMBEDDING, dimensions=3, model="default"),
            ),
            self._patched_httpx(app),
        ):
            client = api_provider_mod.ApiProvider(
                api_key="", api_url="http://testserver", model="default"
            )

            # models: the client's own health check reads /v1/models
            conn = client.test_connection()
            assert conn["status"] == "connected"
            assert conn["model_available"] is True
            assert "default" in conn["available_models"]

            # chat: non-streaming round-trip
            text = asyncio.run(client.chat([{"role": "user", "content": "hi"}], max_tokens=8))
            assert text == "Hello from SloNet"

            # chat: streaming round-trip + usage accounting
            async def _collect():
                return [
                    tok
                    async for tok in client.chat_stream(
                        [{"role": "user", "content": "hi"}], max_tokens=8
                    )
                ]

            assert asyncio.run(_collect()) == ["Hello", " ", "SloNet"]
            assert client.last_usage == {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            }

            # embeddings: raw wire call (ApiProvider.embed is a stub by
            # design — the wire shape is what the dogfood pins here)
            emb = TestClient(app).post("/v1/embeddings", json={"input": "hi"})
            assert emb.json()["data"][0]["embedding"] == _EMBEDDING


# ── response cache: the ladder (card 20261102_021851) ───────────────────────


class TestResponseCacheLadder:
    """The /v1 answer tier: a repeated deterministic request is replayed.

    The ladder is a total function and every ambiguous case falls to
    "generate", because a wrong miss costs latency while a wrong hit costs
    correctness and is invisible downstream (a replay is fluent either way).
    These pin each rung — including the rung that is deliberately absent: rule
    6, the paraphrase tier. Re-enabling it has to be an argument, not a flag
    flip.
    """

    def setup_method(self):
        from domain.inference._internal.response_cache import reset_response_cache_for_tests

        reset_response_cache_for_tests()

    @staticmethod
    def _stats() -> dict:
        from domain.inference._internal.response_cache import get_response_cache

        return get_response_cache().get_stats()

    def test_repeated_deterministic_request_is_replayed(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            first = _post_chat(app, temperature=0.0)
            second = _post_chat(app, temperature=0.0)

        assert len(provider.calls) == 1, "second identical request reached the engine"
        generated = first.json()
        replay = second.json()
        assert generated["choices"][0]["message"]["content"] == "Hello from SloNet"
        assert replay["cached"] is True
        assert replay["object"] == "chat.completion"
        assert replay["choices"] == generated["choices"]
        assert replay["usage"] == generated["usage"]
        assert self._stats()["hits"] == 1

    def test_replay_is_tagged_but_generation_is_not(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            generated = _post_chat(app, temperature=0.0)
            replay = _post_chat(app, temperature=0.0)

        assert "cached" not in generated.json()
        assert replay.json()["cached"] is True

    def test_stochastic_request_always_generates(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.9)
            second = _post_chat(app, temperature=0.9)

        assert len(provider.calls) == 2
        assert "cached" not in second.json()
        assert self._stats()["bypass_stochastic"] == 2

    def test_streaming_neither_reads_nor_fills_the_cache(self):
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0, stream=True)
            nonstream = _post_chat(app, temperature=0.0)

        assert len(provider.calls) == 2, "a stream must not seed the exact tier"
        assert "cached" not in nonstream.json()
        assert self._stats()["bypass_stream"] == 1

    def test_top_p_is_in_the_key(self):
        """Sampling params that change the output must change the key."""
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0, top_p=0.5)
            _post_chat(app, temperature=0.0, top_p=0.9)

        assert len(provider.calls) == 2

    def test_top_k_is_keyed_as_sent(self):
        """top_k=0 is mapped to the SloNet default before the engine sees it.

        Keying the raw 0 would make two requests the engine cannot distinguish
        miss each other; keying the effective value keeps the key honest.
        """
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0, top_k=0)
            equivalent = _post_chat(app, temperature=0.0, top_k=40)

        assert len(provider.calls) == 1
        assert equivalent.json()["cached"] is True

        from domain.inference._internal.response_cache import (
            reset_response_cache_for_tests,
        )

        reset_response_cache_for_tests()
        reset = _fake_provider()
        with _engine(provider=reset):
            _post_chat(app, temperature=0.0, top_k=40)
            _post_chat(app, temperature=0.0, top_k=20)

        assert len(reset.calls) == 2

    def test_session_id_is_in_the_key(self):
        """A session selects a KV state; two sessions are not one request."""
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0, session_id="a")
            _post_chat(app, temperature=0.0, session_id="b")
            same = _post_chat(app, temperature=0.0, session_id="a")

        assert len(provider.calls) == 2
        assert same.json()["cached"] is True

    def test_model_swap_invalidates_the_cache(self):
        from domain.inference._internal.response_cache import bump_cache_generation

        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0)
            bump_cache_generation()
            after = _post_chat(app, temperature=0.0)

        assert len(provider.calls) == 2, "a model swap must not replay the old weights"
        assert "cached" not in after.json()

    def test_tool_carrying_request_is_never_replayed(self):
        """The real input of a tool turn is the tool *result*, not the prompt."""
        provider = _fake_provider()
        app = _build_app()
        tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
        with _engine(provider=provider):
            body = {"temperature": 0.0, "tools": tools}
            _post_chat(app, **body)
            _post_chat(app, **body)

        assert len(provider.calls) == 2

    def test_paraphrase_is_not_promoted(self):
        """Rule 6 is deliberately absent: similarity may propose, never decide.

        Two prompts that share their content words but ask different questions
        must both hit the engine. If this ever starts passing as a replay,
        someone wired the semantic tier into /v1 — that needs a verifier and an
        explicit decision, and this test is where that argument starts.
        """
        provider = _fake_provider()
        app = _build_app()
        with _engine(provider=provider):
            _post_chat(app, temperature=0.0, messages=[{"role": "user", "content": "2+3"}])
            second = _post_chat(
                app, temperature=0.0, messages=[{"role": "user", "content": "4+12"}]
            )

        assert len(provider.calls) == 2
        assert "cached" not in second.json()


class TestCacheObservability:
    def test_health_summary_snapshot_carries_cache_stats(self):
        from routers.health import _response_cache_snapshot

        stats = _response_cache_snapshot()
        assert "hit_rate" in stats
        assert "generation" in stats
        assert "bypassed" not in stats  # the real names, not a placeholder
        assert {"hits", "misses", "bypass_stochastic", "bypass_stream"} <= set(stats)

    def test_snapshot_never_breaks_health(self):
        """A cache that cannot report must not take the status bar down."""
        with patch(
            "domain.inference._internal.response_cache.get_response_cache",
            side_effect=RuntimeError("boom"),
        ):
            from routers.health import _response_cache_snapshot

            assert _response_cache_snapshot() == {}
