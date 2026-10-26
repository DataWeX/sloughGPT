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
import routers.infer as infer_mod
import routers.llm_v1 as shim
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import REGISTRY
from infrastructure.exception_handlers import register_app_error_handler
from routers.infer import EmbedResponse

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
                api_key="test-key", api_url="http://testserver", model="default"
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
