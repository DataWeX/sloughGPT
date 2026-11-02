"""Versioned LLM API — the ``/v1`` surface every serving stack converged on.

model + messages (plus sampling knobs) in; choices + usage out; streaming as
``data: <chunk>`` deltas terminated by ``data: [DONE]``. That symmetry — not
any one vendor — is the stable interface external clients speak, so this is
native infrastructure rather than a compatibility layer.

One generation core; ``/v1`` is a projection of it (Endpoint & Transport Rule):

    chat        -> domain.models.get_provider("default").chat / .chat_stream
                   (the same seam routers/infer.py projects — no second
                   generate implementation)
    info        -> routers.infer.InferRouter.infer_info()
    embeddings  -> routers.infer.InferRouter.infer_embed()

Responses are the raw wire shape (``choices``/``data`` at the top level —
that IS the contract; the project envelope belongs to our internal routes).
The descriptors below register in the ContractRegistry, so the capability
inventory and TS contract halves include this surface. Auth
(``require_auth_if_enabled``), error classification (``classify_and_raise``;
mid-stream failures go out in-band as error frames) and ``safe_audit_log``
are the shared server story.

Non-stream requests pass a response-cache ladder before the engine: a repeated
deterministic request is replayed instead of regenerated (``_cache_key`` states
the whole rule). Streaming and stochastic requests always generate, and a
replay is tagged ``cached: true`` so it is distinguishable from a fast
generation. This is the *answer* tier — it skips the model call entirely,
where the KV session below it only skips re-prefilling.

Tier: external utility tooling — out-of-process clients only; never enters
the chat/agent tool registry or any logic the model reasons over.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time as _time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import REGISTRY, RouteEntry
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log

from domain.agents import ToolParam, ToolSpec

logger = logging.getLogger("slo.llm_v1")

# Sampling defaults — mirrored from InferRequest so /v1 and /infer agree on
# what "unspecified" means.
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.85
_DEFAULT_TOP_K = 40


# ── Wire models (the wire contract itself IS the validator) ─────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant (passed through)")
    content: str | None = Field(default=None, description="Message text")


class StreamOptions(BaseModel):
    include_usage: bool = Field(default=False, description="Emit a usage frame before [DONE]")


class ChatBody(BaseModel):
    model: str = Field(default="default", description="Model id (SloNet does not route on it)")
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=_DEFAULT_TOP_P, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=500, description="Serving-stack extension")
    stop: str | list[str] | None = Field(default=None, description="Accepted; phase 1 ignores")
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Accepted; phase 1 ignores (and a tool-carrying request is never cache-served)",
    )
    tool_choice: Any | None = Field(
        default=None,
        description="Accepted; phase 1 ignores (and a tool-carrying request is never cache-served)",
    )
    stream: bool = False
    stream_options: StreamOptions | None = None
    session_id: str | None = Field(
        default=None,
        description=(
            "Serving-session key — cross-turn KV-cache reuse (SloNet extension; "
            "absent = no reuse, fresh state per request)"
        ),
    )


class EmbedBody(BaseModel):
    model: str = Field(default="default", description="Model id (SloNet does not route on it)")
    input: str | list[str]


# ── Shared guards (parity with routers/infer.py) ────────────────────────────


def _require_engine() -> None:
    try:
        from domain.infrastructure.memory_pressure import (
            PressureLevel,
            get_memory_pressure_monitor,
        )

        if get_memory_pressure_monitor().check() == PressureLevel.EMERGENCY:
            raise_error(
                "System memory too low for inference — try again later",
                "E_MEMORY_PRESSURE",
                status_code=503,
            )
    except ImportError as exc:
        logger.debug("Memory pressure guard unavailable: %s", exc)

    from routers.inference import _get_model_status

    ms = _get_model_status()
    if not ms["ready"]:
        raise_error(ms["reason"], ms["code"], status_code=ms["status"])


def _provider():
    from domain.models import get_provider

    provider = get_provider("default")
    if provider is None:
        raise_error(
            "No provider available — load a model first", "E_INFRA_REGISTRY", status_code=503
        )
    return provider


def _sampling(req: ChatBody) -> dict[str, Any]:
    return {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k or _DEFAULT_TOP_K,
    }


def _messages(req: ChatBody) -> list[dict[str, str]]:
    return [
        {"role": m.role, "content": m.content or ""} for m in req.messages if m.content is not None
    ]


def _estimate(text: str) -> int:
    """Word-split estimate — mirrors InferRouter.infer's own accounting."""
    return len(text.split())


def _usage(prompt: str, completion: str) -> dict[str, int]:
    p, c = _estimate(prompt), _estimate(completion)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


# ── Response cache: the ladder (card 20261102_021851) ───────────────────────


def _cache_key(
    req: ChatBody,
    messages: list[dict[str, str]],
    model_id: str,
    sampling: dict[str, Any],
) -> tuple[str | None, str]:
    """Exact-tier key for this request, or ``None`` when it must generate.

    A total decision function, first match wins, and every ambiguous case falls
    to "generate": a wrong miss costs latency, while a wrong hit costs
    correctness and is invisible downstream (a replay is fluent either way).

        stream             → generate: a hit is not a stream; replay is its own
                             contract, not a substitution inside this one
        temperature > 0.2  → generate: the output is not a function of the inputs
        tools / tool_choice → generate: the real input is the tool *result*
        otherwise          → exact key (rule 6, the paraphrase tier, is
                             deliberately absent — similarity may propose a
                             candidate but must never decide one)

    Key completeness is the entire correctness argument, so every input that
    can change the output participates. ``top_k`` is keyed **as sent** because
    ``_sampling`` maps 0 to the SloNet default — keying the raw 0 would make two
    identical requests miss each other. ``stop`` is ignored by phase 1 but keyed
    anyway, so honouring it later cannot silently collide. ``session_id`` selects
    a KV state; a state that does not prefix-match falls back to a full prefill,
    which ``test_slonet_kv_cache`` pins as output-identical. The module's
    generation token already covers model load/unload/swap.
    """
    from domain.inference._internal.response_cache import ResponseCache

    if req.stream:
        return None, "stream"
    if not ResponseCache.is_deterministic(req.temperature):
        return None, "stochastic"
    if req.tools or req.tool_choice:
        return None, "tools"
    key = ResponseCache.build_key(
        messages=messages,
        model=model_id,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        system_prompt="",  # /v1 carries the system prompt as a message
        extra={
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "stop": req.stop,
            "session_id": req.session_id,
        },
    )
    return key, "cacheable"


def _completion(
    result: str, model_id: str, prompt_text: str, *, cached: bool = False
) -> dict[str, Any]:
    """The ``chat.completion`` wire shape, shared by the generated and replayed paths."""
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(prompt_text, result),
    }
    if cached:
        # Additive wire tag (clients ignore unknown keys). Without it a replay
        # is indistinguishable from a suspiciously fast generation.
        payload["cached"] = True
    return payload


def _record_inference(model_id: str, tokens: int, elapsed_ms: float) -> None:
    try:
        from domain.infrastructure.server_state import get_server_state

        get_server_state().record_inference(tokens=tokens, elapsed_ms=elapsed_ms, model=model_id)
    except Exception as e:
        logger.warning("Failed to record inference metrics: %s", e)


# ── Handlers ────────────────────────────────────────────────────────────────


async def chat_completions(
    req: ChatBody,
    request: Request,
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict[str, Any] | StreamingResponse:
    """POST /v1/chat/completions — stream + non-stream over the one seam."""
    _require_engine()
    provider = _provider()
    messages = _messages(req)
    model_id = req.model or "default"
    sampling = _sampling(req)
    kwargs = dict(sampling)
    if req.session_id:
        # Keys cross-turn KV reuse below the seam (SloNetServer._resolve_kv_state
        # -> SessionKVManager). Injected only when set — requests without it stay
        # byte-identical for every adapter provider. The stream path inherits it
        # (_stream_frames spreads kwargs into chat_stream).
        kwargs["session_id"] = req.session_id
    prompt_text = "\n".join(m["content"] for m in messages)
    start = datetime.datetime.now()

    from domain.inference._internal.response_cache import get_response_cache

    cache = get_response_cache()

    if req.stream:
        cache.record_bypass_stream()
        return StreamingResponse(
            _stream_frames(req, request, provider, messages, model_id, kwargs, prompt_text),
            media_type="text/event-stream",
        )

    # The engine gates above stay first: with no model loaded the generation
    # token has already been bumped, so every entry is stale and a lookup would
    # only add a miss (a request with nothing to serve from is not servable).
    cache_key, why = _cache_key(req, messages, model_id, sampling)
    if cache_key is None:
        if why == "stochastic":
            cache.record_bypass_stochastic()
    else:
        cached = cache.get(cache_key)
        if cached is not None:
            elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
            # tokens=0: the model ran zero tokens. Counting the replayed text
            # would inflate the tokens/s window with work nobody did.
            _record_inference(model_id, 0, elapsed_ms)
            logger.info("response cache: exact hit (%.1fms, %d chars)", elapsed_ms, len(cached))
            safe_audit_log(
                "llm.chat.completions",
                resource=model_id,
                detail=f"elapsed={elapsed_ms:.0f}ms stream=false cached=1",
            )
            return _completion(cached, model_id, prompt_text, cached=True)

    try:
        result = await provider.chat(messages, **kwargs)
    except Exception as e:
        classify_and_raise(e, source="llm.chat.completions")
    elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
    _record_inference(model_id, _estimate(result), elapsed_ms)
    safe_audit_log(
        "llm.chat.completions",
        resource=model_id,
        detail=f"elapsed={elapsed_ms:.0f}ms stream=false",
    )
    if cache_key is not None and result:
        # Exact tier only, and never fed from a replay: a semantic hit served
        # elsewhere must not be re-stored under this prompt's key, or a
        # paraphrase chain drifts the cache away from anything ever generated.
        cache.put(key=cache_key, response=result, model=model_id)
    return _completion(result, model_id, prompt_text)


async def _stream_frames(
    req: ChatBody,
    request: Request,
    provider: Any,
    messages: list[dict[str, str]],
    model_id: str,
    kwargs: dict[str, Any],
    prompt_text: str,
) -> AsyncIterator[str]:
    """Chunk frames, finish_reason=stop, optional usage frame, [DONE]."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(_time.time())
    parts: list[str] = []
    # Client-abandon signal (native /infer parity, inference.py): set in the
    # finally below on ANY non-complete exit so the model pump stops pulling
    # tokens and releases SloNetServer._model_lock. Without this an abandoned
    # SSE client keeps the generator thread holding the lock until its own
    # internal deadline — every later request then starves behind it.
    cancel_event = threading.Event()
    kwargs = {**kwargs, "cancel_event": cancel_event}
    completed = False

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    try:
        yield _frame(chunk({"role": "assistant", "content": ""}, None))
        async for token in provider.chat_stream(messages, **kwargs):
            if await request.is_disconnected():
                return
            if token:
                parts.append(token)
                yield _frame(chunk({"content": token}, None))
        yield _frame(chunk({}, "stop"))
        if req.stream_options and req.stream_options.include_usage:
            yield _frame(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": _usage(prompt_text, "".join(parts)),
                }
            )
        yield "data: [DONE]\n\n"
        completed = True
    except Exception as e:
        from domain.infrastructure import classify_exception, emit_error_event

        err = classify_exception(e)
        emit_error_event(err, source="llm.chat.completions.stream")
        # Mid-stream failure: clients read error frames in-band.
        yield _frame({"error": {"message": err.user_message, "type": err.code, "code": err.code}})
        return
    finally:
        if not completed:
            # Client disconnect / GeneratorExit / mid-stream failure. A
            # completed stream must NOT set it (the provider contract treats
            # the caller's cancel_event as a request, never a completion
            # side effect — test_cancel_event_not_set_when_fast).
            cancel_event.set()


async def list_models(auth_user: dict = Depends(require_auth_if_enabled)) -> dict[str, Any]:
    """GET /v1/models — the model list, raw wire shape.

    Discovery, not a generation gate: out-of-process clients poll this as a
    health probe, so a server whose load has not materialized answers 200
    with an empty list (nothing to serve) instead of 503. Chat and
    embeddings keep the full readiness gate.
    """
    from routers.inference import _model_ready

    if not _model_ready():
        return {"object": "list", "data": []}

    from routers.infer import InferRouter

    try:
        info = await InferRouter().infer_info()
    except Exception as e:
        classify_and_raise(e, source="llm.models.list")
    return {
        "object": "list",
        "data": [{"id": info.model_id, "object": "model", "created": 0, "owned_by": "sloughgpt"}],
    }


async def create_embeddings(
    req: EmbedBody,
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict[str, Any]:
    """POST /v1/embeddings — batch through InferRouter.infer_embed."""
    _require_engine()
    from routers.infer import EmbedRequest, InferRouter

    items = [req.input] if isinstance(req.input, str) else list(req.input)
    if not items:
        raise_error("input must contain at least one string", "E_BAD_REQUEST", status_code=422)

    router = InferRouter()
    data: list[dict[str, Any]] = []
    prompt_tokens = 0
    try:
        for index, text in enumerate(items):
            resp = await router.infer_embed(EmbedRequest(text=text, model=req.model))
            data.append({"object": "embedding", "embedding": resp.embedding, "index": index})
            prompt_tokens += _estimate(text)
    except Exception as e:
        classify_and_raise(e, source="llm.embeddings")

    safe_audit_log("llm.embeddings", resource=req.model or "default", detail=f"items={len(items)}")
    return {
        "object": "list",
        "data": data,
        "model": req.model or "default",
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ── Descriptors — one declaration per capability ────────────────────────────

CHAT_SPEC = ToolSpec(
    name="llm.chat.completions",
    description=(
        "Chat completion over the SloNet generation seam (stream + non-stream); "
        "wire: top-level choices/usage, data: chunk frames terminated by [DONE]."
    ),
    parameters=[
        ToolParam(name="model", type="string", description="Model id (routing ignored)"),
        ToolParam(
            name="messages", type="array", description="Message objects [{role, content}, ...]"
        ),
        ToolParam(name="max_tokens", type="integer", description="Sampling cap"),
        ToolParam(name="temperature", type="number", description="Sampling temperature"),
        ToolParam(name="stream", type="boolean", description="Streaming toggle"),
        ToolParam(
            name="session_id",
            type="string",
            description="Serving-session key for cross-turn KV-cache reuse (optional)",
        ),
    ],
    execute=chat_completions,
    auth_scope="authenticated",
    idempotent=False,
    result={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "object": {"type": "string"},
            "choices": {"type": "array"},
            "usage": {"type": "object"},
        },
    },
    version="1",
)

MODELS_SPEC = ToolSpec(
    name="llm.models.list",
    description="Model listing (the loaded SloNet model), wire list shape.",
    parameters=[],
    execute=list_models,
    auth_scope="authenticated",
    idempotent=True,
    result={"type": "object", "properties": {"data": {"type": "array"}}},
    version="1",
)

EMBEDDINGS_SPEC = ToolSpec(
    name="llm.embeddings.create",
    description="Embeddings through InferRouter.infer_embed, wire list shape.",
    parameters=[
        ToolParam(name="model", type="string", description="Model id (routing ignored)"),
        ToolParam(name="input", type="string", description="Text or batch of texts"),
    ],
    execute=create_embeddings,
    auth_scope="authenticated",
    idempotent=False,
    result={
        "type": "object",
        "properties": {
            "data": {"type": "array"},
            "model": {"type": "string"},
            "usage": {"type": "object"},
        },
    },
    version="1",
)


def _register(spec: ToolSpec, method: str, path: str) -> None:
    operation_id = REGISTRY.claim_operation_id(f"{spec.name}.{method.lower()}", method, path)
    REGISTRY.register(
        RouteEntry(
            name=spec.name,
            version=spec.version,
            auth_scope=spec.auth_scope,
            idempotent=spec.idempotent,
            description=spec.description,
            method=method,
            path=path,
            operation_id=operation_id,
            module=__name__,
            params=spec.params,
            result=spec.result,
        )
    )


_register(CHAT_SPEC, "POST", "/v1/chat/completions")
_register(MODELS_SPEC, "GET", "/v1/models")
_register(EMBEDDINGS_SPEC, "POST", "/v1/embeddings")

router = APIRouter(prefix="/v1", tags=["llm-api"])
router.add_api_route(
    "/chat/completions",
    chat_completions,
    methods=["POST"],
    response_model=None,  # wire contract — schema lives in the descriptor
    summary="Chat completion (stream + non-stream)",
    description=CHAT_SPEC.description,
    operation_id=f"{CHAT_SPEC.name}.post",
)
router.add_api_route(
    "/models",
    list_models,
    methods=["GET"],
    response_model=None,
    summary="Model listing",
    description=MODELS_SPEC.description,
    operation_id=f"{MODELS_SPEC.name}.get",
)
router.add_api_route(
    "/embeddings",
    create_embeddings,
    methods=["POST"],
    response_model=None,
    summary="Embeddings",
    description=EMBEDDINGS_SPEC.description,
    operation_id=f"{EMBEDDINGS_SPEC.name}.post",
)
