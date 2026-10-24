"""
OpenAI-compatible surface — the /v1 projection of the SloNet engine.

Lets any OpenAI SDK (OpenWebUI, opencode, curl) drive SloNet at :8000/v1::

    POST /v1/chat/completions   streaming + non-streaming
    GET  /v1/models
    POST /v1/embeddings

One generation core; HTTP is a projection (AGENTS.md "Endpoint & Transport
Rule", applied to a *foreign* protocol):

    chat        -> domain.models.get_provider("default").chat / .chat_stream
                   (the exact seam ``routers/infer.py`` projects — no second
                   generate implementation; the training-loop-duplication
                   rule applied to HTTP)
    info        -> routers.infer.InferRouter.infer_info()
    embeddings  -> routers.infer.InferRouter.infer_embed()

FOREIGN-PROTOCOL EXCEPTION (deliberate, documented):

    OpenAI clients read ``choices`` / ``data`` at the JSON top level and
    speak ``data: {...}`` SSE frames. The project envelope
    (``{status, data}``) would break every SDK, and the flat ToolParam
    projection cannot express the nested ``messages`` array — so these
    three routes serve the OpenAI wire shapes *exactly*. The capability
    descriptors below stay the single source for name / auth / idempotency
    / version / result and are registered in the ContractRegistry (so the
    capability inventory and the TS contract projections include them).
    The shared story is unchanged: ``require_auth_if_enabled``,
    ``classify_and_raise`` error classification, ``safe_audit_log``.

Tier: external utility tooling — a site surface for out-of-process clients.
It never enters the chat/agent tool registry, ``domain.tools``, or any logic
the model reasons over.
"""

from __future__ import annotations

import datetime
import logging
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

logger = logging.getLogger("slo.openai_compat")

# Inference sampling defaults — mirrored from routers/infer.py InferRequest so
# /v1 and /infer agree on what "unspecified" means.
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.85
_DEFAULT_TOP_K = 40


# ── Wire models (the OpenAI protocol itself IS the validator) ──────────


class OpenAIChatMessage(BaseModel):
    """One chat message — role/content pair, pass-through roles."""

    role: str = Field(..., description="system | user | assistant (passed through)")
    content: str | None = Field(default=None, description="Message text")


class OpenAIStreamOptions(BaseModel):
    """``stream_options`` — OpenAI's streaming extension block."""

    include_usage: bool = Field(default=False, description="Emit a final usage chunk before [DONE]")


class OpenAIChatRequest(BaseModel):
    """OpenAI /v1/chat/completions request body (phase-1 subset)."""

    model: str = Field(default="default", description="Model id (SloNet ignores routing)")
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=_DEFAULT_TOP_P, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=500, description="Ollama-style extension")
    stop: str | list[str] | None = Field(default=None, description="Accepted; phase 1 ignores")
    stream: bool = False
    stream_options: OpenAIStreamOptions | None = None


class OpenAIEmbedRequest(BaseModel):
    """OpenAI /v1/embeddings request body (single string or batch)."""

    model: str = Field(default="default", description="Model id (SloNet ignores routing)")
    input: str | list[str]


# ── Shared guards (parity with routers/infer.py) ───────────────────────


def _require_engine() -> None:
    """Same pre-flight as /infer: memory pressure first, then model status."""
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
    """The one generation seam — the same ``get_provider`` /infer projects."""
    from domain.models import get_provider

    provider = get_provider("default")
    if provider is None:
        raise_error(
            "No provider available — load a model first", "E_INFRA_REGISTRY", status_code=503
        )
    return provider


def _sampling_kwargs(req: OpenAIChatRequest) -> dict[str, Any]:
    return {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k or _DEFAULT_TOP_K,
    }


def _provider_messages(req: OpenAIChatRequest) -> list[dict[str, str]]:
    return [
        {"role": m.role, "content": m.content or ""} for m in req.messages if m.content is not None
    ]


def _estimate_tokens(text: str) -> int:
    """Word-split estimate — mirrors InferRouter.infer's own accounting."""
    return len(text.split())


def _model_id(req_model: str | None = None) -> str:
    try:
        import state as _state

        loaded = getattr(_state, "model", None)
        if loaded is not None:
            return req_model or getattr(loaded, "model_id", None) or "default"
    except ImportError:
        pass
    return req_model or "default"


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _usage(prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse_frame(payload: dict[str, Any]) -> str:
    import json

    return "data: " + json.dumps(payload) + "\n\n"


# ── Handlers ───────────────────────────────────────────────────────────


async def chat_completions(
    req: OpenAIChatRequest,
    request: Request,
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict[str, Any] | StreamingResponse:
    """POST /v1/chat/completions — stream + non-stream over the one seam."""
    _require_engine()
    provider = _provider()
    messages = _provider_messages(req)
    model_id = _model_id(req.model)
    kwargs = _sampling_kwargs(req)
    prompt_text = "\n".join(m["content"] for m in messages)

    if req.stream:
        return StreamingResponse(
            _openai_stream(req, request, provider, messages, model_id, kwargs, prompt_text),
            media_type="text/event-stream",
        )

    start = datetime.datetime.now()
    try:
        result = await provider.chat(messages, **kwargs)
    except Exception as e:
        classify_and_raise(e, source="openai.chat.completions")
    elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
    try:
        from domain.infrastructure.server_state import get_server_state

        get_server_state().record_inference(
            tokens=_estimate_tokens(result), elapsed_ms=elapsed_ms, model=req.model
        )
    except Exception as e:
        logger.warning("Failed to record inference metrics: %s", e)
    safe_audit_log(
        "openai.chat.completions",
        resource=model_id,
        detail=f"elapsed={elapsed_ms:.0f}ms stream=false",
    )
    return {
        "id": _completion_id(),
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


async def _openai_stream(
    req: OpenAIChatRequest,
    request: Request,
    provider: Any,
    messages: list[dict[str, str]],
    model_id: str,
    kwargs: dict[str, Any],
    prompt_text: str,
) -> AsyncIterator[str]:
    """OpenAI SSE: chat.completion.chunk frames, finish_reason, [DONE]."""
    cid = _completion_id()
    created = int(_time.time())
    completion_parts: list[str] = []

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    try:
        yield _sse_frame(chunk({"role": "assistant", "content": ""}, None))
        async for token in provider.chat_stream(messages, **kwargs):
            if await request.is_disconnected():
                return
            if token:
                completion_parts.append(token)
                yield _sse_frame(chunk({"content": token}, None))
        yield _sse_frame(chunk({}, "stop"))
        if req.stream_options and req.stream_options.include_usage:
            completion = "".join(completion_parts)
            yield _sse_frame(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": _usage(prompt_text, completion),
                }
            )
        yield "data: [DONE]\n\n"
    except Exception as e:
        from domain.infrastructure import classify_exception, emit_error_event

        err = classify_exception(e)
        emit_error_event(err, source="openai.chat.completions.stream")
        # Mid-stream failure: OpenAI clients read error frames in-band.
        import json

        yield (
            "data: "
            + json.dumps(
                {
                    "error": {
                        "message": err.user_message,
                        "type": err.code,
                        "code": err.code,
                    }
                }
            )
            + "\n\n"
        )
        return


async def list_models(auth_user: dict = Depends(require_auth_if_enabled)) -> dict[str, Any]:
    """GET /v1/models — one model (the loaded one), OpenAI list shape."""
    _require_engine()
    from routers.infer import InferRouter

    try:
        info = await InferRouter().infer_info()
    except Exception as e:
        classify_and_raise(e, source="openai.models.list")
    return {
        "object": "list",
        "data": [
            {
                "id": info.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "sloughgpt",
            }
        ],
    }


async def create_embeddings(
    req: OpenAIEmbedRequest,
    auth_user: dict = Depends(require_auth_if_enabled),
) -> dict[str, Any]:
    """POST /v1/embeddings — batch of strings through InferRouter.infer_embed."""
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
            data.append(
                {
                    "object": "embedding",
                    "embedding": resp.embedding,
                    "index": index,
                }
            )
            prompt_tokens += _estimate_tokens(text)
    except Exception as e:
        classify_and_raise(e, source="openai.embeddings")

    safe_audit_log(
        "openai.embeddings",
        resource=req.model or "default",
        detail=f"items={len(items)}",
    )
    return {
        "object": "list",
        "data": data,
        "model": req.model or "default",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


# ── Descriptors — one declaration per capability ───────────────────────

CHAT_SPEC = ToolSpec(
    name="openai.chat.completions",
    description=(
        "OpenAI-compatible chat completion over the SloNet generation seam "
        "(stream + non-stream). Foreign wire shape: top-level choices/usage, "
        "SSE chat.completion.chunk frames terminated by [DONE]."
    ),
    parameters=[
        ToolParam(name="model", type="string", description="Model id (routing ignored)"),
        ToolParam(
            name="messages",
            type="array",
            description="OpenAI message objects [{role, content}, ...]",
        ),
        ToolParam(name="max_tokens", type="integer", description="Sampling cap"),
        ToolParam(name="temperature", type="number", description="Sampling temperature"),
        ToolParam(name="stream", type="boolean", description="SSE streaming toggle"),
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
    name="openai.models.list",
    description="OpenAI-compatible model listing (the loaded SloNet model).",
    parameters=[],
    execute=list_models,
    auth_scope="authenticated",
    idempotent=True,
    result={"type": "object", "properties": {"data": {"type": "array"}}},
    version="1",
)

EMBEDDINGS_SPEC = ToolSpec(
    name="openai.embeddings.create",
    description="OpenAI-compatible embeddings through InferRouter.infer_embed.",
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


def _register_descriptor(spec: ToolSpec, method: str, path: str) -> None:
    """Claim the capability in the ContractRegistry (inventory + TS halves).

    Same rows ``create_router`` would write; the serving layer differs only
    because the wire shape is foreign (see module docstring).
    """
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


_register_descriptor(CHAT_SPEC, "POST", "/v1/chat/completions")
_register_descriptor(MODELS_SPEC, "GET", "/v1/models")
_register_descriptor(EMBEDDINGS_SPEC, "POST", "/v1/embeddings")

router = APIRouter(prefix="/v1", tags=["openai-compat"])
router.add_api_route(
    "/chat/completions",
    chat_completions,
    methods=["POST"],
    response_model=None,
    summary="OpenAI-compatible chat completion (stream + non-stream)",
    description=CHAT_SPEC.description,
    operation_id=f"{CHAT_SPEC.name}.post",
)
router.add_api_route(
    "/models",
    list_models,
    methods=["GET"],
    response_model=None,  # foreign wire shape — declared raw, schema in the descriptor
    summary="OpenAI-compatible model listing",
    description=MODELS_SPEC.description,
    operation_id=f"{MODELS_SPEC.name}.get",
)
router.add_api_route(
    "/embeddings",
    create_embeddings,
    methods=["POST"],
    response_model=None,
    summary="OpenAI-compatible embeddings",
    description=EMBEDDINGS_SPEC.description,
    operation_id=f"{EMBEDDINGS_SPEC.name}.post",
)
