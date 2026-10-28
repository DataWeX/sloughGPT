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

Tier: external utility tooling — out-of-process clients only; never enters
the chat/agent tool registry or any logic the model reasons over.
"""

from __future__ import annotations

import datetime
import json
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
    stream: bool = False
    stream_options: StreamOptions | None = None


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
    kwargs = _sampling(req)
    prompt_text = "\n".join(m["content"] for m in messages)

    if req.stream:
        return StreamingResponse(
            _stream_frames(req, request, provider, messages, model_id, kwargs, prompt_text),
            media_type="text/event-stream",
        )

    start = datetime.datetime.now()
    try:
        result = await provider.chat(messages, **kwargs)
    except Exception as e:
        classify_and_raise(e, source="llm.chat.completions")
    elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
    try:
        from domain.infrastructure.server_state import get_server_state

        get_server_state().record_inference(
            tokens=_estimate(result), elapsed_ms=elapsed_ms, model=model_id
        )
    except Exception as e:
        logger.warning("Failed to record inference metrics: %s", e)
    safe_audit_log(
        "llm.chat.completions",
        resource=model_id,
        detail=f"elapsed={elapsed_ms:.0f}ms stream=false",
    )
    return {
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
    except Exception as e:
        from domain.infrastructure import classify_exception, emit_error_event

        err = classify_exception(e)
        emit_error_event(err, source="llm.chat.completions.stream")
        # Mid-stream failure: clients read error frames in-band.
        yield _frame({"error": {"message": err.user_message, "type": err.code, "code": err.code}})
        return


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
