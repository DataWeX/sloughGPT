"""
Unified Inference Router — /infer prefix

Single language-agnostic entrypoint for all inference operations:
  POST /infer/embed    — text embedding
  POST /infer/tokenize — tokenization
  POST /infer/detokenize — detokenization
  GET  /infer/health   — engine health + model info
  GET  /infer/info     — loaded model metadata

Thin adapter: delegates to provider/domain logic, no business logic here.
"""

import logging
import time as _time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.inference_generation import generate_text, stream_text
from infrastructure.sse_fallback import sse_complete, sse_error, sse_token
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, endpoint, raise_error, safe_audit_log

from config import ServerConfig
from domain.infrastructure import AppError

logger = logging.getLogger("slo.infer")
cfg = ServerConfig.from_env()


class InferRequest(BaseModel):
    """Text generation request."""

    prompt: str = Field(..., min_length=1, max_length=50000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    model: str | None = None
    response_format: str | None = Field(default=None, description='"text" or "json"')


class InferResponse(BaseModel):
    """Generation result."""

    text: str
    model: str
    tokens_generated: int = 0
    elapsed_ms: float = 0


class EmbedRequest(BaseModel):
    """Embedding request."""

    text: str = Field(..., min_length=1, max_length=50000)
    model: str | None = None


class EmbedResponse(BaseModel):
    """Embedding result."""

    embedding: list[float]
    dimensions: int
    model: str


class TokenizeRequest(BaseModel):
    """Tokenization request."""

    text: str = Field(..., min_length=1, max_length=50000)
    model: str | None = None


class TokenizeResponse(BaseModel):
    """Tokenization result."""

    tokens: list[str]
    ids: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    """Detokenization request."""

    ids: list[int] = Field(..., max_length=10000)
    model: str | None = None


class DetokenizeResponse(BaseModel):
    """Detokenization result."""

    text: str
    count: int


class InferHealthResponse(BaseModel):
    """Engine health status."""

    status: str
    model_loaded: bool
    model_id: str | None = None
    engine_type: str | None = None
    has_streaming: bool = True
    has_embedding: bool = False


class InferInfoResponse(BaseModel):
    """Loaded model metadata."""

    model_id: str
    model_type: str
    num_parameters: int = 0
    vocab_size: int = 0
    max_context: int = 0
    num_layers: int = 0
    has_tokenizer: bool = False
    has_streaming: bool = True
    has_embedding: bool = False
    extra: dict = Field(default_factory=dict)


class InferRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/infer", tags=["infer"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.infer, methods=["POST"], response_model=InferResponse)
        self.router.add_api_route("/stream", self.infer_stream, methods=["POST"])
        self.router.add_api_route(
            "/embed", self.infer_embed, methods=["POST"], response_model=EmbedResponse
        )
        self.router.add_api_route(
            "/tokenize", self.infer_tokenize, methods=["POST"], response_model=TokenizeResponse
        )
        self.router.add_api_route(
            "/detokenize",
            self.infer_detokenize,
            methods=["POST"],
            response_model=DetokenizeResponse,
        )
        self.router.add_api_route(
            "/health", self.infer_health, methods=["GET"], response_model=InferHealthResponse
        )
        self.router.add_api_route(
            "/info", self.infer_info, methods=["GET"], response_model=InferInfoResponse
        )

    async def infer(
        self, req: InferRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> InferResponse:
        """Generate plain text without chat-specific orchestration."""
        from domain.infrastructure.server_state import get_server_state
        from domain.models import get_provider
        from routers.inference import _check_memory_pressure, _get_model_status

        status = _get_model_status()
        if not status["ready"]:
            raise_error(status["reason"], status["code"], status_code=status["status"])
        memory_error = _check_memory_pressure()
        if memory_error:
            raise_error(memory_error, "E_MEMORY_PRESSURE", status_code=503)

        provider = get_provider("default")
        if provider is None:
            raise_error("No provider available — load a model first", "E_INFRA_REGISTRY", status_code=503)

        prompt = req.prompt
        if req.response_format == "json":
            prompt += "\n\nRespond ONLY with valid JSON. No markdown, no explanation, no code fences."

        state = get_server_state()
        model_id = req.model or state.model_type.get() or "unknown"
        try:
            result = await generate_text(
                provider,
                [{"role": "user", "content": prompt}],
                max_tokens=req.max_new_tokens,
                generation_params={
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                    "repetition_penalty": req.repetition_penalty,
                },
                tokenizer=state.tokenizer.get(),
            )
            state.record_inference(
                tokens=result.tokens_generated,
                elapsed_ms=result.elapsed_ms,
                model=model_id,
            )
            return InferResponse(
                text=result.text,
                model=model_id,
                tokens_generated=result.tokens_generated,
                elapsed_ms=round(result.elapsed_ms, 1),
            )
        except AppError as exc:
            classify_and_raise(exc, source="infer.generate")
        except Exception as exc:
            classify_and_raise(exc, source="infer.generate")

    async def infer_stream(
        self,
        req: InferRequest,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Stream plain text from the shared provider generation core."""
        from domain.infrastructure.server_state import get_server_state
        from domain.models import get_provider
        from routers.inference import _check_memory_pressure, _get_model_status

        status = _get_model_status()
        if not status["ready"]:
            async def not_ready() -> AsyncIterator[str]:
                yield sse_error("infer", "IDLE", status["reason"], code=status["code"], http_status=status["status"])
            return StreamingResponse(not_ready(), media_type="text/event-stream")

        memory_error = _check_memory_pressure()
        if memory_error:
            async def memory_blocked() -> AsyncIterator[str]:
                yield sse_error("infer", "IDLE", memory_error, code="E_MEMORY_PRESSURE", http_status=503)
            return StreamingResponse(memory_blocked(), media_type="text/event-stream")

        provider = get_provider("default")
        if provider is None:
            async def provider_missing() -> AsyncIterator[str]:
                yield sse_error(
                    "infer", "IDLE", "No provider available — load a model first",
                    code="E_INFRA_REGISTRY", http_status=503,
                )
            return StreamingResponse(provider_missing(), media_type="text/event-stream")

        prompt = req.prompt
        if req.response_format == "json":
            prompt += "\n\nRespond ONLY with valid JSON. No markdown, no explanation, no code fences."
        state = get_server_state()

        async def response_stream() -> AsyncIterator[str]:
            started = _time.monotonic()
            collected: list[str] = []
            try:
                async for token in stream_text(
                    provider,
                    [{"role": "user", "content": prompt}],
                    max_tokens=req.max_new_tokens,
                    generation_params={
                        "temperature": req.temperature,
                        "top_p": req.top_p,
                        "top_k": req.top_k,
                        "repetition_penalty": req.repetition_penalty,
                    },
                ):
                    if await request.is_disconnected():
                        return
                    collected.append(token)
                    yield sse_token("infer", token)
                elapsed_ms = (_time.monotonic() - started) * 1000
                count = generate_text.count_tokens("".join(collected)) if hasattr(generate_text, "count_tokens") else len("".join(collected).split())
                state.record_inference(tokens=count, elapsed_ms=elapsed_ms, model=req.model or "")
                yield sse_complete("infer", data={"tokens_generated": count}, meta={"elapsed_ms": round(elapsed_ms, 1)})
            except Exception as exc:
                yield sse_error("infer", "ERROR", str(exc), code="E_INFERENCE", http_status=500)

        return StreamingResponse(response_stream(), media_type="text/event-stream")

    # --- Helpers ---

    def _get_model(self):
        """Get the currently loaded model from server state."""
        try:
            import state as _state

            return _state.model
        except ImportError:
            return None

    def _get_model_interface(self):
        """Get the loaded model wrapped as ModelInterface."""
        model = self._get_model()
        if model is None:
            return None
        from domain.models import ModelInterface

        if isinstance(model, ModelInterface):
            return model
        return None


    # --- Endpoints ---

    @endpoint("infer.infer_embed")
    async def infer_embed(
        self, req: EmbedRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> EmbedResponse:
        try:
            """Return embedding vector for text.

            Uses the loaded model's embed() method if available, otherwise falls back
            to the n-gram TF-IDF embedder.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None and hasattr(model, "embed"):
                model_name = req.model or getattr(model, "model_id", "unknown")
                try:
                    import numpy as np

                    vec = model.embed(req.text)
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    safe_audit_log(
                        "infer.embed",
                        resource=model_name,
                        detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)}",
                    )
                    return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
                except NotImplementedError:
                    logger.debug(
                        "Model %s does not support embed, falling back to n-gram", model_name
                    )
                except Exception as e:
                    logger.debug("Model embed failed, falling back to n-gram: %s", e)

            # Fallback: n-gram TF-IDF embedder
            try:
                import numpy as np

                from domain.inference import _ngram_embed

                vec = _ngram_embed(req.text)
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                model_name = req.model or "ngram-tfidf"
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log(
                    "infer.embed",
                    resource=model_name,
                    detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)} fallback=ngram",
                )
                return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
            except ImportError:
                raise_error("No embedding backend available", "E_BAD_REQUEST", status_code=503)

        except Exception as e:
            classify_and_raise(e, source="infer.infer_embed")

    @endpoint("infer.infer_tokenize")
    async def infer_tokenize(
        self, req: TokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> TokenizeResponse:
        try:
            """Tokenize text into token IDs and strings.

            Uses the loaded model's tokenizer if available, otherwise falls back to
            simple byte-level tokenization.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, "_tokenizer", None) or getattr(
                        model, "tokenizer", None
                    )
                    if tokenizer is not None and hasattr(tokenizer, "encode"):
                        ids = tokenizer.encode(req.text)
                        tokens = [getattr(tokenizer, "itos", {}).get(i, f"<{i}>") for i in ids]
                        model_name = req.model or "model-tokenizer"
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log(
                            "infer.tokenize",
                            resource=model_name,
                            detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}",
                        )
                        return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))
                except Exception as e:
                    logger.debug("Model tokenize failed, falling back to byte-level: %s", e)

            # Fallback: byte-level tokenization (always works, no dependencies)
            ids = list(req.text.encode("utf-8"))
            tokens = [f"b{b}" for b in ids]
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "infer.tokenize",
                resource="byte-level",
                detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}",
            )
            return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_tokenize")

    @endpoint("infer.infer_detokenize")
    async def infer_detokenize(
        self, req: DetokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> DetokenizeResponse:
        try:
            """Convert token IDs back to text."""
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, "_tokenizer", None) or getattr(
                        model, "tokenizer", None
                    )
                    if tokenizer is not None and hasattr(tokenizer, "decode"):
                        text = tokenizer.decode(req.ids)
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log(
                            "infer.detokenize",
                            resource="model",
                            detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}",
                        )
                        return DetokenizeResponse(text=text, count=len(req.ids))
                except Exception as exc:
                    logger.debug("Tokenizer decode failed: %s", exc)

            # Fallback: interpret IDs as byte values
            try:
                text = bytes(req.ids).decode("utf-8", errors="replace")
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log(
                    "infer.detokenize",
                    resource="byte-level",
                    detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}",
                )
                return DetokenizeResponse(text=text, count=len(req.ids))
            except Exception as exc:
                import logging

                logging.getLogger("slo.infer").warning("Detokenize failed: %s", exc)
                return DetokenizeResponse(text="", count=len(req.ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_detokenize")

    @endpoint("infer.infer_health")
    async def infer_health(self) -> InferHealthResponse:
        """Engine health — whether a model is loaded and what capabilities it has."""
        model = self._get_model_interface()
        if model is None:
            return InferHealthResponse(
                status="no_model",
                model_loaded=False,
                has_streaming=False,
                has_embedding=False,
            )

        info = model.info() if hasattr(model, "info") else None
        return InferHealthResponse(
            status="ready",
            model_loaded=True,
            model_id=info.model_id if info else "unknown",
            engine_type=info.model_type if info else type(model).__name__,
            has_streaming=info.has_streaming if info else hasattr(model, "generate_stream"),
            has_embedding=info.has_embedding if info else hasattr(model, "embed"),
        )

    @endpoint("infer.infer_info")
    async def infer_info(self) -> InferInfoResponse:
        """Metadata about the currently loaded model."""
        from routers.inference import _get_model_status

        ms = _get_model_status()
        if not ms["ready"]:
            raise_error(ms["reason"], ms["code"], status_code=ms["status"])

        model = self._get_model_interface()

        info = model.info() if hasattr(model, "info") else None
        if info is None:
            return InferInfoResponse(
                model_id="unknown",
                model_type=type(model).__name__,
                num_parameters=model.num_parameters() if hasattr(model, "num_parameters") else 0,
            )

        return InferInfoResponse(
            model_id=info.model_id,
            model_type=info.model_type,
            num_parameters=info.num_parameters,
            vocab_size=info.vocab_size,
            max_context=info.max_context,
            num_layers=info.num_layers,
            has_tokenizer=info.has_tokenizer,
            has_streaming=info.has_streaming,
            has_embedding=info.has_embedding,
            extra=info.extra,
        )

        # --- SSE helpers (local, avoid import issues) ---




router = InferRouter().router
