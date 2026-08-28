"""
Unified Inference Router — /infer prefix

Single language-agnostic entrypoint for all inference operations:
  POST /infer          — generate text
  POST /infer/stream   — streaming generation (SSE)
  POST /infer/embed    — text embedding
  POST /infer/tokenize — tokenization
  POST /infer/detokenize — detokenization
  GET  /infer/health   — engine health + model info
  GET  /infer/info     — loaded model metadata

Thin adapter: delegates to provider/domain logic, no business logic here.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncIterator, AsyncGenerator
import datetime
import logging
from schemas.common import raise_error, classify_and_raise, safe_audit_log
from infrastructure.auth import require_auth_if_enabled
from domains.infrastructure.errors import AppError
import time as _time

logger = logging.getLogger("slo.infer")


class InferRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., min_length=1, max_length=50000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    model: Optional[str] = None


class InferResponse(BaseModel):
    """Generation result."""
    text: str
    model: str
    tokens_generated: int = 0
    elapsed_ms: float = 0


class EmbedRequest(BaseModel):
    """Embedding request."""
    text: str = Field(..., min_length=1, max_length=50000)
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    """Embedding result."""
    embedding: List[float]
    dimensions: int
    model: str


class TokenizeRequest(BaseModel):
    """Tokenization request."""
    text: str = Field(..., min_length=1, max_length=50000)
    model: Optional[str] = None


class TokenizeResponse(BaseModel):
    """Tokenization result."""
    tokens: List[str]
    ids: List[int]
    count: int


class DetokenizeRequest(BaseModel):
    """Detokenization request."""
    ids: List[int]
    model: Optional[str] = None


class DetokenizeResponse(BaseModel):
    """Detokenization result."""
    text: str
    count: int


class InferHealthResponse(BaseModel):
    """Engine health status."""
    status: str
    model_loaded: bool
    model_id: Optional[str] = None
    engine_type: Optional[str] = None
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
        self.router.add_api_route("/embed", self.infer_embed, methods=["POST"], response_model=EmbedResponse)
        self.router.add_api_route("/tokenize", self.infer_tokenize, methods=["POST"], response_model=TokenizeResponse)
        self.router.add_api_route("/detokenize", self.infer_detokenize, methods=["POST"], response_model=DetokenizeResponse)
        self.router.add_api_route("/health", self.infer_health, methods=["GET"], response_model=InferHealthResponse)
        self.router.add_api_route("/info", self.infer_info, methods=["GET"], response_model=InferInfoResponse)

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
        from domains.models import ModelInterface
        if isinstance(model, ModelInterface):
            return model
        return None

    def _sse_event(self, stream, phase, status, data=None, meta=None, message=""):
        """SSE event helper."""
        import json
        return "data: " + json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message,
        }) + "\n\n"

    # --- Endpoints ---

    async def infer(self, req: InferRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> InferResponse:
        """Generate text from a prompt.

        Single non-streaming generation endpoint. Delegates to the active provider.
        """
        from domains.models.provider import get_provider

        if self._get_model() is None:
            raise_error("Model still loading — please wait.", "E_BAD_REQUEST", status_code=503)

        provider = get_provider("default")
        if provider is None:
            raise_error("No provider available", "E_BAD_REQUEST", status_code=503)

        provider_messages = [{"role": "user", "content": req.prompt}]
        start = datetime.datetime.now()
        try:
            result = await provider.chat(
                provider_messages,
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
            elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
            tokens = len(result.split())
            try:
                from domains.infrastructure.server_state import get_server_state
                get_server_state().record_inference(tokens=tokens, elapsed_ms=elapsed_ms, model=req.model)
            except Exception as e:
                logger.warning("Failed to record inference metrics: %s", e)
            model = self._get_model()
            model_name = req.model or (getattr(model, 'model_id', None) or type(model).__name__ if model else 'unknown')
            safe_audit_log("infer.generate", resource=model_name, detail=f"elapsed={elapsed_ms:.0f}ms tokens={tokens}")
            return InferResponse(text=result, model=model_name, tokens_generated=tokens, elapsed_ms=round(elapsed_ms, 1))
        except AppError as e:
            classify_and_raise(e, source="infer.infer")
        except Exception as e:
            logger.warning("Inference failed: %s", e)
            classify_and_raise(e, source="infer")

    async def infer_stream(self, req: InferRequest, request: Request, auth_user: dict = Depends(require_auth_if_enabled)) -> AsyncGenerator[str, None]:
        try:
            """Stream generated tokens as SSE.

            Yields standard envelope: {stream, phase, status, data: {token}, meta}.
            """
            if self._get_model() is None:
                async def error_stream() -> AsyncIterator[str]:
                    """error_stream."""
                    yield self._sse_error("infer", "IDLE", "Model still loading — please wait.", code="MODEL_LOADING", http_status=503)
                return StreamingResponse(error_stream(), media_type="text/event-stream")

            async def generate() -> AsyncIterator[str]:
                """generate."""
                from domains.models.provider import get_provider
                provider = get_provider("default")
                if provider is None:
                    yield self._sse_error("infer", "IDLE", "No provider available", code="E_INFRA_REGISTRY", http_status=503)
                    return

                provider_messages = [{"role": "user", "content": req.prompt}]
                start = datetime.datetime.now()
                token_count = 0
                _token_gen_start = time.time()
                _max_token_wait_s = getattr(cfg, "generate_timeout", 60)
                _heartbeat_interval_s = 10.0
                _last_heartbeat = time.time()
                try:
                    async for token in provider.chat_stream(
                        provider_messages,
                        max_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                    ):
                        if await request.is_disconnected():
                            return
                        if token:
                            _token_gen_start = time.time()
                            token_count += 1
                            yield self._sse_token("infer", token)
                        else:
                            now = time.time()
                            if now - _last_heartbeat >= _heartbeat_interval_s:
                                yield ": heartbeat\n\n"
                                _last_heartbeat = now
                        elapsed_since_token = time.time() - _token_gen_start
                        if elapsed_since_token > _max_token_wait_s:
                            logger.warning(
                                "Infer stream stalled for %.1fs (limit=%.1fs)",
                                elapsed_since_token, _max_token_wait_s,
                                extra={"tag": "INF"},
                            )
                            yield self._sse_error("infer", "TIMEOUT", f"Generation stalled for {elapsed_since_token:.0f}s", code="MODEL_TIMEOUT", http_status=504)
                            return
                except Exception as e:
                    from domains.infrastructure.errors import classify_exception, emit_error_event
                    err = classify_exception(e)
                    emit_error_event(err, source="infer_stream")
                    yield self._sse_error("infer", err.code, err.user_message)
                    return
                elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
                try:
                    from domains.infrastructure.server_state import get_server_state
                    get_server_state().record_inference(tokens=token_count, elapsed_ms=elapsed_ms, model=req.model)
                except Exception as e:
                    logger.warning("Failed to record inference metrics: %s", e)
                safe_audit_log("infer.stream", resource=req.model or "default", detail=f"elapsed={elapsed_ms:.0f}ms tokens={token_count}")
                import json
                yield "data: " + json.dumps({
                    "stream": "infer", "phase": "STREAMING", "status": "complete",
                    "data": {}, "meta": {"tokens": token_count, "elapsed_ms": round(elapsed_ms, 1)},
                }) + "\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="infer.infer_stream")
    async def infer_embed(self, req: EmbedRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> EmbedResponse:
        try:
            """Return embedding vector for text.

            Uses the loaded model's embed() method if available, otherwise falls back
            to the n-gram TF-IDF embedder.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None and hasattr(model, 'embed'):
                try:
                    import numpy as np
                    vec = model.embed(req.text)
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    model_name = req.model or getattr(model, 'model_id', 'unknown')
                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    safe_audit_log("infer.embed", resource=model_name, detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)}")
                    return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
                except NotImplementedError:
                    pass
                except Exception as e:
                    logger.debug("Model embed failed, falling back to n-gram: %s", e)

            # Fallback: n-gram TF-IDF embedder
            try:
                from domains.inference.vector_store import _ngram_embed
                import numpy as np
                vec = _ngram_embed(req.text)
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                model_name = req.model or "ngram-tfidf"
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log("infer.embed", resource=model_name, detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)} fallback=ngram")
                return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
            except ImportError:
                raise_error("No embedding backend available", "E_BAD_REQUEST", status_code=503)

        except Exception as e:
            classify_and_raise(e, source="infer.infer_embed")
    async def infer_tokenize(self, req: TokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> TokenizeResponse:
        try:
            """Tokenize text into token IDs and strings.

            Uses the loaded model's tokenizer if available, otherwise falls back to
            simple byte-level tokenization.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, '_tokenizer', None) or getattr(model, 'tokenizer', None)
                    if tokenizer is not None and hasattr(tokenizer, 'encode'):
                        ids = tokenizer.encode(req.text)
                        tokens = [getattr(tokenizer, 'itos', {}).get(i, f"<{i}>") for i in ids]
                        model_name = req.model or "model-tokenizer"
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log("infer.tokenize", resource=model_name, detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}")
                        return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))
                except Exception as e:
                    logger.debug("Model tokenize failed, falling back to byte-level: %s", e)

            # Fallback: byte-level tokenization (always works, no dependencies)
            ids = list(req.text.encode("utf-8"))
            tokens = [f"b{b}" for b in ids]
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("infer.tokenize", resource="byte-level", detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}")
            return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_tokenize")
    async def infer_detokenize(self, req: DetokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> DetokenizeResponse:
        try:
            """Convert token IDs back to text."""
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, '_tokenizer', None) or getattr(model, 'tokenizer', None)
                    if tokenizer is not None and hasattr(tokenizer, 'decode'):
                        text = tokenizer.decode(req.ids)
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log("infer.detokenize", resource="model", detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}")
                        return DetokenizeResponse(text=text, count=len(req.ids))
                except Exception as exc:
                    logger.debug("Tokenizer decode failed: %s", exc)

            # Fallback: interpret IDs as byte values
            try:
                text = bytes(req.ids).decode("utf-8", errors="replace")
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log("infer.detokenize", resource="byte-level", detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}")
                return DetokenizeResponse(text=text, count=len(req.ids))
            except Exception:
                return DetokenizeResponse(text="", count=len(req.ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_detokenize")
    async def infer_health(self) -> InferHealthResponse:
        try:
            """Engine health — whether a model is loaded and what capabilities it has."""
            model = self._get_model_interface()
            if model is None:
                return InferHealthResponse(
                    status="no_model",
                    model_loaded=False,
                    has_streaming=False,
                    has_embedding=False,
                )

            info = model.info() if hasattr(model, 'info') else None
            return InferHealthResponse(
                status="ready",
                model_loaded=True,
                model_id=info.model_id if info else "unknown",
                engine_type=info.model_type if info else type(model).__name__,
                has_streaming=info.has_streaming if info else hasattr(model, 'generate_stream'),
                has_embedding=info.has_embedding if info else hasattr(model, 'embed'),
            )

        except Exception as e:
            classify_and_raise(e, source="infer.infer_health")
    async def infer_info(self) -> InferInfoResponse:
        try:
            """Metadata about the currently loaded model."""
            model = self._get_model_interface()
            if model is None:
                raise_error("No model loaded", "E_BAD_REQUEST", status_code=503)

            info = model.info() if hasattr(model, 'info') else None
            if info is None:
                return InferInfoResponse(
                    model_id="unknown",
                    model_type=type(model).__name__,
                    num_parameters=model.num_parameters() if hasattr(model, 'num_parameters') else 0,
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

        except Exception as e:
            classify_and_raise(e, source="infer.infer_info")
    def _sse_token(self, stream: str, token: str, done: bool = False, meta: dict = None, elapsed_ms: float = None) -> str:
        phase = "STREAMING"
        status = "complete" if done else "working"
        m = dict(meta) if meta else {}
        if done and elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
        return self._sse_event(stream, phase, status, {"token": token}, m, "")

    def _sse_error(self, stream: str, phase: str, error: str, meta: dict = None, code: str = None, http_status: int = None) -> str:
        data = {"error": error}
        if code is not None:
            data["code"] = code
        if http_status is not None:
            data["http_status"] = http_status
        return self._sse_event(stream, phase, "error", data, meta or {}, f"Error: {error}")


router = InferRouter().router