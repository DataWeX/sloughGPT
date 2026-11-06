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

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
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

    @endpoint("infer.generate")

    @endpoint("infer.generate_stream")

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

    @endpoint("infer.infer")

    @endpoint("infer.infer_stream")

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
