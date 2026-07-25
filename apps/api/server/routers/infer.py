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
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncIterator
import datetime
import logging

logger = logging.getLogger("slo.infer")

router = APIRouter(prefix="/infer", tags=["infer"])

# --- Schemas ---


class InferRequest(BaseModel):
    """Text generation request."""
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.2, ge=0.5, le=2.0)
    model: Optional[str] = None


class InferResponse(BaseModel):
    """Generation result."""
    text: str
    model: str
    tokens_generated: int = 0
    elapsed_ms: float = 0


class EmbedRequest(BaseModel):
    """Embedding request."""
    text: str
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    """Embedding result."""
    embedding: List[float]
    dimensions: int
    model: str


class TokenizeRequest(BaseModel):
    """Tokenization request."""
    text: str
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
    extra: dict = {}


# --- Helpers ---


def _get_model():
    """Get the currently loaded model from server state."""
    try:
        import state as _state
        return _state.model
    except ImportError:
        return None


def _get_model_interface():
    """Get the loaded model wrapped as ModelInterface."""
    model = _get_model()
    if model is None:
        return None
    from domains.models import ModelInterface
    if isinstance(model, ModelInterface):
        return model
    return None


def _sse_event(stream, phase, status, data=None, meta=None, message=""):
    """SSE event helper."""
    import json
    return "data: " + json.dumps({
        "stream": stream, "phase": phase, "status": status,
        "data": data or {}, "meta": meta or {}, "message": message,
    }) + "\n\n"


# --- Endpoints ---


@router.post("", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    """Generate text from a prompt.

    Single non-streaming generation endpoint. Delegates to the active provider.
    """
    from domains.models.provider import get_provider

    if _get_model() is None:
        raise HTTPException(status_code=503, detail="Model still loading — please wait.")

    provider = get_provider("default")
    if provider is None:
        raise HTTPException(status_code=503, detail="No provider available")

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
        except Exception:
            pass
        model = _get_model()
        model_name = req.model or (getattr(model, 'model_id', None) or type(model).__name__ if model else 'unknown')
        return InferResponse(text=result, model=model_name, tokens_generated=tokens, elapsed_ms=round(elapsed_ms, 1))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def infer_stream(req: InferRequest, request: Request) -> StreamingResponse:
    """Stream generated tokens as SSE.

    Yields standard envelope: {stream, phase, status, data: {token}, meta}.
    """
    if _get_model() is None:
        async def error_stream() -> AsyncIterator[str]:
            yield _sse_error("infer", "IDLE", "Model still loading — please wait.")
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate() -> AsyncIterator[str]:
        from domains.models.provider import get_provider
        provider = get_provider("default")
        if provider is None:
            yield _sse_error("infer", "IDLE", "No provider available")
            return

        provider_messages = [{"role": "user", "content": req.prompt}]
        start = datetime.datetime.now()
        token_count = 0
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
                    token_count += 1
                    yield _sse_token("infer", token)
        except Exception as e:
            yield _sse_error("infer", "STREAMING", str(e))
            return
        elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
        try:
            from domains.infrastructure.server_state import get_server_state
            get_server_state().record_inference(tokens=token_count, elapsed_ms=elapsed_ms, model=req.model)
        except Exception:
            pass
        import json
        yield "data: " + json.dumps({
            "stream": "infer", "phase": "STREAMING", "status": "complete",
            "data": {}, "meta": {"tokens": token_count, "elapsed_ms": round(elapsed_ms, 1)},
        }) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/embed", response_model=EmbedResponse)
async def infer_embed(req: EmbedRequest) -> EmbedResponse:
    """Return embedding vector for text.

    Uses the loaded model's embed() method if available, otherwise falls back
    to the n-gram TF-IDF embedder.
    """
    model = _get_model_interface()
    if model is not None and hasattr(model, 'embed'):
        try:
            import numpy as np
            vec = model.embed(req.text)
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            model_name = req.model or getattr(model, 'model_id', 'unknown')
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
        return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
    except ImportError:
        raise HTTPException(status_code=503, detail="No embedding backend available")


@router.post("/tokenize", response_model=TokenizeResponse)
async def infer_tokenize(req: TokenizeRequest) -> TokenizeResponse:
    """Tokenize text into token IDs and strings.

    Uses the loaded model's tokenizer if available, otherwise falls back to
    simple byte-level tokenization.
    """
    model = _get_model_interface()
    if model is not None:
        try:
            tokenizer = getattr(model, '_tokenizer', None) or getattr(model, 'tokenizer', None)
            if tokenizer is not None and hasattr(tokenizer, 'encode'):
                ids = tokenizer.encode(req.text)
                tokens = [getattr(tokenizer, 'itos', {}).get(i, f"<{i}>") for i in ids]
                model_name = req.model or "model-tokenizer"
                return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))
        except Exception as e:
            logger.debug("Model tokenize failed, falling back to byte-level: %s", e)

    # Fallback: byte-level tokenization (always works, no dependencies)
    ids = list(req.text.encode("utf-8"))
    tokens = [f"b{b}" for b in ids]
    return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))


@router.post("/detokenize", response_model=DetokenizeResponse)
async def infer_detokenize(req: DetokenizeRequest) -> DetokenizeResponse:
    """Convert token IDs back to text."""
    model = _get_model_interface()
    if model is not None:
        try:
            tokenizer = getattr(model, '_tokenizer', None) or getattr(model, 'tokenizer', None)
            if tokenizer is not None and hasattr(tokenizer, 'decode'):
                text = tokenizer.decode(req.ids)
                return DetokenizeResponse(text=text, count=len(req.ids))
        except Exception:
            pass

    # Fallback: interpret IDs as byte values
    try:
        text = bytes(req.ids).decode("utf-8", errors="replace")
        return DetokenizeResponse(text=text, count=len(req.ids))
    except Exception:
        return DetokenizeResponse(text="", count=len(req.ids))


@router.get("/health", response_model=InferHealthResponse)
async def infer_health() -> InferHealthResponse:
    """Engine health — whether a model is loaded and what capabilities it has."""
    model = _get_model_interface()
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


@router.get("/info", response_model=InferInfoResponse)
async def infer_info() -> InferInfoResponse:
    """Metadata about the currently loaded model."""
    model = _get_model_interface()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

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


def _sse_token(stream: str, token: str, done: bool = False, meta: dict = None, elapsed_ms: float = None) -> str:
    phase = "STREAMING"
    status = "complete" if done else "working"
    m = dict(meta) if meta else {}
    if done and elapsed_ms is not None:
        m["elapsed_ms"] = round(elapsed_ms, 1)
    return _sse_event(stream, phase, status, {"token": token}, m, "")


def _sse_error(stream: str, phase: str, error: str, meta: dict = None) -> str:
    return _sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
