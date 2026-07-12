"""
Python core server — the ML engine.

Zero external dependencies for the server layer.
Uses our own minimal HTTP server (domains.server).

This server listens on 127.0.0.1:8000.

Usage:
    python apps/core-server/main.py
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path

# Ensure core-py is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core-py"))

from domains.server import App, Request, JSONResponse, StreamingResponse, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("core-server")

app = App(title="slough-core", version="0.1.0")

# ── Lazy-loaded state ───────────────────────────────────────────────────────

_model = None
_tokenizer = None
_model_name = None
_provider = None


def _get_provider():
    """Lazy-load the provider pipeline."""
    global _provider, _model, _tokenizer, _model_name
    if _provider is not None:
        return _provider

    from domains.models.provider import setup_providers, get_provider
    from domains.infrastructure.model_loader import get_model_loader

    model_name = os.environ.get("MAN_CORE_MODEL", "gpt2")
    logger.info("Loading model: %s", model_name)

    result = get_model_loader().load(model_name, device="cpu", verify=False)
    model = result.model
    tokenizer = result.tokenizer
    _model = model
    _tokenizer = tokenizer
    _model_name = model_name

    setup_providers(hf_model=model, hf_tokenizer=tokenizer, hf_model_id=model_name)
    _provider = get_provider("default")
    logger.info("Model loaded: %s", model_name)
    return _provider


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(req: Request):
    model_loaded = _model is not None
    return {"status": "ok", "model_loaded": model_loaded, "model": _model_name or "none", "core": True}


# ── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.8)

    provider = _get_provider()
    result = await provider.chat(messages, max_tokens=max_tokens, temperature=temperature)
    return {"message": result, "session_id": "core", "done": True}


@app.post("/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.8)

    provider = _get_provider()

    async def event_generator():
        try:
            async for token in provider.chat_stream(
                messages, max_tokens=max_tokens, temperature=temperature
            ):
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator())


# ── Inference ────────────────────────────────────────────────────────────────

@app.post("/inference/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    max_new_tokens = body.get("max_new_tokens", 100)
    temperature = body.get("temperature", 0.7)

    provider = _get_provider()
    result = await provider.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return {"text": result, "model": _model_name or "unknown", "tokens_generated": 0}


@app.post("/inference/generate/stream")
async def generate_stream(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    max_new_tokens = body.get("max_new_tokens", 100)
    temperature = body.get("temperature", 0.7)

    provider = _get_provider()

    async def event_generator():
        try:
            async for token in provider.chat_stream(
                [{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator())


# ── Models ───────────────────────────────────────────────────────────────────

@app.get("/models")
async def list_models(req: Request):
    models = []
    if _model is not None:
        models.append({
            "id": _model_name,
            "name": _model_name,
            "loaded": True,
            "source": "huggingface",
        })
    return {"models": models}


@app.get("/models/hf")
async def list_hf_models(req: Request):
    return {"models": [], "note": "Use /models for loaded models"}


@app.post("/models/load")
async def load_model(req: Request):
    body = await req.json()
    model_id = body.get("model_id", body.get("model", "gpt2"))

    global _model, _tokenizer, _model_name, _provider
    from domains.infrastructure.model_loader import get_model_loader
    from domains.models.provider import setup_providers, get_provider

    try:
        result = get_model_loader().load(model_id, device="cpu", verify=False)
        model = result.model
        tokenizer = result.tokenizer
        _model = model
        _tokenizer = tokenizer
        _model_name = model_id
        setup_providers(hf_model=model, hf_tokenizer=tokenizer, hf_model_id=model_id)
        _provider = get_provider("default")
        return {"status": "loaded", "model": model_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status=500)


@app.post("/models/unload")
async def unload_model(req: Request):
    global _model, _tokenizer, _model_name, _provider
    _model = None
    _tokenizer = None
    _model_name = None
    _provider = None
    return {"status": "unloaded"}


# ── Catch-all ────────────────────────────────────────────────────────────────

@app.route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(req: Request):
    return JSONResponse({"error": f"Not found: {req.method} {req.path}", "core": True}, status=404)


# ── Startup ──────────────────────────────────────────────────────────────────

@app.startup
async def startup():
    """Pre-load model on startup."""
    try:
        _get_provider()
    except Exception as e:
        logger.warning("Startup model load failed: %s (will retry on first request)", e)


# ── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("MAN_CORE_PORT", "8000"))
    logger.info("Starting core-server on port %d", port)
    run(app, host="127.0.0.1", port=port)
