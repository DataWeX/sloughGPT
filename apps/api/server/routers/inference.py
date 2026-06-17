"""
Inference Router - Chat and text generation endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncIterator, Any
from pathlib import Path
import json
import logging
import threading
logger = logging.getLogger(__name__)

try:
    from domains.api.sse_envelope import sse_event as _sse_event, sse_token, sse_error
except ImportError:
    import json as _json
    def _sse_event(stream, phase, status, data=None, meta=None, message=""):
        return "data: " + _json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message,
        }) + "\n\n"
    def sse_token(stream, token, done=False, meta=None, elapsed_ms=None):
        phase = "STREAMING"
        status = "complete" if done else "working"
        m = dict(meta) if meta else {}
        if done and elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
        return _sse_event(stream, phase, status, {"token": token}, m, "")
    def sse_error(stream, phase, error, meta=None):
        return _sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
import asyncio
import datetime
import uuid
import time
from threading import Thread

router = APIRouter(prefix="", tags=["inference"])

SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chat_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_session_cache: Optional[list] = None
_session_cache_ts: float = 0
_session_cache_ttl = 2.0  # seconds

# Lazy import to avoid circular deps
_context_core = None
_vector_store_ref = None


def _get_context_core():
    global _context_core, _vector_store_ref
    if _context_core is None:
        try:
            from domains.infrastructure.context_core import get_context_core
            _context_core = get_context_core()
        except ImportError:
            return None
    # If vector store is available, connect it
    if _context_core and _vector_store_ref and _context_core._vector_store is None:
        try:
            from domains.inference.vector_store import simple_embed
            _context_core.set_vector_store(_vector_store_ref, simple_embed)
        except Exception:
            pass
    return _context_core


def set_vector_store_ref(store):
    global _vector_store_ref
    _vector_store_ref = store


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt2"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge: Optional[List[str]] = None
    images: Optional[List[str]] = Field(default=None, description="Base64 encoded images")
    use_context_core: bool = Field(default=True, description="Use ContextCore for multi-layer context")


class ChatResponse(BaseModel):
    message: str
    session_id: str
    done: bool = True


class ContextInspectorResponse(BaseModel):
    system_prompt: str
    session_messages: List[dict]
    working_memory: List[dict]
    semantic_keys: List[str]
    episodic_count: int
    sensory_buffer_size: int
    last_frame: Optional[dict]


def _extract_user_message(messages: List[Message]) -> Optional[str]:
    """Extract the last user message from conversation."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content or None
    return None


def _enrich_knowledge(user_msg: str, auto_search: bool = True, max_facts: int = 5) -> dict:
    """Search learned knowledge + optionally live web search. Returns {facts, source, topics}."""
    try:
        from domains.learner.knowledge_augmenter import enrich_with_knowledge
        return enrich_with_knowledge(user_msg, auto_search=auto_search, max_facts=max_facts)
    except Exception as e:
        logger.warning(f"Knowledge enrichment failed: {e}")
        return {"facts": [], "source": "none", "topics": []}


def _load_session(session_id: str) -> dict:
    """Load session data from disk."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file) as f:
            return json.load(f)
    return {"id": session_id, "messages": [], "created_at": datetime.datetime.now().isoformat(), "updated_at": datetime.datetime.now().isoformat()}


def _save_session(session_id: str, data: dict) -> None:
    """Save session data to disk."""
    global _session_cache
    _session_cache = None  # invalidate cache
    data["updated_at"] = datetime.datetime.now().isoformat()
    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)


@router.post("/generate/demo")
async def generate_demo(prompt: str = "Hello", max_new_tokens: int = 100):
    """Demo endpoint - works without loading any model."""
    responses = [
        "I'm Aria, your self-learning AI companion. I'm running entirely on-device!",
        "That's interesting! I'm continuously learning from our conversation.",
        "I process everything locally - your data never leaves your device.",
        "My transformer model updates its weights in real-time. I'm getting smarter as we talk!",
    ]
    import random
    response = random.choice(responses)
    return {
        "text": response,
        "model": "demo",
        "prompt": prompt[:50],
    }


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.2, ge=0.5, le=2.0)
    model: str = "gpt2"


class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int = 0


@router.post("/inference/generate")
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Non-streaming generation — returns complete text response."""
    from domains.models.provider import get_provider
    from startup_progress import STARTUP_PHASE
    
    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model still loading (phase: {STARTUP_PHASE.get('phase', 'unknown')}). Please wait."
        )
    
    provider = get_provider("default")
    if provider is None:
        raise HTTPException(status_code=503, detail="No provider available")

    provider_messages = [{"role": "user", "content": req.prompt}]
    try:
        result = await provider.chat(
            provider_messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
        tokens = len(result.split())
        try:
            from domains.infrastructure.server_state import get_server_state
            get_server_state().record_inference(tokens=tokens, elapsed_ms=0, model=req.model)
        except Exception:
            pass
        return GenerateResponse(text=result, model=req.model, tokens_generated=tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Streaming generation — yields tokens as SSE."""
    from startup_progress import STARTUP_PHASE
    
    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready":
        async def error_stream() -> AsyncIterator[str]:
            yield sse_error(
                "generate",
                "IDLE",
                f"Model still loading (phase: {STARTUP_PHASE.get('phase', 'unknown')}). Please wait."
            )
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate() -> AsyncIterator[str]:
        from domains.models.provider import get_provider
        provider = get_provider("default")
        if provider is None:
            yield sse_error("generate", "IDLE", "No provider available")
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
                if token:
                    token_count += 1
                    yield sse_token("generate", token)
        except Exception as e:
            yield sse_error("generate", "STREAMING", str(e))
            return
        elapsed = (datetime.datetime.now() - start).total_seconds() * 1000
        try:
            from domains.infrastructure.server_state import get_server_state
            get_server_state().record_inference(tokens=token_count, elapsed_ms=elapsed, model=req.model)
        except Exception:
            pass
        yield sse_token("generate", "", done=True, meta={"tokens": token_count, "elapsed_ms": round(elapsed, 1)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/info")
async def get_info():
    """Get detailed server info."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from host_metrics import sample_host_metrics_async
    import state as server_state
    import torch

    torch_ver = torch.__version__ if server_state._torch_available else None
    cuda_avail = torch.cuda.is_available() if server_state._torch_available else False

    data = {
        "api_version": "1.0.0",
        "model": {
            "type": server_state.model_type,
            "loaded": server_state.model is not None,
        },
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
    }

    mrl = server_state.model_request_logger
    if mrl is not None:
        data["model"]["request_stats"] = mrl.get_stats()

    host = await sample_host_metrics_async()
    if host is not None:
        data["host"] = host

    cp = server_state.checkpoint
    if cp:
        data["model"].update({
            "vocab_size": len(cp.get("stoi", {})) if isinstance(cp, dict) else 0,
            "chars": len(cp.get("chars", [])) if isinstance(cp, dict) else 0,
        })

    se = server_state.soul_engine
    cs = server_state.current_soul
    if se is not None and getattr(se, 'is_loaded', False):
        data["soul_engine"] = se.get_stats()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_b = int(props.total_memory)
        used_b = int(torch.cuda.memory_allocated(0))
        data["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": total_b / 1e9,
            "memory_total_bytes": total_b,
            "memory_used_bytes": used_b,
            "memory_percent": round(100.0 * used_b / max(total_b, 1), 2),
        }

    return data


@router.get("/info/soul")
async def get_info_soul():
    """Get soul personality info for the server info endpoint."""
    import state as server_state
    cs = server_state.current_soul
    if not cs:
        return {}
    soul_info = {}
    try:
        soul_info["soul"] = {
            "name": cs.name if hasattr(cs, "name") else "",
            "description": cs.description if hasattr(cs, "description") else "",
            "integrity_hash": getattr(cs, "integrity_hash", ""),
            "born_at": getattr(cs, "born_at", ""),
            "tags": getattr(cs, "tags", []),
            "certifications": getattr(cs, "certifications", []),
        }
    except Exception:
        pass
    return soul_info


@router.get("/")
async def root():
    """Root endpoint - server status."""
    import state as server_state
    soul_name = None
    if server_state.soul_engine is not None and getattr(server_state.soul_engine, 'slo', None):
        soul_name = server_state.soul_engine.slo.name
    elif server_state.current_soul and hasattr(server_state.current_soul, "name"):
        soul_name = server_state.current_soul.name
    return {
        "name": "SloughGPT API",
        "version": "1.0.0",
        "status": "running",
        "model": server_state.model_type,
        "soul_loaded": soul_name,
        "soul_engine_active": server_state.soul_engine is not None and getattr(server_state.soul_engine, 'is_loaded', False),
        "endpoints": {
            "generate": "/generate (POST)",
            "v1_infer": "/v1/infer (POST) — SloughGPT Standard v1 envelope",
            "generate_stream": "/generate/stream (POST)",
            "generate_ws": "/ws/generate (WebSocket)",
            "load_soul": "/load-soul (POST) - loads into SloEngine",
            "soul": "/soul (GET)",
            "models": "/models (GET)",
            "datasets": "/datasets (GET)",
            "train_resolve": "/train/resolve (POST) — preview manifest → data_path",
            "info": "/info (GET)",
        },
    }


class LoadCheckpointRequest(BaseModel):
    model_path: str


@router.post("/load")
async def load_model_endpoint(request: Optional[LoadCheckpointRequest] = None):
    """Load the model on demand. Optionally specify model_path in request body."""
    import state as server_state
    model_path = request.model_path if request else None
    if server_state.model is not None and model_path is None:
        return {"status": "already_loaded", "model": server_state.model_type}
    from main import load_model
    load_model(model_path)
    return {"status": "loaded", "model": server_state.model_type, "checkpoint": model_path}


class LoadSoulRequest(BaseModel):
    soul_path: str


@router.post("/load-soul")
async def load_soul(request: LoadSoulRequest):
    """Load a .soul Soul Unit file into SloEngine."""
    try:
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages" / "core-py"))
        from domains.core import SloEngine

        engine = SloEngine(device="cpu")
        soul = engine.load_soul(request.soul_path)

        import state as server_state
        server_state.soul_engine = engine
        server_state.current_soul = soul
        server_state.model = engine.model
        server_state.model_type = f"sou/{soul.name}" if hasattr(soul, "name") else "sou/loaded"

        return {
            "status": "loaded",
            "soul_name": soul.name if hasattr(soul, "name") else "unknown",
            "lineage": soul.lineage if hasattr(soul, "lineage") else "unknown",
            "born_at": soul.born_at if hasattr(soul, "born_at") else "",
            "generation_params": {
                "temperature": soul.generation.temperature if soul.generation else 0.8,
                "top_p": soul.generation.top_p if soul.generation else 0.9,
                "max_tokens": soul.generation.max_tokens if soul.generation else 2048,
            },
            "personality": soul.personality.to_dict()
            if hasattr(soul, "personality") and soul.personality
            else {},
            "cognition": soul.cognition.to_dict()
            if hasattr(soul, "cognition") and soul.cognition
            else {},
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream chat responses with multi-layer context + live knowledge enrichment."""
    from startup_progress import STARTUP_PHASE
    
    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready":
        async def error_stream() -> AsyncIterator[str]:
            yield sse_error(
                "chat",
                "IDLE",
                f"Model still loading (phase: {STARTUP_PHASE.get('phase', 'unknown')}). Please wait."
            )
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate() -> AsyncIterator[str]:
        cancel_event = threading.Event()
        user_msg = _extract_user_message(req.messages)
        if not user_msg:
            yield sse_error("chat", "IDLE", "No user message")
            return
        
        start_time = datetime.datetime.now()
        
        # ── Progressive: emit "thinking" immediately, start enrichment in parallel ──
        yield _sse_event("chat", "STREAMING", "thinking",
            data={}, message="Thinking...")

        # Store injected knowledge (from KnowledgePanel) in vector store so
        # vector search finds it. Then query vector store for relevant facts.
        if req.knowledge:
            try:
                from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
                import time
                mem = get_knowledge_memory()
                stored = 0
                for k in req.knowledge:
                    if k and len(k) > 10:
                        fact = KnowledgeFact(content=k, topic="injected", source="injected",
                                             timestamp=time.time(), importance=0.7)
                        if mem.add_fact(fact):
                            stored += 1
                if stored:
                    logger.info(f"Stored {stored} injected knowledge items in vector store")
            except Exception as e:
                logger.warning(f"Failed to store injected knowledge: {e}")
        know_result = await asyncio.to_thread(
            _enrich_knowledge, user_msg, False, 5
        )
        provider_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if req.images:
            content_parts = [{"type": "text", "text": user_msg}]
            for img_data in req.images:
                content_parts.append({"type": "image_url", "image_url": {"url": img_data}})
            for i in range(len(provider_messages) - 1, -1, -1):
                if provider_messages[i]["role"] == "user":
                    provider_messages[i]["content"] = content_parts
                    break
        
        session_id = req.session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        session_data = _load_session(session_id)
        session_data.setdefault("messages", []).append({
            "role": "user",
            "content": user_msg,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        
        ctx_core = _get_context_core()
        context_info = {}
        frame = None
        if ctx_core and req.use_context_core:
            ctx_core.set_session_id(session_id)
            ctx_core.add_message("user", user_msg)
            frame = ctx_core.build_context_frame(
                include_rag=True,
                include_memory=True,
                query=user_msg,
            )
            context_info = {
                "layers": [l.layer_type for l in frame.layers],
                "total_tokens": frame.total_tokens,
                "max_tokens": frame.max_tokens,
            }
            if frame.system_prompt:
                for i, m in enumerate(provider_messages):
                    if m["role"] == "system":
                        provider_messages[i] = {"role": "system", "content": frame.system_prompt}
                        break
                else:
                    provider_messages.insert(0, {"role": "system", "content": frame.system_prompt})
        
        # ── Emit single "ready" event with enrichment + context info ──
        ready_data: dict[str, Any] = {}
        if know_result["source"] != "none":
            req.knowledge = know_result["facts"]
            logger.info(f"Knowledge: {len(know_result['facts'])} facts from {know_result['source']}")
            ready_data["source"] = know_result["source"]
            ready_data["topics"] = know_result["topics"]
            ready_data["fact_count"] = len(know_result["facts"])
        if context_info:
            ready_data["context"] = context_info
        if ready_data:
            yield _sse_event("chat", "STREAMING", "working",
                data=ready_data,
                message=f"Found {ready_data.get('fact_count', 0)} facts, {len(context_info.get('layers', []))} context layers" if ready_data else "")
        
        try:
            from domains.models.provider import get_provider
            # Use VLM provider when images are present and VLM is loaded
            if req.images:
                vlm_provider = get_provider("vlm")
                if vlm_provider is not None:
                    provider = vlm_provider
                else:
                    provider = get_provider("default")
            else:
                provider = get_provider("default")
            
            if provider is not None:
                if req.knowledge:
                    knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
                    provider_messages.insert(0, {
                        "role": "system",
                        "content": f"[KNOWLEDGE]\n{knowledge_str}\n[/KNOWLEDGE]"
                    })

                full_response = ""
                try:
                    try:
                        async for token in provider.chat_stream(
                            provider_messages,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            cancel_event=cancel_event,
                            repetition_penalty=1.3,
                            stop_sequences=["\nUser:", "\n\nAssistant:"],
                        ):
                            if token:
                                full_response += token
                                yield sse_token("chat", token)
                        yield sse_token("chat", "", done=True)
                    except GeneratorExit:
                        cancel_event.set()
                        logger.info("Client disconnected from chat stream")
                        return
                except Exception as e:
                    logger.error("Provider chat_stream error: %s", e, exc_info=True)
                    yield sse_error("chat", "ERROR", f"Generation failed: {e}")
                    return
            else:
                # Fallback: direct HF model (no provider pipeline)
                from controllers.models import get_models_controller
                from transformers import TextIteratorStreamer

                ctrl = get_models_controller()
                if not ctrl._hf_model or not ctrl._tokenizer:
                    yield sse_error("chat", "STREAMING", "No model loaded")
                    return

                system_prompt = frame.system_prompt if frame and frame.system_prompt else (req.system_prompt or "")
                if req.knowledge:
                    knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
                    system_prompt = f"{system_prompt}\n\n[KNOWLEDGE]\n{knowledge_str}\n[/KNOWLEDGE]"
                full_prompt = f"{system_prompt}\n{user_msg}" if system_prompt else user_msg

                inputs = ctrl._tokenizer(full_prompt, return_tensors="pt")
                input_ids_tensor = inputs["input_ids"].to(ctrl._hf_model.device)

                streamer = TextIteratorStreamer(
                    ctrl._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                def run_generation():
                    try:
                        with torch.no_grad():
                            ctrl._hf_model.generate(
                                input_ids=input_ids_tensor,
                                max_new_tokens=req.max_tokens,
                                temperature=req.temperature,
                                do_sample=req.temperature > 0,
                                pad_token_id=ctrl._tokenizer.eos_token_id,
                                streamer=streamer,
                            )
                    except Exception as e:
                        logger.error("HF model.generate error: %s", e, exc_info=True)
                        raise

                thread = Thread(target=run_generation)
                thread.start()

                full_response = ""
                try:
                    try:
                        for text in streamer:
                            if text:
                                full_response += text
                                yield sse_token("chat", text)

                        thread.join(timeout=120)
                        if thread.is_alive():
                            logger.warning("Generation thread timed out after 120s")
                            yield sse_error("chat", "ERROR", "Generation timed out")
                            return
                    except GeneratorExit:
                        cancel_event.set()
                        logger.info("Client disconnected from chat stream (fallback)")
                        return
                except Exception as e:
                    logger.error("Streaming error: %s", e, exc_info=True)
                    yield sse_error("chat", "ERROR", f"Streaming failed: {e}")
                    return

                yield sse_token("chat", "", done=True)
            
            # Log response for benchmarking
            try:
                from domains.feedback.response_tracker import get_response_tracker
                tracker = get_response_tracker()
                duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
                tracker.log(
                    user_message=user_msg or "",
                    assistant_response=full_response,
                    model=req.model,
                    config={"temperature": req.temperature, "max_tokens": req.max_tokens},
                    session_id=session_id,
                    user_id=req.user_id or "default",
                    tokens_generated=len(full_response.split()),
                    duration_ms=duration_ms,
                    has_images=bool(req.images),
                )
            except Exception:
                pass

            # Record inference metrics
            try:
                from domains.infrastructure.server_state import get_server_state
                tokens = len(full_response.split())
                elapsed_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
                get_server_state().record_inference(tokens=tokens, elapsed_ms=elapsed_ms, model=req.model)
            except Exception:
                pass
            
            # Save response
            session_data["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.datetime.now().isoformat(),
            })
            _save_session(session_id, session_data)
            
            # Update ContextCore with response
            if ctx_core and req.use_context_core:
                ctx_core.add_response(full_response, model=req.model)

            # Feed conversation pair to continual learner (fire-and-forget)
            try:
                from domains.learner import get_learner
                get_learner().ingest_conversation([(user_msg, full_response)])
            except Exception:
                pass

            # Auto-extract entities and relationships into knowledge base
            try:
                from domains.learner.entity_extractor import extract_and_store
                extract_and_store(user_msg or "", full_response)
            except Exception:
                pass

            logger.info(f"Chat stream: generated {len(full_response)} chars")

        except Exception as e:
            yield sse_error("chat", "STREAMING", str(e))
            yield sse_token("chat", "", done=True)
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/context/inspect")
async def inspect_context() -> dict:
    """Get current ContextCore state for inspection."""
    ctx_core = _get_context_core()
    if not ctx_core:
        return {"error": "ContextCore not available"}
    return ctx_core.get_context_inspector()


@router.post("/context/fact")
async def store_fact(key: str, value: str) -> dict:
    """Store a fact in semantic memory."""
    ctx_core = _get_context_core()
    if not ctx_core:
        return {"error": "ContextCore not available"}
    ctx_core.store_fact(key, value)
    return {"stored": key}


@router.get("/context/facts")
async def get_facts(query: str = "") -> dict:
    """Search semantic memory."""
    ctx_core = _get_context_core()
    if not ctx_core:
        return {"error": "ContextCore not available", "facts": []}
    if query:
        return {"facts": ctx_core.search_semantic(query)}
    return {"facts": [{"key": k, **v} for k, v in ctx_core.semantic_memory.items()]}


@router.post("/context/reset")
async def reset_context(all: bool = False) -> dict:
    """Reset ContextCore."""
    global _context_core
    ctx_core = _get_context_core()
    if not ctx_core:
        return {"error": "ContextCore not available"}
    if all:
        ctx_core.reset_all()
    else:
        ctx_core.reset_session()
    return {"reset": "session" if not all else "all"}


@router.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Non-streaming chat using ChatDomain."""
    from domains import get_chat_domain
    from startup_progress import STARTUP_PHASE
    
    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model still loading (phase: {STARTUP_PHASE.get('phase', 'unknown')}). Please wait."
        )
    
    user_msg = _extract_user_message(req.messages)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message")
    
    system_prompt = req.system_prompt or ""
    
    # Build messages list for domain
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    # Use ChatDomain for generation + logging
    chat_domain = get_chat_domain()
    
    # Inject knowledge into system prompt if provided
    if req.knowledge:
        knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
        system_prompt = f"{system_prompt}\n\n[KNOWLEDGE]\n{knowledge_str}\n[/KNOWLEDGE]" if system_prompt else f"[KNOWLEDGE]\n{knowledge_str}\n[/KNOWLEDGE]"
    
    result = await chat_domain.respond(
        messages=messages,
        model=req.model,
        system_prompt=system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        session_id=req.session_id or "default",
        user_id=req.user_id or "default",
    )

    # Record inference metrics
    try:
        from domains.infrastructure.server_state import get_server_state
        tokens = len(result.text.split())
        get_server_state().record_inference(tokens=tokens, elapsed_ms=0, model=req.model)
    except Exception:
        pass

    return ChatResponse(
        message=result.text,
        session_id=result.session_id,
        done=result.done,
    )


def _build_session_cache() -> list:
    global _session_cache, _session_cache_ts
    now = time.time()
    if _session_cache is not None and now - _session_cache_ts < _session_cache_ttl:
        return _session_cache
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            sessions.append(data)
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    _session_cache = sessions
    _session_cache_ts = now
    return sessions


@router.get("/chat/sessions")
async def list_sessions():
    return {"sessions": _build_session_cache()}


@router.get("/chat/sessions/current")
async def get_current_session():
    """Return the most recently updated session, or null."""
    sessions = _build_session_cache()
    if not sessions:
        return {"session": None}
    return {"session": sessions[0]}


@router.put("/chat/sessions/{session_id}")
async def upsert_session(session_id: str, req: dict):
    """Create or update a session."""
    _save_session(session_id, req)
    return {"status": "saved", "session_id": session_id}


@router.post("/chat/sessions")
async def create_session(req: dict):
    """Create a new session."""
    session_id = req.get("session_id") or str(uuid.uuid4())
    _save_session(session_id, req)
    return {"status": "created", "session_id": session_id}


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    data = _load_session(session_id)
    if not data.get("messages"):
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/providers")
async def list_model_providers():
    """List all registered model providers with capabilities."""
    from domains.models.provider import list_providers, get_provider

    result = {}
    for name in list_providers():
        provider = get_provider(name)
        if provider is not None:
            try:
                caps = provider.capabilities
                result[name] = {
                    "model_id": provider.model_id,
                    "capabilities": {
                        "chat": caps.chat,
                        "streaming": caps.streaming,
                        "embedding": caps.embedding,
                        "vision": caps.vision,
                    },
                    "metadata": provider.metadata,
                }
            except Exception:
                result[name] = {"model_id": str(provider)}
        else:
            result[name] = {"error": "provider not found"}
    return {"providers": result}
