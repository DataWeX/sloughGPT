"""
Inference Router - Chat and text generation endpoints
"""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncIterator
from pathlib import Path
import json
import logging
import threading
try:
    import torch
except ImportError:
    torch = None

from schemas.common import success_response

logger = logging.getLogger("slo.inference")

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

_SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chat_sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_VOICE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "voice_messages"
_VOICE_DIR.mkdir(parents=True, exist_ok=True)

# FileRepository-backed session store
from domains.infrastructure.repository import FileRepository, Serializer


class _SessionDictSerializer(Serializer[dict]):
    """JSON serializer for session dict data."""

    def serialize(self, obj: dict) -> dict:
        return obj

    def deserialize(self, data: dict) -> dict:
        return data


_session_repo = FileRepository[dict](
    directory=str(_SESSIONS_DIR),
    serializer=_SessionDictSerializer(),
    key_suffix=".json",
)
_session_repo.enable_cache(ttl_seconds=2.0)

_session_cache: Optional[list] = None
_session_cache_ts: float = 0
_session_cache_ttl = 2.0  # seconds

# In-memory session cache for hot path — avoids sync disk I/O on every chat message
_session_memory_cache: dict[str, dict] = {}
_session_dirty: set[str] = set()

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
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.2, ge=0.5, le=2.0)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge: Optional[List[str]] = None
    images: Optional[List[str]] = Field(default=None, description="Base64 encoded images")
    use_context_core: bool = Field(default=True, description="Use ContextCore for multi-layer context")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for role-based system instructions")


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
        logger.warning("Knowledge enrichment failed: %s", e, extra={"tag": "INF", "context": {"error": str(e)}})
        return {"facts": [], "source": "none", "topics": []}


def _load_session_from_disk(session_id: str) -> dict:
    """Load session data from disk via FileRepository (cold path)."""
    data = _session_repo.get(session_id)
    if data is not None:
        return data
    return {"id": session_id, "messages": [], "created_at": datetime.datetime.now().isoformat(), "updated_at": datetime.datetime.now().isoformat()}


def _get_session(session_id: str) -> dict:
    """Get session data from memory cache or disk (hot path uses cache)."""
    _start_background_flush()
    if session_id in _session_memory_cache:
        return _session_memory_cache[session_id]
    data = _load_session_from_disk(session_id)
    _session_memory_cache[session_id] = data
    return data


def _save_session(session_id: str, data: dict) -> None:
    """Save session — updates memory cache, queues async disk write."""
    global _session_cache
    _session_cache = None  # invalidate session list cache
    data["updated_at"] = datetime.datetime.now().isoformat()
    _session_memory_cache[session_id] = data
    _session_dirty.add(session_id)


async def _flush_session_to_disk(session_id: str) -> None:
    """Write a single dirty session to disk via FileRepository.

    Catches disk errors to prevent them from crashing request handlers.
    """
    data = _session_memory_cache.get(session_id)
    if data is None:
        _session_dirty.discard(session_id)
        return
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _session_repo.save, session_id, data)
    except Exception as exc:
        logger.warning("Disk write failed for session %s: %s", session_id, exc, extra={"tag": "REQ"})
    finally:
        _session_dirty.discard(session_id)


async def flush_dirty_sessions() -> int:
    """Flush all dirty sessions to disk. Returns count flushed."""
    dirty = list(_session_dirty)
    if not dirty:
        return 0
    for sid in dirty:
        await _flush_session_to_disk(sid)
    return len(dirty)


_background_flush_task: Optional[asyncio.Task] = None


def _start_background_flush() -> None:
    """Start periodic flush of dirty sessions every 10s."""
    global _background_flush_task
    if _background_flush_task is not None and not _background_flush_task.done():
        return
    async def _flush_loop():
        while True:
            await asyncio.sleep(10)
            try:
                await flush_dirty_sessions()
            except Exception:
                pass
    try:
        _background_flush_task = asyncio.create_task(_flush_loop())
    except RuntimeError:
        pass  # No running event loop


@router.post("/generate/demo")
async def generate_demo(prompt: str = "Hello", max_new_tokens: int = 100):
    """Demo endpoint — returns hardcoded mock responses (no model required).

    WARNING: This endpoint does NOT use any loaded model. Responses are
    randomly selected from a fixed list. Use POST /chat or POST /inference/generate
    for real model inference.
    """
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
        "warning": "This is a demo response — no model was used. Use POST /chat for real inference.",
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
    import state as _gen_state

    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready" or _gen_state.model is None:
        raise HTTPException(status_code=503, detail="Model still loading — please wait.")

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
async def generate_stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Streaming generation — yields tokens as SSE."""
    from startup_progress import STARTUP_PHASE
    import state as _stream_state

    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready" or _stream_state.model is None:
        async def error_stream() -> AsyncIterator[str]:
            yield sse_error("generate", "IDLE", "Model still loading — please wait.")
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
                if await request.is_disconnected():
                    logger.info("Client disconnected from generate stream", extra={"tag": "INF"})
                    return
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


@router.get("/chat/tools")
async def list_chat_tools():
    """List all available tools that can be invoked during chat."""
    try:
        from domains.agents.tools import get_tool_registry
        return {"tools": get_tool_registry().list_tools()}
    except Exception as e:
        logger.warning("Failed to list tools: %s", e, extra={"tag": "INF"})
        return {"tools": []}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream chat responses with multi-layer context + live knowledge enrichment."""
    from startup_progress import STARTUP_PHASE

    # Check if model is ready before processing
    import state as _check_state
    if STARTUP_PHASE.get("phase") != "ready" or _check_state.model is None:
        phase = STARTUP_PHASE.get("phase", "unknown")
        if phase == "ready":
            msg = "Model still loading — please wait."
        else:
            msg = f"Server starting (phase: {phase}). Please wait."
        async def error_stream() -> AsyncIterator[str]:
            yield sse_error("chat", "IDLE", msg)
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate() -> AsyncIterator[str]:
        logger.debug("chat_stream.generate() ENTERED")
        cancel_event = threading.Event()
        user_msg = _extract_user_message(req.messages)
        if not user_msg:
            yield sse_error("chat", "IDLE", "No user message")
            return

        start_time = datetime.datetime.now()

        # ── Progressive: emit "thinking" immediately, start enrichment in parallel ──
        logger.debug("chat_stream: yielding thinking event")
        yield _sse_event("chat", "STREAMING", "thinking",
            data={}, message="Thinking...")

        # Store injected knowledge (from KnowledgePanel) in vector store so
        # vector search finds it. Then query vector store for relevant facts.
        if req.knowledge:
            try:
                def _store_knowledge(k_list):
                    from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
                    import time
                    mem = get_knowledge_memory()
                    stored = 0
                    for k in k_list:
                        if k and len(k) > 10:
                            fact = KnowledgeFact(content=k, topic="injected", source="injected",
                                                 timestamp=time.time(), importance=0.7)
                            if mem.add_fact(fact):
                                stored += 1
                    return stored
                stored = await asyncio.to_thread(_store_knowledge, req.knowledge)
                if stored:
                    logger.info("Stored %d injected knowledge items in vector store", stored, extra={"tag": "INF", "context": {"count": stored}})
            except Exception as e:
                logger.warning("Failed to store injected knowledge: %s", e, extra={"tag": "INF", "context": {"error": str(e)}})
        # Build provider messages from request
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

        session_data = _get_session(session_id)
        session_data.setdefault("messages", []).append({
            "role": "user",
            "content": user_msg,
            "timestamp": datetime.datetime.now().isoformat(),
        })

        ctx_core = _get_context_core()
        context_info = {}
        frame = None
        skip_context = False
        if ctx_core and req.use_context_core:
            try:
                from domains.learner.knowledge import get_knowledge_memory
                kmem = get_knowledge_memory()
                if kmem.stats().get("total_items", 0) == 0 and not req.knowledge:
                    skip_context = True
            except Exception:
                pass
        if ctx_core and req.use_context_core and not skip_context:
            ctx_core.set_session_id(session_id)
            ctx_core.add_message("user", user_msg)
            frame = await ctx_core.build_context_frame(
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

        # ── Inject agent instructions into system prompt ──
        if req.agent_id:
            try:
                from domains.agents.system import get_agent_system
                agent_sys = get_agent_system()
                agent_instructions = agent_sys.get_instructions(req.agent_id)
                if agent_instructions:
                    agent_msg = {"role": "system", "content": f"[AGENT: {req.agent_id}]\n{agent_instructions}"}
                    # Replace existing agent system message or insert before knowledge
                    replaced = False
                    for i, m in enumerate(provider_messages):
                        if m["role"] == "system" and m["content"].startswith("[AGENT:"):
                            provider_messages[i] = agent_msg
                            replaced = True
                            break
                    if not replaced:
                        provider_messages.insert(0, agent_msg)
            except Exception:
                logger.warning("Failed to inject agent instructions", exc_info=True, extra={"tag": "INF"})

        # ── Tool intent detection & execution ──
        tool_result_data = None
        try:
            from domains.agents.tools import get_tool_registry
            tool_reg = get_tool_registry()
            tool_intent = tool_reg.detect_tool_intent(user_msg)
            if tool_intent:
                tool_name, tool_args = tool_intent
                spec = tool_reg.get(tool_name)
                if spec and spec.requires_approval:
                    # Yield approval request — frontend will show dialog and POST /chat/execute-tool
                    yield _sse_event("chat", "TOOL_APPROVAL", "pending",
                        data={"tool": tool_name, "args": tool_args, "requires_approval": True},
                        message=f"Approval needed: {tool_name}")
                else:
                    yield _sse_event("chat", "TOOL", "working",
                        data={"tool": tool_name, "args": tool_args, "status": "executing"},
                        message=f"Running tool: {tool_name}")
                    result = await tool_reg.execute(tool_name, tool_args)
                    tool_result_data = {
                        "tool": tool_name,
                        "status": "success" if result.success else "error",
                        "output": result.output,
                        "error": result.error,
                        "duration_ms": round(result.duration_ms, 1),
                    }
                    if result.success:
                        yield _sse_event("chat", "TOOL", "complete",
                            data=tool_result_data,
                            message=f"Tool {tool_name} completed in {result.duration_ms:.0f}ms")
                        provider_messages.append({
                            "role": "system",
                            "content": f"[TOOL RESULT: {tool_name}]\n{result.output}\n[/TOOL RESULT]"
                        })
                    else:
                        yield _sse_event("chat", "TOOL", "error",
                            data=tool_result_data,
                            message=f"Tool {tool_name} failed: {result.error}")
                        provider_messages.append({
                            "role": "system",
                            "content": f"[TOOL RESULT: {tool_name}]\nError: {result.error}\n[/TOOL RESULT]"
                        })
        except Exception:
            logger.warning("Tool execution failed", exc_info=True, extra={"tag": "INF"})

        # ── Emit context info if available ──
        if context_info:
            yield _sse_event("chat", "STREAMING", "working",
                data={"context": context_info},
                message=f"{len(context_info.get('layers', []))} context layers")

        # ── Knowledge enrichment from KnowledgeMemory (offload to thread to avoid event loop deadlock) ──
        knowledge_retrieved = []
        try:
            enrichment = await asyncio.to_thread(_enrich_knowledge, user_msg, False, 5)
            if enrichment.get("facts"):
                knowledge_retrieved = enrichment["facts"]
                k_text = "\n".join(f"- {f}" for f in enrichment["facts"])
                provider_messages.insert(0, {
                    "role": "system",
                    "content": f"Use the following context to answer the question:\n{k_text}"
                })
        except Exception:
            pass

        try:
            from domains.models.provider import get_provider
            provider = get_provider("default")

            if provider is not None:
                if req.knowledge:
                    knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
                    provider_messages.insert(0, {
                        "role": "system",
                        "content": f"Use the following context to answer:\n{knowledge_str}"
                    })

                full_response = ""
                logger.debug("chat_stream: about to call provider.chat_stream()")
                try:
                    try:
                        async for token in provider.chat_stream(
                            provider_messages,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            top_k=req.top_k,
                            repetition_penalty=req.repetition_penalty,
                            cancel_event=cancel_event,
                            session_id=session_id,
                        ):
                            if await request.is_disconnected():
                                cancel_event.set()
                                logger.info("Client disconnected from chat stream (request)", extra={"tag": "INF", "context": {"session_id": session_id}})
                                return
                            if token:
                                full_response += token
                                yield sse_token("chat", token)
                        yield sse_token("chat", "", done=True)
                    except GeneratorExit:
                        cancel_event.set()
                        logger.info("Client disconnected from chat stream", extra={"tag": "INF", "context": {"session_id": session_id}})
                        return
                except Exception as e:
                    logger.error("Provider chat_stream error: %s", e, exc_info=True, extra={"tag": "INF", "context": {"session_id": session_id, "error": str(e)}})
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
                    system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{knowledge_str}"
                if knowledge_retrieved:
                    k_text = "\n".join(f"- {f}" for f in knowledge_retrieved)
                    system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{k_text}"
                full_prompt = f"{system_prompt}\n{user_msg}" if system_prompt else user_msg

                inputs = ctrl._tokenizer(full_prompt, return_tensors="pt")
                input_ids_tensor = inputs["input_ids"].to(ctrl._hf_model.device)

                streamer = TextIteratorStreamer(
                    ctrl._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                _thread_error = [None]

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
                        _thread_error[0] = e
                        logger.error("HF model.generate error: %s", e, exc_info=True, extra={"tag": "INF", "context": {"session_id": session_id, "error": str(e)}})

                thread = Thread(target=run_generation)
                thread.start()

                full_response = ""
                try:
                    try:
                        while thread.is_alive() or not streamer.text_queue.empty():
                            if _thread_error[0] is not None:
                                yield sse_error("chat", "ERROR", f"Generation failed: {_thread_error[0]}")
                                return
                            if await request.is_disconnected():
                                cancel_event.set()
                                logger.info("Client disconnected from chat stream (fallback, request)", extra={"tag": "INF", "context": {"session_id": session_id}})
                                return
                            try:
                                text = streamer.text_queue.get(timeout=0.01)
                            except Exception:
                                await asyncio.sleep(0)
                                continue
                            if text == streamer.stop_signal:
                                break
                            if text:
                                full_response += text
                                yield sse_token("chat", text)

                        thread.join(timeout=120)
                        if thread.is_alive():
                            logger.warning("Generation thread timed out after 120s", extra={"tag": "INF", "context": {"session_id": session_id, "timeout_s": 120}})
                            yield sse_error("chat", "ERROR", "Generation timed out")
                            return
                    except GeneratorExit:
                        cancel_event.set()
                        logger.info("Client disconnected from chat stream (fallback)", extra={"tag": "INF", "context": {"session_id": session_id}})
                        return
                except Exception as e:
                    logger.error("Streaming error: %s", e, exc_info=True, extra={"tag": "INF", "context": {"session_id": session_id, "error": str(e)}})
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

            # Save response (memory cache + async disk flush)
            session_data["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.datetime.now().isoformat(),
            })
            _save_session(session_id, session_data)
            await _flush_session_to_disk(session_id)

            # Update ContextCore with response
            if ctx_core and req.use_context_core:
                ctx_core.add_response(full_response, model=req.model)

            # Feed conversation pair to continual learner (fire-and-forget)
            try:
                from domains.learner import get_learner
                get_learner().ingest_conversation([(user_msg, full_response)])
            except Exception as e:
                logger.debug("Continual learner ingest failed: %s", e)

            # Auto-extract entities and relationships into knowledge base
            try:
                from domains.learner.entity_extractor import extract_and_store
                asyncio.create_task(extract_and_store(user_msg or "", full_response))
            except Exception as e:
                logger.debug("Entity extraction failed: %s", e)

            logger.info("Chat stream: generated %d chars", len(full_response), extra={"tag": "INF", "context": {"char_count": len(full_response), "session_id": session_id}})

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
    import state as _chat_state

    # Check if model is ready before processing
    if STARTUP_PHASE.get("phase") != "ready" or _chat_state.model is None:
        raise HTTPException(status_code=503, detail="Model still loading — please wait.")

    # Check if circuit breaker is open (model failing repeatedly)
    try:
        from domains.models.provider import get_provider
        _router = get_provider("default")
        _server = getattr(_router, '_server', None)
        if _server is not None:
            _cb = getattr(_server, '_circuit_breaker', None)
            if _cb is not None and _cb.state.value == "open":
                raise HTTPException(status_code=503, detail="Model is degraded — circuit breaker open. Please wait or reload the model.")
    except HTTPException:
        raise
    except Exception:
        pass

    user_msg = _extract_user_message(req.messages)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message")

    system_prompt = req.system_prompt or ""

    # Build messages list for domain
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Use ChatDomain for generation + logging
    chat_domain = get_chat_domain()

    # Inject agent instructions into system prompt if provided
    if req.agent_id:
        try:
            from domains.agents.system import get_agent_system
            agent_sys = get_agent_system()
            agent_instructions = agent_sys.get_instructions(req.agent_id)
            if agent_instructions:
                system_prompt = f"{system_prompt}\n\n[AGENT: {req.agent_id}]\n{agent_instructions}" if system_prompt else f"[AGENT: {req.agent_id}]\n{agent_instructions}"
        except Exception:
            logger.warning("Failed to inject agent instructions", exc_info=True, extra={"tag": "INF"})

    # Inject knowledge into system prompt if provided
    if req.knowledge:
        knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
        system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{knowledge_str}" if system_prompt else f"Use the following context to answer:\n{knowledge_str}"

    # Knowledge enrichment from KnowledgeMemory
    try:
        enrichment = await asyncio.to_thread(_enrich_knowledge, user_msg, False, 5)
        if enrichment.get("facts"):
            k_text = "\n".join(f"- {f}" for f in enrichment["facts"])
            system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{k_text}" if system_prompt else f"Use the following context to answer:\n{k_text}"
    except Exception:
        pass

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
    for sid in _session_repo.keys():
        data = _session_repo.get(sid)
        if data is None:
            continue
        # Normalize: ensure every session has id, name, updated_at
        data["id"] = sid
        data.pop("session_id", None)
        if not data.get("name"):
            msgs = data.get("messages", [])
            if msgs:
                first = msgs[0].get("content", "").split("\n")[0]
                data["name"] = first[:60]
            else:
                data["name"] = sid
        if not data.get("updated_at"):
            data["updated_at"] = data.get("created_at") or datetime.datetime.now().isoformat()
        sessions.append(data)
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    _session_cache = sessions
    _session_cache_ts = now
    return sessions


@router.post("/chat/voice/{session_id}")
async def send_voice_message(
    session_id: str,
    file: UploadFile = File(...),
    duration_ms: int = Form(0),
):
    """Upload a voice message to a session. Stores audio and creates a message entry."""
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files accepted")

    msg_id = str(uuid.uuid4())
    session_msg_dir = _VOICE_DIR / session_id
    session_msg_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio.m4a").suffix or ".m4a"
    audio_path = session_msg_dir / f"{msg_id}{ext}"

    content = await file.read()
    audio_path.write_bytes(content)

    session_data = _get_session(session_id)
    session_data.setdefault("messages", []).append({
        "role": "user",
        "content": "[Voice Message]",
        "audio_path": f"{session_id}/{msg_id}{ext}",
        "audio_duration_ms": duration_ms,
        "timestamp": datetime.datetime.now().isoformat(),
        "_voice": True,
    })
    _save_session(session_id, session_data)

    return success_response(data={
        "message_id": msg_id,
        "audio_path": f"{session_id}/{msg_id}{ext}",
        "session_id": session_id,
    })


@router.get("/chat/audio/{session_id}/{message_id}")
async def get_voice_audio(session_id: str, message_id: str):
    """Serve a stored voice message audio file."""
    audio_path = _VOICE_DIR / session_id / message_id
    if not audio_path.exists():
        # Try with common extensions
        for ext in [".m4a", ".wav", ".mp3", ".ogg", ".webm"]:
            candidate = audio_path.parent / f"{audio_path.stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        else:
            raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(audio_path), media_type="audio/m4a")


@router.get("/chat/sessions")
async def list_sessions(archived: Optional[bool] = None):
    sessions = _build_session_cache()
    if archived is not None:
        sessions = [s for s in sessions if s.get("archived", False) == archived]
    return success_response(data=sessions)


@router.get("/chat/sessions/search")
async def search_sessions(q: str = "", limit: int = 20):
    """Full-text search across all conversation files.

    Searches session names and message content. Returns session
    summaries with matching message excerpts.
    """
    if not q.strip():
        return success_response(data=[], meta={"query": q, "total": 0})

    q_lower = q.lower().strip()
    results = []

    # Search both possible session directories
    search_dirs: list[Path] = []
    sessions_dir = Path(__file__).parent.parent.parent.parent / "data" / "chat_sessions"
    if sessions_dir.is_dir():
        search_dirs.append(sessions_dir)
    conv_dir = Path(__file__).parent.parent.parent.parent / "data" / "conversations"
    if conv_dir.is_dir():
        search_dirs.append(conv_dir)
    seen = set()

    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for f in sorted(sdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if len(results) >= limit:
                break
            try:
                data = json.loads(f.read_text())
                sid = data.get("id") or data.get("session_id") or f.stem
                if sid in seen:
                    continue
                seen.add(sid)
                name = data.get("name", "") or ""
                messages = data.get("messages", [])
                matches = []

                if q_lower in name.lower():
                    matches.append({"role": "session", "content": name, "timestamp": data.get("updated_at", "")})

                for msg in messages:
                    content = msg.get("content", "")
                    if q_lower in content.lower():
                        matches.append({
                            "role": msg.get("role", "unknown"),
                            "content": content,
                            "timestamp": msg.get("timestamp", ""),
                        })

                if matches:
                    results.append({
                        "id": sid,
                        "name": name or sid,
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "match_count": len(matches),
                        "matches": matches[:3],  # Top 3 matches per session
                    })
            except (json.JSONDecodeError, OSError):
                continue

    return success_response(data=results, meta={"query": q, "total": len(results)})


@router.get("/chat/sessions/current")
async def get_current_session():
    """Return the most recently updated session, or null."""
    sessions = _build_session_cache()
    if not sessions:
        return success_response(data=None)
    return success_response(data=sessions[0])


@router.put("/chat/sessions/{session_id}")
async def upsert_session(session_id: str, req: dict):
    """Merge fields into existing session (preserves messages and metadata)."""
    existing = _get_session(session_id)
    existing.update(req)
    _save_session(session_id, existing)
    await _flush_session_to_disk(session_id)
    return success_response(data={"session_id": session_id}, message="saved")


@router.post("/chat/sessions")
async def create_session(req: dict):
    """Create a new session.

    Accepts any JSON body. Generates a session_id if not provided.
    Saves to in-memory cache and queues async disk write.
    """
    try:
        session_id = req.get("session_id") or str(uuid.uuid4())
        _save_session(session_id, req)
        await _flush_session_to_disk(session_id)
        return success_response(data={"session_id": session_id}, message="created")
    except Exception as exc:
        logger.error("create_session failed: %s", exc, exc_info=True, extra={"tag": "REQ"})
        raise HTTPException(status_code=500, detail=f"Session creation failed: {exc}")


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    data = _get_session(session_id)
    if not data.get("messages"):
        raise HTTPException(status_code=404, detail="Session not found")
    return success_response(data=data)


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    if _session_repo.delete(session_id):
        _session_memory_cache.pop(session_id, None)
        _session_dirty.discard(session_id)
        return success_response(data={"session_id": session_id}, message="deleted")
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/suggestions")
@router.get("/chat/suggestions")
async def chat_suggestions():
    """Return a set of contextual chat suggestions."""
    return success_response(data=[
        {"text": "What can you help me with?", "icon": "💬"},
        {"text": "Tell me about yourself", "icon": "👤"},
        {"text": "Write a short poem", "icon": "✍️"},
        {"text": "Explain quantum computing simply", "icon": "🔬"},
        {"text": "Help me debug my code", "icon": "🐛"},
        {"text": "Summarize a topic for me", "icon": "📝"},
    ])


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
    return success_response(data=result)
