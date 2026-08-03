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
import sys as _sys

# Ensure server parent dir is on path for host_metrics import (used in /info)
_server_parent = str(Path(__file__).parent.parent)
if _server_parent not in _sys.path:
    _sys.path.insert(0, _server_parent)


class CreateSessionRequest(BaseModel):
    """Schema for creating a new chat session."""
    session_id: Optional[str] = None
    name: Optional[str] = Field(None, max_length=200)
    model: Optional[str] = None

class UpsertSessionRequest(BaseModel):
    """Schema for updating session metadata."""
    name: Optional[str] = Field(None, max_length=200)
    archived: Optional[bool] = None
    starred: Optional[bool] = None
    pinned: Optional[bool] = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt2"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=64, ge=1, le=2048)
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


# ── Pure module-level helpers (no instance state) ──

def _count_tokens(text: str, server_state) -> int:
    """Count real tokens using the loaded tokenizer, falling back to word count.

    Args:
        text: generated text
        server_state: module exposing ``tokenizer`` / ``model_type`` attributes

    Returns:
        token count (via tokenizer.encode when available, else len(text.split()))
    """
    try:
        tokenizer = getattr(server_state, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
    except Exception:
        pass
    return len(text.split())


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


def _search_sessions_sync(q: str, limit: int) -> list:
    """Synchronous full-text search across session files on disk."""
    q_lower = q.lower().strip()
    results = []

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
                        "matches": matches[:3],
                    })
            except (json.JSONDecodeError, OSError):
                continue

    return results


# ── FileRepository-backed session store ──
from domains.infrastructure.repository import FileRepository, Serializer


class _SessionDictSerializer(Serializer[dict]):
    """JSON serializer for session dict data."""

    def serialize(self, obj: dict) -> dict:
        return obj

    def deserialize(self, data: dict) -> dict:
        return data


class InferenceRouter:
    """OOP-style router for inference, chat, sessions, and context endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="", tags=["inference"])

        self._BG_TASKS: set = set()

        _SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chat_sessions"
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._SESSIONS_DIR = _SESSIONS_DIR

        _VOICE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "voice_messages"
        _VOICE_DIR.mkdir(parents=True, exist_ok=True)
        self._VOICE_DIR = _VOICE_DIR

        self._session_repo = FileRepository[dict](
            directory=str(self._SESSIONS_DIR),
            serializer=_SessionDictSerializer(),
            key_suffix=".json",
        )

        self._session_cache: Optional[list] = None
        self._session_cache_ts: float = 0
        self._session_cache_ttl = 2.0

        self._session_memory_cache: dict[str, dict] = {}
        self._SESSION_CACHE_MAX = 500
        self._session_deleted: set[str] = set()
        self._session_dirty: set[str] = set()

        self._context_core = None
        self._vector_store_ref = None

        self._background_flush_task: Optional[asyncio.Task] = None

        self._register_routes()

    # ── Internal helpers ──

    def _session_cache_put(self, session_id: str, data: dict) -> None:
        if len(self._session_memory_cache) >= self._SESSION_CACHE_MAX and session_id not in self._session_memory_cache:
            oldest = next(iter(self._session_memory_cache))
            del self._session_memory_cache[oldest]
        self._session_memory_cache[session_id] = data

    def _get_context_core(self):
        if self._context_core is None:
            try:
                from domains.infrastructure.context_core import get_context_core
                self._context_core = get_context_core()
            except ImportError:
                return None
        if self._context_core and self._vector_store_ref and self._context_core._vector_store is None:
            try:
                from domains.inference.vector_store import simple_embed
                self._context_core.set_vector_store(self._vector_store_ref, simple_embed)
            except Exception as e:
                logger.debug("Vector store connection failed: %s", e)
        return self._context_core

    def set_vector_store_ref(self, store):
        self._vector_store_ref = store

    def _load_session_from_disk(self, session_id: str) -> dict:
        data = self._session_repo.get(session_id)
        if data is not None:
            return data
        return {"id": session_id, "messages": [], "created_at": datetime.datetime.now().isoformat(), "updated_at": datetime.datetime.now().isoformat()}

    def _get_session(self, session_id: str) -> dict:
        self._start_background_flush()
        if session_id in self._session_memory_cache:
            return self._session_memory_cache[session_id]
        data = self._load_session_from_disk(session_id)
        self._session_cache_put(session_id, data)
        return data

    def _save_session(self, session_id: str, data: dict) -> None:
        self._session_cache = None
        data["updated_at"] = datetime.datetime.now().isoformat()
        self._session_cache_put(session_id, data)
        self._session_dirty.add(session_id)

    async def _flush_session_to_disk(self, session_id: str) -> None:
        data = self._session_memory_cache.get(session_id)
        if data is None:
            self._session_dirty.discard(session_id)
            return
        data_copy = json.loads(json.dumps(data))
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._session_repo.save, session_id, data_copy)
        except Exception as exc:
            logger.warning("Disk write failed for session %s: %s", session_id, exc, extra={"tag": "REQ"})
        finally:
            self._session_dirty.discard(session_id)

    async def flush_dirty_sessions(self) -> int:
        dirty = list(self._session_dirty)
        if not dirty:
            return 0
        await asyncio.gather(*[self._flush_session_to_disk(sid) for sid in dirty], return_exceptions=True)
        return len(dirty)

    def _start_background_flush(self) -> None:
        if self._background_flush_task is not None and not self._background_flush_task.done():
            return
        async def _flush_loop():
            while True:
                await asyncio.sleep(10)
                try:
                    await self.flush_dirty_sessions()
                except Exception as e:
                    logger.debug("Background session flush failed: %s", e)
        try:
            self._background_flush_task = asyncio.create_task(_flush_loop())
        except RuntimeError:
            pass

    def _build_session_cache(self) -> list:
        now = time.time()
        if self._session_cache is not None and now - self._session_cache_ts < self._session_cache_ttl:
            return self._session_cache
        sessions = []
        for sid in self._session_repo.keys():
            data = self._session_repo.get(sid)
            if data is None:
                continue
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
        self._session_cache = sessions
        self._session_cache_ts = now
        return sessions

    # ── Route handlers ──

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        from domains.models.provider import get_provider
        from startup_progress import STARTUP_PHASE
        import state as _gen_state

        if STARTUP_PHASE.get("phase") != "ready" or _gen_state.model is None:
            raise HTTPException(status_code=503, detail="Model still loading — please wait.")

        provider = get_provider("default")
        if provider is None:
            raise HTTPException(status_code=503, detail="No provider available")

        provider_messages = [{"role": "user", "content": req.prompt}]
        try:
            import time as _time
            _t0 = _time.monotonic()
            result = await provider.chat(
                provider_messages,
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
            logger.info(
                "DBG generate handler: provider=%s elapsed=%.3fs result=%r",
                getattr(provider, "_text_name", type(provider).__name__),
                _time.monotonic() - _t0,
                result[:80],
                extra={"tag": "DBG"},
            )
            tokens = _count_tokens(result, _gen_state)
            actual_model = _gen_state.model_type or req.model
            try:
                from domains.infrastructure.server_state import get_server_state
                get_server_state().record_inference(tokens=tokens, elapsed_ms=0, model=actual_model)
            except Exception as e:
                logger.debug("Failed to record inference metrics: %s", e)
            try:
                from domains.infrastructure.conversation_log import capture
                capture(
                    req.prompt,
                    result,
                    model=actual_model,
                    tokens_generated=tokens,
                    elapsed_ms=(_time.monotonic() - _t0) * 1000,
                    temperature=req.temperature,
                )
            except Exception as e:
                logger.debug("Failed to capture conversation: %s", e)
            return GenerateResponse(text=result, model=actual_model, tokens_generated=tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_stream(self, req: GenerateRequest, request: Request) -> StreamingResponse:
        from startup_progress import STARTUP_PHASE
        import state as _stream_state

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
            collected = []
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
                        collected.append(token)
                        yield sse_token("generate", token)
            except Exception as e:
                yield sse_error("generate", "STREAMING", str(e))
                return
            elapsed = (datetime.datetime.now() - start).total_seconds() * 1000
            try:
                from domains.infrastructure.server_state import get_server_state
                get_server_state().record_inference(
                    tokens=token_count, elapsed_ms=elapsed, model=_stream_state.model_type or req.model
                )
            except Exception as e:
                logger.debug("Failed to record inference metrics: %s", e)
            try:
                from domains.infrastructure.conversation_log import capture
                capture(
                    req.prompt,
                    "".join(collected),
                    model=_stream_state.model_type or req.model,
                    tokens_generated=token_count,
                    elapsed_ms=elapsed,
                    temperature=req.temperature,
                )
            except Exception as e:
                logger.debug("Failed to capture conversation: %s", e)
            yield sse_token("generate", "", done=True, meta={"tokens": token_count, "elapsed_ms": round(elapsed, 1)})

        return StreamingResponse(generate(), media_type="text/event-stream")

    async def get_info(self):
        from host_metrics import sample_host_metrics_async
        import state as server_state

        data = {
            "api_version": "1.0.0",
            "model": {
                "type": server_state.model_type,
                "loaded": server_state.model is not None,
            },
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

        return data

    async def get_info_soul(self):
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
        except Exception as e:
            logger.debug("Failed to build soul info: %s", e)
        return soul_info

    async def root(self):
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

    async def list_chat_tools(self):
        try:
            from domains.agents.tools import get_tool_registry
            return {"tools": get_tool_registry().list_tools()}
        except Exception as e:
            logger.warning("Failed to list tools: %s", e, extra={"tag": "INF"})
            return {"tools": []}

    async def chat_stream(self, req: ChatRequest, request: Request) -> StreamingResponse:
        from startup_progress import STARTUP_PHASE

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

            logger.debug("chat_stream: yielding thinking event")
            yield _sse_event("chat", "STREAMING", "thinking",
                data={}, message="Thinking...")

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

            session_data = self._get_session(session_id)
            session_data.setdefault("messages", []).append({
                "role": "user",
                "content": user_msg,
                "timestamp": datetime.datetime.now().isoformat(),
            })

            ctx_core = self._get_context_core()
            context_info = {}
            frame = None
            skip_context = False
            if ctx_core and req.use_context_core:
                try:
                    from domains.learner.knowledge import get_knowledge_memory
                    kmem = get_knowledge_memory()
                    if kmem.stats().get("total_items", 0) == 0 and not req.knowledge:
                        skip_context = True
                except Exception as e:
                    logger.debug("Knowledge memory check failed: %s", e)
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

            if req.agent_id:
                try:
                    from domains.agents.system import get_agent_system
                    agent_sys = get_agent_system()
                    agent_instructions = agent_sys.get_instructions(req.agent_id)
                    if agent_instructions:
                        agent_msg = {"role": "system", "content": f"[AGENT: {req.agent_id}]\n{agent_instructions}"}
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

            tool_result_data = None
            try:
                from domains.agents.tools import get_tool_registry
                tool_reg = get_tool_registry()
                tool_intent = tool_reg.detect_tool_intent(user_msg)
                if tool_intent:
                    tool_name, tool_args = tool_intent
                    spec = tool_reg.get(tool_name)
                    if spec and spec.requires_approval:
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

            if context_info:
                yield _sse_event("chat", "STREAMING", "working",
                    data={"context": context_info},
                    message=f"{len(context_info.get('layers', []))} context layers")

            knowledge_retrieved = []
            try:
                enrichment = await asyncio.to_thread(_enrich_knowledge, user_msg, False, 5)
                if enrichment.get("facts"):
                    knowledge_retrieved = enrichment["facts"]
            except Exception as e:
                logger.debug("Knowledge enrichment failed: %s", e)

            all_knowledge = knowledge_retrieved + (req.knowledge or [])
            if all_knowledge:
                try:
                    from domains.models.provider import KnowledgeProcessor, apply_processors
                    k_proc = KnowledgeProcessor(knowledge=all_knowledge)
                    provider_messages = await apply_processors(provider_messages, [k_proc])
                except Exception as e:
                    logger.debug("Knowledge processor failed: %s", e)

            try:
                from domains.models.provider import get_provider
                provider = get_provider("default")

                if provider is not None:
                    full_response_parts: list[str] = []
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
                                    full_response_parts.append(token)
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
                    yield sse_error("chat", "STREAMING", "No inference provider loaded")

                full_response = "".join(full_response_parts)

                if session_id not in self._session_deleted:
                    session_data["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.datetime.now().isoformat(),
                    })
                    self._save_session(session_id, session_data)
                    await self._flush_session_to_disk(session_id)
                else:
                    logger.info("Session %s was deleted during generation, skipping save", session_id, extra={"tag": "INF"})

                duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
                tokens = len(full_response.split())
                _post_gen_tasks = []

                try:
                    from domains.infrastructure.conversation_log import capture
                    capture(
                        user_msg or "",
                        full_response,
                        model=_check_state.model_type or req.model,
                        tokens_generated=tokens,
                        elapsed_ms=duration_ms,
                        temperature=req.temperature,
                        meta={"session_id": session_id},
                    )
                except Exception as e:
                    logger.debug("Failed to capture conversation: %s", e)

                try:
                    from domains.feedback.response_tracker import get_response_tracker
                    tracker = get_response_tracker()
                    _post_gen_tasks.append(asyncio.to_thread(
                        tracker.log,
                        user_message=user_msg or "",
                        assistant_response=full_response,
                        model=req.model,
                        config={"temperature": req.temperature, "max_tokens": req.max_tokens},
                        session_id=session_id,
                        user_id=req.user_id or "default",
                        tokens_generated=tokens,
                        duration_ms=duration_ms,
                        has_images=bool(req.images),
                    ))
                except Exception as e:
                    logger.debug("ResponseTracker.log failed: %s", e)

                try:
                    from domains.infrastructure.server_state import get_server_state
                    _post_gen_tasks.append(asyncio.to_thread(
                        get_server_state().record_inference,
                        tokens=tokens, elapsed_ms=duration_ms, model=_check_state.model_type or req.model,
                    ))
                except Exception as e:
                    logger.debug("Failed to record inference metrics: %s", e)

                if ctx_core and req.use_context_core:
                    _post_gen_tasks.append(asyncio.to_thread(ctx_core.add_response, full_response, model=req.model))

                try:
                    from domains.learner import get_learner
                    _post_gen_tasks.append(asyncio.to_thread(
                        get_learner().ingest_conversation, [(user_msg, full_response)]
                    ))
                except Exception as e:
                    logger.debug("Continual learner ingest failed: %s", e)

                if _post_gen_tasks:
                    await asyncio.gather(*_post_gen_tasks, return_exceptions=True)

                try:
                    from domains.learner.entity_extractor import extract_and_store
                    task = asyncio.create_task(extract_and_store(user_msg or "", full_response))
                    self._BG_TASKS.add(task)
                    task.add_done_callback(self._BG_TASKS.discard)
                except Exception as e:
                    logger.debug("Entity extraction failed: %s", e)

                logger.info("Chat stream: generated %d chars", len(full_response), extra={"tag": "INF", "context": {"char_count": len(full_response), "session_id": session_id}})

            except Exception as e:
                yield sse_error("chat", "STREAMING", str(e))
                yield sse_token("chat", "", done=True)

        return StreamingResponse(generate(), media_type="text/event-stream")

    async def inspect_context(self) -> dict:
        ctx_core = self._get_context_core()
        if not ctx_core:
            return {"error": "ContextCore not available"}
        return ctx_core.get_context_inspector()

    async def store_fact(self, key: str, value: str) -> dict:
        ctx_core = self._get_context_core()
        if not ctx_core:
            return {"error": "ContextCore not available"}
        ctx_core.store_fact(key, value)
        return {"stored": key}

    async def get_facts(self, query: str = "") -> dict:
        ctx_core = self._get_context_core()
        if not ctx_core:
            return {"error": "ContextCore not available", "facts": []}
        if query:
            return {"facts": ctx_core.search_semantic(query)}
        return {"facts": [{"key": k, **v} for k, v in ctx_core.semantic_memory.items()]}

    async def reset_context(self, all: bool = False) -> dict:
        self._context_core = None
        ctx_core = self._get_context_core()
        if not ctx_core:
            return {"error": "ContextCore not available"}
        if all:
            ctx_core.reset_all()
        else:
            ctx_core.reset_session()
        return {"reset": "session" if not all else "all"}

    async def chat(self, req: ChatRequest) -> ChatResponse:
        from domains import get_chat_domain
        from startup_progress import STARTUP_PHASE
        import state as _chat_state

        if STARTUP_PHASE.get("phase") != "ready" or _chat_state.model is None:
            raise HTTPException(status_code=503, detail="Model still loading — please wait.")

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
        except Exception as e:
            logger.debug("Circuit breaker check failed: %s", e)

        user_msg = _extract_user_message(req.messages)
        if not user_msg:
            raise HTTPException(status_code=400, detail="No user message")

        system_prompt = req.system_prompt or ""

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        if req.images:
            content_parts = [{"type": "text", "text": user_msg}]
            for img_data in req.images:
                content_parts.append({"type": "image_url", "image_url": {"url": img_data}})
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = content_parts
                    break

        chat_domain = get_chat_domain()

        if req.agent_id:
            try:
                from domains.agents.system import get_agent_system
                agent_sys = get_agent_system()
                agent_instructions = agent_sys.get_instructions(req.agent_id)
                if agent_instructions:
                    system_prompt = f"{system_prompt}\n\n[AGENT: {req.agent_id}]\n{agent_instructions}" if system_prompt else f"[AGENT: {req.agent_id}]\n{agent_instructions}"
            except Exception:
                logger.warning("Failed to inject agent instructions", exc_info=True, extra={"tag": "INF"})

        if req.knowledge:
            knowledge_str = "\n".join(f"- {k}" for k in req.knowledge)
            system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{knowledge_str}" if system_prompt else f"Use the following context to answer:\n{knowledge_str}"

        try:
            enrichment = await asyncio.to_thread(_enrich_knowledge, user_msg, False, 5)
            if enrichment.get("facts"):
                k_text = "\n".join(f"- {f}" for f in enrichment["facts"])
                system_prompt = f"{system_prompt}\n\nUse the following context to answer:\n{k_text}" if system_prompt else f"Use the following context to answer:\n{k_text}"
        except Exception as e:
            logger.debug("Knowledge enrichment failed: %s", e)

        result = await chat_domain.respond(
            messages=messages,
            model=_chat_state.model_type or req.model,
            system_prompt=system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            session_id=req.session_id or "default",
            user_id=req.user_id or "default",
        )

        try:
            from domains.infrastructure.server_state import get_server_state
            tokens = _count_tokens(result.text, _chat_state)
            get_server_state().record_inference(
                tokens=tokens, elapsed_ms=0, model=_chat_state.model_type or req.model
            )
        except Exception as e:
            logger.debug("Failed to record inference metrics: %s", e)

        try:
            from domains.infrastructure.conversation_log import capture
            capture(
                user_msg or "",
                result.text,
                model=_chat_state.model_type or req.model,
                tokens_generated=_count_tokens(result.text, _chat_state),
                temperature=req.temperature,
                meta={"session_id": req.session_id or "default"},
            )
        except Exception as e:
            logger.debug("Failed to capture conversation: %s", e)

        return ChatResponse(
            message=result.text,
            session_id=result.session_id,
            done=result.done,
        )

    async def send_voice_message(
        self,
        session_id: str,
        file: UploadFile = File(...),
        duration_ms: int = Form(0),
    ):
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Only audio files accepted")

        msg_id = str(uuid.uuid4())
        session_msg_dir = (self._VOICE_DIR / session_id).resolve()
        if not str(session_msg_dir).startswith(str(self._VOICE_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid session ID")
        session_msg_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "audio.m4a").suffix or ".m4a"
        audio_path = session_msg_dir / f"{msg_id}{ext}"

        content = await file.read()
        audio_path.write_bytes(content)

        session_data = self._get_session(session_id)
        session_data.setdefault("messages", []).append({
            "role": "user",
            "content": "[Voice Message]",
            "audio_path": f"{session_id}/{msg_id}{ext}",
            "audio_duration_ms": duration_ms,
            "timestamp": datetime.datetime.now().isoformat(),
            "_voice": True,
        })
        self._save_session(session_id, session_data)

        return success_response(data={
            "message_id": msg_id,
            "audio_path": f"{session_id}/{msg_id}{ext}",
            "session_id": session_id,
        })

    async def get_voice_audio(self, session_id: str, message_id: str):
        base = self._VOICE_DIR.resolve()
        audio_path = (self._VOICE_DIR / session_id / message_id).resolve()
        if not str(audio_path).startswith(str(base)):
            raise HTTPException(status_code=403, detail="Invalid path")
        if not audio_path.exists():
            for ext in [".m4a", ".wav", ".mp3", ".ogg", ".webm"]:
                candidate = audio_path.parent / f"{audio_path.stem}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            else:
                raise HTTPException(status_code=404, detail="Audio not found")
        return FileResponse(str(audio_path), media_type="audio/m4a")

    async def list_sessions(self, archived: Optional[bool] = None):
        sessions = await asyncio.to_thread(self._build_session_cache)
        if archived is not None:
            sessions = [s for s in sessions if s.get("archived", False) == archived]
        return success_response(data=sessions)

    async def search_sessions(self, q: str = "", limit: int = 20):
        if not q.strip():
            return success_response(data=[], meta={"query": q, "total": 0})
        results = await asyncio.to_thread(_search_sessions_sync, q, limit)
        return success_response(data=results, meta={"query": q, "total": len(results)})

    async def get_current_session(self):
        sessions = await asyncio.to_thread(self._build_session_cache)
        if not sessions:
            return success_response(data=None)
        return success_response(data=sessions[0])

    async def upsert_session(self, session_id: str, req: UpsertSessionRequest):
        existing = self._get_session(session_id)
        update_data = req.model_dump(exclude_none=True)
        for key, value in update_data.items():
            existing[key] = value
        self._save_session(session_id, existing)
        await self._flush_session_to_disk(session_id)
        return success_response(data={"session_id": session_id}, message="saved")

    async def create_session(self, req: CreateSessionRequest):
        try:
            session_id = req.session_id or str(uuid.uuid4())
            session_data = req.model_dump(exclude_none=True)
            session_data["session_id"] = session_id
            self._save_session(session_id, session_data)
            await self._flush_session_to_disk(session_id)
            return success_response(data={"session_id": session_id}, message="created")
        except Exception as exc:
            logger.error("create_session failed: %s", exc, exc_info=True, extra={"tag": "REQ"})
            raise HTTPException(status_code=500, detail=f"Session creation failed: {exc}")

    async def get_session(self, session_id: str):
        data = self._get_session(session_id)
        if not data.get("messages"):
            raise HTTPException(status_code=404, detail="Session not found")
        return success_response(data=data)

    async def delete_session(self, session_id: str):
        if self._session_repo.delete(session_id):
            self._session_memory_cache.pop(session_id, None)
            self._session_dirty.discard(session_id)
            self._session_deleted.add(session_id)
            return success_response(data={"session_id": session_id}, message="deleted")
        raise HTTPException(status_code=404, detail="Session not found")

    async def chat_suggestions(self):
        return success_response(data=[
            {"text": "What can you help me with?", "icon": "chat"},
            {"text": "Tell me about yourself", "icon": "user"},
            {"text": "Write a short poem", "icon": "pen"},
            {"text": "Explain quantum computing simply", "icon": "atom"},
            {"text": "Help me debug my code", "icon": "bug"},
            {"text": "Summarize a topic for me", "icon": "document"},
        ])

    async def list_model_providers(self):
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

    # ── Route registration ──

    def _register_routes(self):
        r = self.router
        r.add_api_route("/inference/generate", self.generate, methods=["POST"], response_model=GenerateResponse)
        r.add_api_route("/inference/generate/stream", self.generate_stream, methods=["POST"])
        r.add_api_route("/info", self.get_info, methods=["GET"])
        r.add_api_route("/info/soul", self.get_info_soul, methods=["GET"])
        r.add_api_route("/", self.root, methods=["GET"])
        r.add_api_route("/chat/tools", self.list_chat_tools, methods=["GET"])
        r.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        r.add_api_route("/context/inspect", self.inspect_context, methods=["GET"])
        r.add_api_route("/context/fact", self.store_fact, methods=["POST"])
        r.add_api_route("/context/facts", self.get_facts, methods=["GET"])
        r.add_api_route("/context/reset", self.reset_context, methods=["POST"])
        r.add_api_route("/chat", self.chat, methods=["POST"], response_model=ChatResponse)
        r.add_api_route("/chat/voice/{session_id}", self.send_voice_message, methods=["POST"])
        r.add_api_route("/chat/audio/{session_id}/{message_id}", self.get_voice_audio, methods=["GET"])
        r.add_api_route("/chat/sessions", self.list_sessions, methods=["GET"])
        r.add_api_route("/chat/sessions/search", self.search_sessions, methods=["GET"])
        r.add_api_route("/chat/sessions/current", self.get_current_session, methods=["GET"])
        r.add_api_route("/chat/sessions/{session_id}", self.upsert_session, methods=["PUT"])
        r.add_api_route("/chat/sessions", self.create_session, methods=["POST"])
        r.add_api_route("/chat/sessions/{session_id}", self.get_session, methods=["GET"])
        r.add_api_route("/chat/sessions/{session_id}", self.delete_session, methods=["DELETE"])
        r.add_api_route("/suggestions", self.chat_suggestions, methods=["GET"])
        r.add_api_route("/chat/suggestions", self.chat_suggestions, methods=["GET"])
        r.add_api_route("/providers", self.list_model_providers, methods=["GET"])


_instance = InferenceRouter()
router = _instance.router

def set_vector_store_ref(ref):
    return _instance.set_vector_store_ref(ref)

def flush_dirty_sessions():
    return _instance.flush_dirty_sessions()
