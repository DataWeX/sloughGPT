"""
Inference Router - Chat and text generation endpoints
"""
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Any, Optional, List, AsyncIterator
from pathlib import Path
import json
import logging
import threading

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from domains.infrastructure.errors import AppError
from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
from domains.infrastructure.server_state import get_server_state
from domains.infrastructure.conversation_log import capture
from domains.models.provider import get_provider, KnowledgeProcessor, apply_processors
from domains.agents.system import get_agent_system
from domains.agents.tools import get_tool_registry
from domains.memory.memory_service import get_memory_service
from domains.cognitive.rag_service import get_rag_service
from domains.feedback.response_tracker import get_response_tracker
from domains.learner import get_learner
from domains.learner.entity_extractor import extract_and_store
from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
from domains.infrastructure.request_coalescer import get_coalescer
from config import ServerConfig
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.inference")

cfg = ServerConfig.from_env()

try:
    from domains.api.sse_envelope import sse_event as _sse_event, sse_token, sse_error
except ImportError:
    import json as _json
    def _sse_event(stream, phase, status, data=None, meta=None, message=""):
        return "data: " + _json.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message,
        }) + "\n\n"
    def sse_token(stream, token, done=False, meta=None, elapsed_ms=None) -> dict:
        """sse_token."""
        phase = "STREAMING"
        status = "complete" if done else "working"
        m = dict(meta) if meta else {}
        if done and elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
        return _sse_event(stream, phase, status, {"token": token}, m, "")
    def sse_error(stream, phase, error, meta=None, code=None, http_status=None) -> dict:
        """sse_error."""
        data = {"error": error}
        if code is not None:
            data["code"] = code
        if http_status is not None:
            data["http_status"] = http_status
        return _sse_event(stream, phase, "error", data, meta or {}, f"Error: {error}")
import asyncio
import datetime
import uuid
import time
import sys as _sys

# Ensure server parent dir is on path for host_metrics import (used in /info)
_server_parent = str(Path(__file__).parent.parent)
if _server_parent not in _sys.path:
    _sys.path.insert(0, _server_parent)

def _model_ready() -> bool:
    """True when a model is actually materialized and ready for inference.

    Checks both server_state.model (direct load) and the provider's _model
    attribute.  For lazy-guard providers, the model lives on the provider
    (not the SloNetServer) until first use, so we must check the provider
    directly.
    """
    import state as server_state
    if server_state.model is not None:
        return True
    provider = server_state.provider
    if provider is None:
        return False
    return getattr(provider, '_model', None) is not None


_memory_pressure_cache: Optional[str] = None
_memory_pressure_cache_ts: float = 0.0
_memory_pressure_lock = threading.Lock()


def _check_memory_pressure() -> Optional[str]:
    """Return an error message if system memory is critically low, else None.

    Blocks new inference when >95% memory used to prevent OOM kills (SIGKILL -9).
    Allows inference to proceed (with a warning log) when 85-95% used.
    Caches result for 2 seconds to avoid repeated psutil calls.
    """
    global _memory_pressure_cache, _memory_pressure_cache_ts
    now = time.time()
    with _memory_pressure_lock:
        if _memory_pressure_cache is not None and now - _memory_pressure_cache_ts < 2.0:
            return _memory_pressure_cache if _memory_pressure_cache else None
    try:
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 95:
            result = f"System memory at {mem.percent:.0f}% — too low for safe inference. Free some memory and retry."
            with _memory_pressure_lock:
                _memory_pressure_cache = result
                _memory_pressure_cache_ts = now
            return result
        if mem.percent > 85:
            logger.warning("Memory pressure: %.0f%% used — inference may be slow", mem.percent, extra={"tag": "INF"})
    except Exception as exc:
        logger.warning("Memory pressure check failed: %s", exc)
    with _memory_pressure_lock:
        _memory_pressure_cache = ""
        _memory_pressure_cache_ts = now
    return None

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
    role: str = Field(..., pattern=r'^(user|assistant|system)$')
    content: str = Field(..., min_length=1, max_length=100000)

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "qwen2.5-0.5b-instruct"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=2048, description="Alias for max_tokens (frontend compat)")
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge: Optional[List[str]] = None
    images: Optional[List[str]] = Field(default=None, description="Base64 encoded images")
    use_context_core: bool = Field(default=True, description="Use ContextCore for multi-layer context")
    use_rag: bool = Field(default=True, description="Use production RAG for document-grounded responses")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for role-based system instructions")

    def model_post_init(self, __context: Any) -> None:
        """Resolve max_new_tokens alias into max_tokens."""
        try:
            if self.max_new_tokens is not None and self.max_tokens == 128:
                object.__setattr__(self, 'max_tokens', self.max_new_tokens)

        except Exception as e:
            classify_and_raise(e, source="inference.model_post_init")
class ChatResponse(BaseModel):
    message: str
    session_id: str
    done: bool = True

class ChatControlRequest(BaseModel):
    session_id: str
    action: str = Field(..., pattern=r'^(cancel|approve|context)$')
    tool_name: Optional[str] = None
    approved: Optional[bool] = None
    context: Optional[str] = None

# In-memory store for pending control requests per session
_chat_control_store: dict[str, dict] = {}
_chat_control_lock = threading.Lock()

def get_chat_control(session_id: str) -> Optional[dict]:
    """Get and consume pending control for a session."""
    with _chat_control_lock:
        return _chat_control_store.pop(session_id, None)

def set_chat_control(session_id: str, control: dict) -> None:
    """Set pending control for a session."""
    with _chat_control_lock:
        _chat_control_store[session_id] = control

# Cache for partial chat responses (for Last-Event-ID reconnection)
_chat_response_cache: dict[str, dict] = {}
_chat_cache_lock = threading.Lock()
_CHAT_CACHE_MAX_SESSIONS = 100
_CHAT_CACHE_MAX_AGE_S = 300  # 5 minutes

def get_chat_response_cache(session_id: str) -> Optional[dict]:
    """Get cached response for a session."""
    with _chat_cache_lock:
        return _chat_response_cache.get(session_id)

def set_chat_response_cache(session_id: str, response: dict) -> None:
    """Cache a partial response for a session."""
    import time
    with _chat_cache_lock:
        # Purge old entries if cache is full
        if len(_chat_response_cache) >= _CHAT_CACHE_MAX_SESSIONS:
            now = time.time()
            expired = [k for k, v in _chat_response_cache.items()
                       if now - v.get("timestamp", 0) > _CHAT_CACHE_MAX_AGE_S]
            for k in expired[:10]:  # Remove up to 10 expired entries
                _chat_response_cache.pop(k, None)
        _chat_response_cache[session_id] = {**response, "timestamp": time.time()}

def clear_chat_response_cache(session_id: str) -> None:
    """Clear cached response for a session."""
    with _chat_cache_lock:
        _chat_response_cache.pop(session_id, None)

class ContextInspectorResponse(BaseModel):
    system_prompt: str
    session_messages: List[dict]
    working_memory: List[dict]
    semantic_keys: List[str]
    episodic_count: int
    sensory_buffer_size: int
    last_frame: Optional[dict]

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    model: str = "qwen2.5-0.5b-instruct"

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
    except Exception as e:
        logger.debug("tokenizer encode fallback failed: %s", e)
    return len(text.split())

def _extract_user_message(messages: List[Message]) -> Optional[str]:
    """Extract the last user message from conversation."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content or None
    return None

_META_WEIGHT_CACHE: dict[str, tuple[float, dict]] = {}
_META_WEIGHT_CACHE_TTL = 5.0  # seconds
_META_WEIGHT_CACHE_LOCK = threading.Lock()

def _apply_meta_weights(
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    user_message: str,
    user_id: str = "default",
) -> dict:
    """Apply feedback-driven meta-weight adjustments to generation parameters.

    Looks up similar past messages in the feedback database and adjusts
    temperature, top_p, top_k, and repetition_penalty accordingly.
    Results are cached for 5 seconds to avoid repeated similarity searches.

    Returns a dict of adjusted parameters to pass to the provider.
    """
    import time
    cache_key = f"{user_id}:{hash(user_message)}"
    now = time.monotonic()

    with _META_WEIGHT_CACHE_LOCK:
        if cache_key in _META_WEIGHT_CACHE:
            cached_time, cached_params = _META_WEIGHT_CACHE[cache_key]
            if now - cached_time < _META_WEIGHT_CACHE_TTL:
                return cached_params

    try:
        from domains.feedback.meta_weights import get_meta_weight_manager
        manager = get_meta_weight_manager()
        adj = manager.get_adjustment(
            user_message=user_message, k=5, user_id=user_id,
        )
        result = {
            "temperature": adj.temperature,
            "top_p": adj.top_p,
            "top_k": adj.top_k,
            "repetition_penalty": adj.repetition_penalty,
        }
        with _META_WEIGHT_CACHE_LOCK:
            _META_WEIGHT_CACHE[cache_key] = (now, result)
            if len(_META_WEIGHT_CACHE) > 1000:
                logger.info("Meta-weight cache overflow, clearing %d entries", len(_META_WEIGHT_CACHE))
                safe_audit_log("inference.meta_weight_cache_clear", resource="meta_weight_cache", detail=f"entries_cleared={len(_META_WEIGHT_CACHE)}")
                _META_WEIGHT_CACHE.clear()
        return result
    except Exception as e:
        logger.debug("Meta-weight adjustment failed: %s", e)
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

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
    max_matches_per_session = 3  # stop scanning messages after this many matches

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
        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except (OSError, ValueError):
                return 0.0
        for f in sorted(sdir.glob("*.json"), key=_safe_mtime, reverse=True):
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
                    if len(matches) >= max_matches_per_session:
                        break
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

    _REQUIRED_KEYS = {"id", "messages"}

    def serialize(self, obj: dict) -> dict:
        if not isinstance(obj, dict):
            raise ValueError(f"Session must be a dict, got {type(obj).__name__}")
        result = dict(obj)
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "messages" not in result:
            result["messages"] = []
        return result

    def deserialize(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError(f"Session data must be a dict, got {type(data).__name__}")
        result = dict(data)
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "messages" not in result:
            result["messages"] = []
        return result
class InferenceRouter:
    """OOP-style router for inference, chat, sessions, and context endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="", tags=["inference"])

        self._BG_TASKS: set = set()
        self._bg_tasks_lock = threading.Lock()

        _SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "chat_sessions"
        self._SESSIONS_DIR = _SESSIONS_DIR

        _VOICE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "voice_messages"
        self._VOICE_DIR = _VOICE_DIR

        self._dirs_created = False

        self._session_repo = FileRepository[dict](
            directory=str(self._SESSIONS_DIR),
            serializer=_SessionDictSerializer(),
            key_suffix=".json",
        )

        self._session_cache: Optional[list] = None
        self._session_cache_ts: float = 0
        self._session_cache_ttl = 5.0

        self._session_metadata_cache: Optional[list] = None
        self._session_metadata_cache_ts: float = 0

        self._session_memory_cache: dict[str, dict] = {}
        self._SESSION_CACHE_MAX = 500
        self._session_deleted: set[str] = set()
        self._session_dirty: set[str] = set()

        self._context_core = None
        self._vector_store_ref = None

        self._background_flush_task: Optional[asyncio.Task] = None

        self._register_routes()

    def _bg_tasks_lock_discard(self, task: asyncio.Task) -> None:
        """Thread-safe discard callback for background tasks."""
        with self._bg_tasks_lock:
            self._BG_TASKS.discard(task)

    def _ensure_dirs(self):
        """Create session/voice directories on first access (lazy)."""
        if self._dirs_created:
            return
        self._SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._VOICE_DIR.mkdir(parents=True, exist_ok=True)
        self._dirs_created = True

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
                logger.warning("Vector store connection failed: %s", e)
        return self._context_core

    def set_vector_store_ref(self, store) -> dict:
        """set_vector_store_ref."""
        self._vector_store_ref = store
        return {"status": "ok"}
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
        try:
            """flush_dirty_sessions."""
            dirty = list(self._session_dirty)
            if not dirty:
                return 0
            await asyncio.gather(*[self._flush_session_to_disk(sid) for sid in dirty], return_exceptions=True)
            return len(dirty)

        except Exception as e:
            classify_and_raise(e, source="inference.flush_dirty_sessions")
    def _start_background_flush(self) -> None:
        if self._background_flush_task is not None and not self._background_flush_task.done():
            return
        async def _flush_loop():
            while True:
                await asyncio.sleep(10)
                try:
                    await self.flush_dirty_sessions()
                except Exception as e:
                    logger.warning("Background session flush failed: %s", e)
        try:
            self._background_flush_task = asyncio.create_task(_flush_loop())
        except RuntimeError:
            pass

    def _build_session_metadata_index(self) -> list:
        """Build a lightweight metadata index without loading full message content.

        Reads only the first 4KB of each file to extract id, name, updated_at,
        created_at, and message_count. Falls back to full read if partial parse fails.
        """
        self._ensure_dirs()
        now = time.time()
        if self._session_metadata_cache is not None and now - self._session_metadata_cache_ts < self._session_cache_ttl:
            return self._session_metadata_cache

        metadata = []
        for sdir in [self._SESSIONS_DIR, self._SESSIONS_DIR.parent / "conversations"]:
            if not sdir.is_dir():
                continue
            for f in sdir.glob("*.json"):
                try:
                    with open(f, "r") as fh:
                        raw = fh.read(4096)
                    if not raw.strip():
                        continue
                    # Try to parse the partial JSON for header fields
                    data = json.loads(raw)
                    sid = data.get("id") or data.get("session_id") or f.stem
                    name = data.get("name", "") or ""
                    messages = data.get("messages", [])
                    # If messages array is truncated, count from file size heuristic
                    msg_count = len(messages)
                    if not name and messages:
                        name = messages[0].get("content", "").split("\n")[0][:60]
                    if not name:
                        name = sid
                    metadata.append({
                        "id": sid,
                        "name": name,
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", "") or data.get("created_at", ""),
                        "message_count": msg_count,
                    })
                except (json.JSONDecodeError, OSError):
                    continue

        metadata.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        self._session_metadata_cache = metadata
        self._session_metadata_cache_ts = now
        return metadata

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

    async def generate(self, req: GenerateRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> GenerateResponse:
        """generate."""

        from startup_progress import STARTUP_PHASE
        import state as _gen_state

        if STARTUP_PHASE.get("phase") != "ready" or not _model_ready():
            raise_error("Model still loading — please wait.", "E_BAD_REQUEST", status_code=503)

        mem_err = _check_memory_pressure()
        if mem_err:
            raise_error(mem_err, "E_RESOURCE", status_code=503)

        provider = get_provider("default")
        if provider is None:
            raise_error("No provider available", "E_BAD_REQUEST", status_code=503)

        provider_messages = [{"role": "user", "content": req.prompt}]
        try:
            _t0 = time.monotonic()
            gen_params = _apply_meta_weights(
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                user_message=req.prompt,
            )

            import state as _gen_state
            _coalescer = get_coalescer()
            _coalesce_key = _coalescer.hash(provider_messages, gen_params, req.max_new_tokens, _gen_state.model_type)
            existing = await _coalescer.start(_coalesce_key)
            if existing is not None:
                await existing.event.wait()
                if existing.error is not None:
                    raise existing.error
                result = existing.result
                tokens = _count_tokens(result, _gen_state)
                actual_model = _gen_state.model_type or req.model
                return GenerateResponse(text=result, model=actual_model, tokens_generated=tokens)

            result = await provider.chat(
                provider_messages,
                max_tokens=req.max_new_tokens,
                **gen_params,
            )
            logger.debug(
                "generate handler",
                extra={
                    "tag": "INFO",
                    "context": {
                        "provider": getattr(provider, "_text_name", type(provider).__name__),
                        "elapsed_ms": round((time.monotonic() - _t0) * 1000, 1),
                        "result": str(result)[:80],
                    },
                },
            )
            tokens = _count_tokens(result, _gen_state)
            actual_model = _gen_state.model_type or req.model
            try:

                get_server_state().record_inference(tokens=tokens, elapsed_ms=round((time.monotonic() - _t0) * 1000, 1), model=actual_model)
            except Exception as e:
                logger.warning("Failed to record inference metrics: %s", e)
            try:

                capture(
                    req.prompt,
                    result,
                    model=actual_model,
                    tokens_generated=tokens,
                    elapsed_ms=(time.monotonic() - _t0) * 1000,
                    temperature=req.temperature,
                )
            except Exception as e:
                logger.warning("Failed to capture conversation: %s", e)
            await _coalescer.complete(_coalesce_key, result)
            return GenerateResponse(text=result, model=actual_model, tokens_generated=tokens)
        except Exception as e:
            try:
                await _coalescer.complete_error(_coalesce_key, e)
            except Exception:
                pass
            logger.warning("Generate failed: %s", e, extra={"tag": "INF"})
            classify_and_raise(e, source="generate")

    async def generate_stream(self, req: GenerateRequest, request: Request, auth_user: dict = Depends(require_auth_if_enabled)) -> StreamingResponse:
        """generate_stream."""
        from startup_progress import STARTUP_PHASE
        import state as _stream_state

        if STARTUP_PHASE.get("phase") != "ready" or not _model_ready():
            async def error_stream() -> AsyncIterator[str]:
                """error_stream."""
                yield sse_error("generate", "IDLE", "Model still loading — please wait.", code="MODEL_LOADING", http_status=503)
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        mem_err = _check_memory_pressure()
        if mem_err:
            async def oom_stream() -> AsyncIterator[str]:
                yield sse_error("generate", "IDLE", mem_err, code="MODEL_OOM", http_status=503)
            return StreamingResponse(oom_stream(), media_type="text/event-stream")

        async def generate() -> AsyncIterator[str]:
            """generate."""

            provider = get_provider("default")
            if provider is None:
                yield sse_error("generate", "IDLE", "No provider available", code="E_INFRA_REGISTRY", http_status=503)
                return

            cancel_event = threading.Event()
            _mgr = get_cancel_manager()
            _op_id = _mgr.register(
                OpType.INFERENCE, f"generate:{req.prompt[:40]}",
                cancel_fn=lambda: cancel_event.set(),
            )
            _mgr.start(_op_id)

            provider_messages = [{"role": "user", "content": req.prompt}]
            start = datetime.datetime.now()
            token_count = 0
            collected = []
            _token_gen_start = time.time()
            _max_token_wait_s = cfg.generate_timeout
            _heartbeat_interval_s = 10.0
            _last_heartbeat = time.time()
            _batch: list[str] = []
            _batch_start = time.time()
            _BATCH_MAX = 5
            _BATCH_INTERVAL_S = 0.005
            _token_count = 0
            gen_params = _apply_meta_weights(
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                user_message=req.prompt,
            )

            _coalescer = get_coalescer()
            _coalesce_key = _coalescer.hash(provider_messages, gen_params, req.max_new_tokens, _stream_state.model_type)
            existing = await _coalescer.start(_coalesce_key)
            if existing is not None:
                await existing.event.wait()
                if existing.error is not None:
                    yield sse_error("generate", "ERROR", str(existing.error), code="E_INFRA_GENERATION", http_status=500)
                    return
                cached = existing.result or ""
                tokens = cached.split()
                for i in range(0, len(tokens), 5):
                    yield sse_token("generate", " ".join(tokens[i:i+5]))
                _mgr.finish(_op_id)
                yield sse_token("generate", "", done=True, meta={"tokens": len(tokens), "cached": True})
                return

            try:
                async for token in provider.chat_stream(
                    provider_messages,
                    max_tokens=req.max_new_tokens,
                    cancel_event=cancel_event,
                    **gen_params,
                ):
                    if cancel_event.is_set() or await request.is_disconnected():
                        cancel_event.set()
                        logger.info("Client disconnected from generate stream", extra={"tag": "INF"})
                        _mgr.finish(_op_id)
                        return
                    if token:
                        _token_gen_start = time.time()
                        token_count += 1
                        _token_count += 1
                        collected.append(token)
                        _batch.append(token)
                        if _token_count == 1 or len(_batch) >= _BATCH_MAX or (time.time() - _batch_start) >= _BATCH_INTERVAL_S:
                            yield sse_token("generate", "".join(_batch))
                            _batch = []
                            _batch_start = time.time()
                    else:
                        now = time.time()
                        if now - _last_heartbeat >= _heartbeat_interval_s:
                            yield ": heartbeat\n\n"
                            _last_heartbeat = now
                    elapsed_since_token = time.time() - _token_gen_start
                    if elapsed_since_token > _max_token_wait_s:
                        logger.warning("Generate stream stalled for %.1fs, aborting", elapsed_since_token, extra={"tag": "INF"})
                        cancel_event.set()
                        _mgr.finish(_op_id, "timeout")
                        yield sse_error("generate", "TIMEOUT", f"Generation stalled for {elapsed_since_token:.0f}s", code="MODEL_TIMEOUT", http_status=504)
                        return
                if _batch:
                    yield sse_token("generate", "".join(_batch))
                    _batch = []
            except Exception as e:
                try:
                    await _coalescer.complete_error(_coalesce_key, e)
                except Exception:
                    pass
                _mgr.finish(_op_id, str(e))
                logger.warning("Generate stream failed: %s", e, extra={"tag": "INF"})
                classify_and_raise(e, source="generate_stream")
            elapsed = (datetime.datetime.now() - start).total_seconds() * 1000
            try:

                get_server_state().record_inference(
                    tokens=token_count, elapsed_ms=elapsed, model=_stream_state.model_type or req.model
                )
            except Exception as e:
                logger.warning("Failed to record inference metrics: %s", e)
            try:

                capture(
                    req.prompt,
                    "".join(collected),
                    model=_stream_state.model_type or req.model,
                    tokens_generated=token_count,
                    elapsed_ms=elapsed,
                    temperature=req.temperature,
                )
            except Exception as e:
                logger.warning("Failed to capture conversation: %s", e)
            full_response = "".join(collected)
            await _coalescer.complete(_coalesce_key, full_response)
            _mgr.finish(_op_id)
            yield sse_token("generate", "", done=True, meta={"tokens": token_count, "elapsed_ms": round(elapsed, 1)})

        return StreamingResponse(generate(), media_type="text/event-stream")

    async def get_info(self) -> dict:
        try:
            """get_info."""
            from host_metrics import sample_host_metrics_async
            import state as server_state

            data = {
                "api_version": "1.0.0",
                "model": {
                    "type": server_state.model_type,
                    "loaded": server_state.model is not None or server_state.provider is not None,
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
            if se is not None and getattr(se, 'is_loaded', False):
                data["soul_engine"] = se.get_stats()

            return data

        except Exception as e:
            classify_and_raise(e, source="inference.get_info")
    async def get_info_soul(self) -> dict:
        try:
            """get_info_soul."""
            import state as server_state
            cs = server_state.current_soul
            if not cs:
                return success_response(data={})
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
            return success_response(data=soul_info)

        except Exception as e:
            classify_and_raise(e, source="inference.get_info_soul")
    async def root(self) -> dict:
        try:
            """root."""
            import state as server_state
            soul_name = None
            if server_state.soul_engine is not None and getattr(server_state.soul_engine, 'slo', None):
                soul_name = server_state.soul_engine.slo.name
            elif server_state.current_soul and hasattr(server_state.current_soul, "name"):
                soul_name = server_state.current_soul.name
            return success_response(data={
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
            })

        except Exception as e:
            classify_and_raise(e, source="inference.root")
    async def list_chat_tools(self) -> dict:
        try:
            """list_chat_tools."""
            try:

                return success_response(data={"tools": get_tool_registry().list_tools()})
            except Exception as e:
                logger.warning("Failed to list tools: %s", e, extra={"tag": "INF"})
                return success_response(data={"tools": []})

        except Exception as e:
            classify_and_raise(e, source="inference.list_chat_tools")
    async def chat_stream(self, req: ChatRequest, request: Request, auth_user: dict = Depends(require_auth_if_enabled)) -> StreamingResponse:
        """chat_stream."""
        corr_id = request.scope.get("correlation_id", "-")
        logger.info(
            "CHAT_STREAM ENTER corr=%s session=%s msgs=%d images=%d max_tokens=%d temp=%.2f",
            corr_id, req.session_id, len(req.messages), len(req.images or []),
            req.max_tokens, req.temperature,
            extra={"tag": "CHAT", "context": {
                "corr": corr_id, "session_id": req.session_id,
                "msg_count": len(req.messages), "has_images": bool(req.images),
                "max_tokens": req.max_tokens, "temperature": req.temperature,
                "use_context": req.use_context_core, "use_rag": req.use_rag,
            }},
        )
        from startup_progress import STARTUP_PHASE

        import state as _check_state
        if STARTUP_PHASE.get("phase") != "ready" or not _model_ready():
            phase = STARTUP_PHASE.get("phase", "unknown")
            if phase == "ready":
                msg = "Model still loading — please wait."
            else:
                msg = f"Server starting (phase: {phase}). Please wait."
            async def error_stream() -> AsyncIterator[str]:
                """error_stream."""
                yield sse_error("chat", "IDLE", msg, code="MODEL_LOADING", http_status=503)
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        mem_err = _check_memory_pressure()
        if mem_err:
            get_server_state().record_memory_pressure_block()
            async def oom_stream() -> AsyncIterator[str]:
                yield sse_error("chat", "IDLE", mem_err, code="MODEL_OOM", http_status=503)
            return StreamingResponse(oom_stream(), media_type="text/event-stream")

        async def generate() -> AsyncIterator[str]:
            """generate."""
            corr_id = request.scope.get("correlation_id", "-")
            logger.debug("chat_stream.generate() ENTERED corr=%s", corr_id)

            # Check for Last-Event-ID header for reconnection
            last_event_id_raw = request.headers.get("last-event-id")
            try:
                last_event_id = int(last_event_id_raw) if last_event_id_raw else 0
            except (ValueError, TypeError):
                last_event_id = 0
            if last_event_id:
                session_id = req.session_id or "default"
                cached = get_chat_response_cache(session_id)
                if cached and cached.get("event_counter", 0) > last_event_id:
                    # Replay cached response from the point of disconnection
                    logger.info("Replaying cached response for session %s from event %s", session_id, last_event_id)
                    cached_counter = last_event_id
                    for token in cached.get("tokens", []):
                        cached_counter += 1
                        yield sse_token("chat", token)
                    if cached.get("complete"):
                        yield sse_token("chat", "", done=True)
                    return

            cancel_event = threading.Event()
            user_msg = _extract_user_message(req.messages)
            if not user_msg:
                yield sse_error("chat", "IDLE", "No user message", code="E_VAL_REQUEST", http_status=400)
                return

            _mgr = get_cancel_manager()
            _op_id = _mgr.register(
                OpType.INFERENCE, f"chat:{user_msg[:40]}",
                cancel_fn=lambda: cancel_event.set(),
            )
            _mgr.start(_op_id)

            start_time = datetime.datetime.now()

            yield _sse_event("chat", "STREAMING", "thinking",
                data={}, message="Thinking...")

            if req.knowledge:
                try:
                    def _store_knowledge(k_list):

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

            logger.debug("CHAT_PIPELINE corr=%s step=SESSION_LOADED session=%s", corr_id, session_id,
                extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "SESSION_LOADED", "session_id": session_id}})

            ctx_core = self._get_context_core()
            context_info = {}
            frame = None
            skip_context = False
            if ctx_core and req.use_context_core:
                try:
                    logger.debug("CHAT_PIPELINE corr=%s step=MEMORY_CHECK start", corr_id,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "MEMORY_CHECK"}})
                    if get_memory_service().stats().get("total_facts", 0) == 0 and not req.knowledge:
                        skip_context = True
                except Exception as e:
                    logger.debug("CHAT_PIPELINE corr=%s step=MEMORY_CHECK error=%s", corr_id, e,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "MEMORY_CHECK", "result": "ERROR", "error": str(e)}})
                    logger.debug("Knowledge memory check failed: %s", e)
            logger.debug("CHAT_PIPELINE corr=%s step=MEMORY_CHECK done skip_context=%s ctx_core=%s", corr_id, skip_context, ctx_core is not None,
                extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "MEMORY_CHECK", "result": "DONE", "skip_context": skip_context, "ctx_core": ctx_core is not None}})
            if ctx_core and req.use_context_core and not skip_context:
                logger.debug("CHAT_PIPELINE corr=%s step=CONTEXTCORE_BUILD start session=%s", corr_id, session_id,
                    extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "CONTEXTCORE_BUILD", "session_id": session_id}})
                ctx_core.set_session_id(session_id)
                ctx_core.add_message("user", user_msg)
                try:
                    frame = await asyncio.wait_for(
                        ctx_core.build_context_frame(
                            include_rag=True,
                            include_memory=True,
                            query=user_msg,
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("CHAT_PIPELINE corr=%s step=CONTEXTCORE_BUILD timeout=5.0s", corr_id,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "CONTEXTCORE_BUILD", "result": "TIMEOUT"}})
                    frame = None
                context_info = {
                    "layers": [l.layer_type for l in frame.layers],
                    "total_tokens": frame.total_tokens,
                    "max_tokens": frame.max_tokens,
                }
                logger.debug("CHAT_PIPELINE corr=%s step=CONTEXTCORE_BUILD done layers=%d tokens=%d", corr_id,
                    len(context_info.get("layers", [])), context_info.get("total_tokens", 0),
                    extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "CONTEXTCORE_BUILD", "result": "DONE", "layers": context_info.get("layers", []), "tokens": context_info.get("total_tokens", 0)}})
                if frame.system_prompt:
                    for i, m in enumerate(provider_messages):
                        if m["role"] == "system":
                            provider_messages[i] = {"role": "system", "content": frame.system_prompt}
                            break
                    else:
                        provider_messages.insert(0, {"role": "system", "content": frame.system_prompt})

            # Production RAG: query for relevant context from ingested documents
            rag_context = ""
            if req.use_rag:
                try:
                    from domains.cognitive.rag_service import is_rag_service_ready
                    if not is_rag_service_ready():
                        logger.debug("RAG service not ready yet, skipping query")
                    else:
                        rag_svc = await asyncio.wait_for(asyncio.to_thread(get_rag_service), timeout=10.0)
                        if rag_svc.stats().get("total_chunks", 0) > 0:
                            rag_result = await asyncio.wait_for(asyncio.to_thread(rag_svc.query, user_msg, 5), timeout=15.0)
                            if rag_result.get("num_results", 0) > 0:
                                rag_context = rag_result["context"]
                                # Inject RAG context into system prompt or as a user context message
                                rag_block = f"[KNOWLEDGE BASE - Retrieved {rag_result['num_results']} relevant passages]\n\n{rag_context}"
                                # Prepend as user context before the conversation
                                provider_messages.insert(0, {"role": "system", "content": rag_block})
                                logger.debug("RAG injected %d passages into chat context", rag_result["num_results"])
                except Exception as e:
                    logger.debug("RAG query skipped: %s", e)

            if req.agent_id:
                try:

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
                logger.debug("CHAT_PIPELINE corr=%s step=TOOL_DETECT start", corr_id,
                    extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "TOOL_DETECT"}})
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
            logger.debug("CHAT_PIPELINE corr=%s step=TOOL_DETECT done", corr_id,
                extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "TOOL_DETECT", "result": "DONE"}})

            if context_info:
                yield _sse_event("chat", "STREAMING", "working",
                    data={"context": context_info},
                    message=f"{len(context_info.get('layers', []))} context layers")

            frame_context = []
            if frame:
                for layer in frame.layers:
                    if layer.layer_type in ("memory", "rag") and layer.content:
                        frame_context.append(layer.content)

            knowledge_retrieved = []
            if not frame_context:
                try:
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_ENRICH start", corr_id,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_ENRICH"}})
                    enrichment = await asyncio.to_thread(_enrich_knowledge, user_msg, False, 5)
                    if enrichment.get("facts"):
                        knowledge_retrieved = enrichment["facts"]
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_ENRICH done facts=%d", corr_id, len(knowledge_retrieved),
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_ENRICH", "result": "DONE", "facts": len(knowledge_retrieved)}})
                except Exception as e:
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_ENRICH error=%s", corr_id, e,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_ENRICH", "result": "ERROR", "error": str(e)}})

            all_knowledge = knowledge_retrieved + frame_context + (req.knowledge or [])
            if all_knowledge:
                try:
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_PROC start count=%d", corr_id, len(all_knowledge),
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_PROC", "count": len(all_knowledge)}})
                    k_proc = KnowledgeProcessor(knowledge=all_knowledge)
                    provider_messages = await apply_processors(provider_messages, [k_proc])
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_PROC done", corr_id,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_PROC", "result": "DONE"}})
                except Exception as e:
                    logger.debug("CHAT_PIPELINE corr=%s step=KNOWLEDGE_PROC error=%s", corr_id, e,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "KNOWLEDGE_PROC", "result": "ERROR", "error": str(e)}})

            try:
                logger.debug("CHAT_PIPELINE corr=%s step=PROVIDER_SETUP start", corr_id,
                    extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "PROVIDER_SETUP"}})

                provider = get_provider("default")

                if provider is not None:
                    full_response_parts: list[str] = []
                    logger.debug("chat_stream: about to call provider.chat_stream()")
                    _token_gen_start = time.time()
                    _max_token_wait_s = cfg.generate_timeout
                    _heartbeat_interval_s = 10.0
                    _last_heartbeat = time.time()
                    _batch: list[str] = []
                    _batch_start = time.time()
                    _BATCH_MAX = 5
                    _BATCH_INTERVAL_S = 0.005
                    _token_count = 0
                    gen_params = _apply_meta_weights(
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                        user_message=user_msg or "",
                        user_id=req.user_id or "default",
                    )

                    import state as _cs_state
                    _coalescer = get_coalescer()
                    _coalesce_key = _coalescer.hash(provider_messages, gen_params, req.max_tokens, _cs_state.model_type)
                    logger.debug("CHAT_PIPELINE corr=%s step=COALESCER_START key=%s", corr_id, _coalesce_key[:16],
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "COALESCER_START", "key": _coalesce_key[:16]}})
                    existing = await _coalescer.start(_coalesce_key)
                    logger.debug("CHAT_PIPELINE corr=%s step=COALESCER_START done existing=%s", corr_id, existing is not None,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "COALESCER_START", "result": "DONE", "existing": existing is not None}})
                    if existing is not None:
                        logger.debug("CHAT_PIPELINE corr=%s step=COALESCER_JOIN existing_key=%s", corr_id, _coalesce_key[:16],
                            extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "COALESCER_JOIN", "key": _coalesce_key[:16]}})
                        await existing.event.wait()
                        logger.debug("CHAT_PIPELINE corr=%s step=COALESCER_JOIN done error=%s", corr_id, existing.error,
                            extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "COALESCER_JOIN", "result": "DONE", "error": str(existing.error) if existing.error else None}})
                        if existing.error is not None:
                            yield sse_error("chat", "ERROR", str(existing.error), code="E_INFRA_GENERATION", http_status=500)
                            return
                        cached = existing.result or ""
                        tokens = cached.split()
                        for i in range(0, len(tokens), 5):
                            yield sse_token("chat", " ".join(tokens[i:i+5]))
                        _mgr.finish(_op_id)
                        yield sse_token("chat", "", done=True)
                        return

                    # Event ID counter for Last-Event-ID reconnection
                    _event_counter = 0
                    _cached_tokens: list[str] = []

                    logger.debug("CHAT_PIPELINE corr=%s step=COALESCER_WAIT done ready_for_provider", corr_id,
                        extra={"tag": "CHAT", "context": {"corr": corr_id, "step": "COALESCER_WAIT", "result": "DONE"}})
                    try:
                        try:
                            _control_check_interval = 0.1  # Check for controls every 100ms
                            _last_control_check = time.time()
                            logger.info(
                                "CHAT_PROVIDER_CALL corr=%s provider=%s session=%s msg_count=%d max_tokens=%d",
                                corr_id, "default", session_id, len(provider_messages), req.max_tokens,
                                extra={"tag": "CHAT", "context": {
                                    "corr": corr_id, "provider": "default",
                                    "session_id": session_id, "msg_count": len(provider_messages),
                                    "max_tokens": req.max_tokens, "gen_params": gen_params,
                                }},
                            )
                            async for token in provider.chat_stream(
                                provider_messages,
                                max_tokens=req.max_tokens,
                                **gen_params,
                                cancel_event=cancel_event,
                                session_id=session_id,
                            ):
                                if await request.is_disconnected():
                                    cancel_event.set()
                                    # Cache partial response for reconnection
                                    set_chat_response_cache(session_id, {
                                        "tokens": _cached_tokens,
                                        "complete": False,
                                        "event_counter": _event_counter,
                                    })
                                    logger.info("Client disconnected from chat stream (request)", extra={"tag": "INF", "context": {"session_id": session_id}})
                                    _mgr.finish(_op_id)
                                    return

                                # Check for control messages periodically
                                now = time.time()
                                if now - _last_control_check >= _control_check_interval:
                                    _last_control_check = now
                                    control = get_chat_control(session_id)
                                    if control:
                                        if control["action"] == "cancel":
                                            cancel_event.set()
                                            _mgr.finish(_op_id)
                                            yield _sse_event("chat", "CONTROL", "cancelled",
                                                data={"action": "cancel"},
                                                message="Stream cancelled by user")
                                            return
                                        elif control["action"] == "approve":
                                            yield _sse_event("chat", "CONTROL", "approved",
                                                data={"tool": control.get("tool_name"), "approved": control.get("approved", True)},
                                                message=f"Tool approval: {control.get('tool_name')}")
                                        elif control["action"] == "context":
                                            yield _sse_event("chat", "CONTROL", "context",
                                                data={"context": control.get("context")},
                                                message="Context injected")

                                if token:
                                    if _token_count == 0:
                                        _first_token_elapsed_ms = (time.time() - _token_gen_start) * 1000
                                        logger.info(
                                            "CHAT_FIRST_TOKEN corr=%s session=%s after=%.1fms",
                                            corr_id, session_id,
                                            _first_token_elapsed_ms,
                                            extra={"tag": "CHAT", "context": {"corr": corr_id, "elapsed_ms": round(_first_token_elapsed_ms, 1)}},
                                        )
                                    _token_gen_start = time.time()
                                    full_response_parts.append(token)
                                    _cached_tokens.append(token)
                                    _batch.append(token)
                                    _token_count += 1
                                    if _token_count == 1 or len(_batch) >= _BATCH_MAX or (time.time() - _batch_start) >= _BATCH_INTERVAL_S:
                                        _event_counter += 1
                                        yield sse_token("chat", "".join(_batch))
                                        _batch = []
                                        _batch_start = time.time()
                                else:
                                    now = time.time()
                                    if now - _last_heartbeat >= _heartbeat_interval_s:
                                        yield ": heartbeat\n\n"
                                        _last_heartbeat = now
                                elapsed_since_token = time.time() - _token_gen_start
                                if elapsed_since_token > _max_token_wait_s:
                                    logger.warning("Token generation stalled for %.1fs, aborting", elapsed_since_token, extra={"tag": "INF"})
                                    cancel_event.set()
                                    _mgr.finish(_op_id, "timeout")
                                    yield sse_error("chat", "TIMEOUT", f"Generation stalled for {elapsed_since_token:.0f}s", code="MODEL_TIMEOUT", http_status=504)
                                    return
                            if _batch:
                                yield sse_token("chat", "".join(_batch))
                                _batch = []
                            yield sse_token("chat", "", done=True)
                            logger.info(
                                "CHAT_STREAM_DONE corr=%s session=%s tokens=%d events=%d",
                                corr_id, session_id, _token_count, _event_counter,
                                extra={"tag": "CHAT", "context": {
                                    "corr": corr_id, "session_id": session_id,
                                    "tokens": _token_count, "events": _event_counter,
                                }},
                            )
                            # Cache complete response for potential reconnection
                            set_chat_response_cache(session_id, {
                                "tokens": _cached_tokens,
                                "complete": True,
                                "event_counter": _event_counter,
                            })
                        except GeneratorExit:
                            cancel_event.set()
                            # Cache partial response on generator exit
                            set_chat_response_cache(session_id, {
                                "tokens": _cached_tokens,
                                "complete": False,
                                "event_counter": _event_counter,
                            })
                            _mgr.finish(_op_id)
                            logger.info("Client disconnected from chat stream", extra={"tag": "INF", "context": {"session_id": session_id}})
                            return
                    except Exception as e:
                        try:
                            await _coalescer.complete_error(_coalesce_key, e)
                        except Exception:
                            pass
                        _mgr.finish(_op_id, str(e))
                        logger.error(
                            "CHAT_STREAM_ERROR corr=%s session=%s error=%s tokens_received=%d",
                            corr_id, session_id, str(e), _token_count,
                            extra={"tag": "CHAT", "context": {
                                "corr": corr_id, "session_id": session_id,
                                "error": str(e), "tokens_received": _token_count,
                                "error_type": type(e).__name__,
                            }},
                        )
                        classify_and_raise(e, source="chat_stream_provider")
                else:
                    yield sse_error("chat", "STREAMING", "No inference provider loaded", code="E_INFRA_REGISTRY", http_status=503)
                    return

                full_response = "".join(full_response_parts)
                await _coalescer.complete(_coalesce_key, full_response)

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

                _memory_stored = False
                _memory_fact = None
                try:

                    _memory_facts = await get_memory_service().remember_facts_async(user_msg or "", full_response)
                    _memory_stored = len(_memory_facts) > 0
                    if _memory_stored:
                        _memory_fact = _memory_facts[0]
                except Exception as e:
                    logger.debug("Auto-memory remember failed: %s", e)

                if _memory_stored:
                    yield _sse_event(
                        "chat", "MEMORY", "success",
                        data={"stored": True, "fact": _memory_fact, "facts": _memory_facts},
                        message="New fact remembered",
                    )

                duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
                tokens = len(full_response.split())
                _post_gen_tasks = []

                # Production RAG: verify response against knowledge base (parallel)
                rag_verification = None
                if req.use_rag and full_response.strip():
                    try:
                        rag_svc = get_rag_service()
                        if rag_svc.stats().get("total_chunks", 0) > 0:
                            _post_gen_tasks.append(asyncio.to_thread(
                                rag_svc.verify_and_ground, full_response, user_msg or "",
                            ))
                    except Exception as e:
                        logger.debug("RAG verification skipped: %s", e)

                try:
                    _post_gen_tasks.append(asyncio.to_thread(
                        capture,
                        user_msg or "",
                        full_response,
                        model=_check_state.model_type or req.model,
                        tokens_generated=tokens,
                        elapsed_ms=duration_ms,
                        temperature=req.temperature,
                        meta={"session_id": session_id},
                    ))
                except Exception as e:
                    logger.warning("Failed to capture conversation: %s", e)

                try:

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
                    logger.warning("ResponseTracker.log failed: %s", e)

                try:
                    get_server_state().record_inference(
                        tokens=tokens, elapsed_ms=duration_ms, model=_check_state.model_type or req.model,
                    )
                except Exception as e:
                    logger.warning("Failed to record inference metrics: %s", e)

                if ctx_core and req.use_context_core:
                    try:
                        ctx_core.add_response(full_response, model=req.model)
                    except Exception as e:
                        logger.warning("ContextCore.add_response failed: %s", e)

                try:

                    _post_gen_tasks.append(asyncio.to_thread(
                        get_learner().ingest_conversation, [(user_msg, full_response)]
                    ))
                except Exception as e:
                    logger.warning("Continual learner ingest failed: %s", e)

                if _post_gen_tasks:
                    results = await asyncio.gather(*_post_gen_tasks, return_exceptions=True)
                    # Extract RAG verification result if it was in the tasks
                    if req.use_rag and full_response.strip():
                        for r in results:
                            if isinstance(r, dict) and "confidence" in r:
                                rag_verification = r
                                break

                # Send RAG verification results as a separate SSE event
                if rag_verification is not None:
                    yield _sse_event(
                        "chat", "RAG_VERIFICATION", "success",
                        data={
                            "confidence": rag_verification.get("confidence", 0),
                            "is_verified": rag_verification.get("is_verified", False),
                            "hallucination_rate": rag_verification.get("verification", {}).get("hallucination_rate", 0),
                            "citations": rag_verification.get("citations", ""),
                            "grounded_claims": len(rag_verification.get("verification", {}).get("grounded_claims", [])),
                            "hallucinated_claims": len(rag_verification.get("verification", {}).get("hallucinations", [])),
                        },
                        message="RAG grounding verification complete",
                    )

                try:

                    task = asyncio.create_task(extract_and_store(user_msg or "", full_response))
                    with self._bg_tasks_lock:
                        self._BG_TASKS.add(task)
                    task.add_done_callback(self._bg_tasks_lock_discard)
                except Exception as e:
                    logger.warning("Entity extraction failed: %s", e)

                logger.info("Chat stream: generated %d chars", len(full_response), extra={"tag": "INF", "context": {"char_count": len(full_response), "session_id": session_id}})
                _mgr.finish(_op_id)

            except Exception as e:
                _mgr.finish(_op_id, str(e))
                logger.warning("Chat stream outer failed: %s", e, extra={"tag": "INF"})
                classify_and_raise(e, source="chat_stream_outer")

        return StreamingResponse(generate(), media_type="text/event-stream")

    async def inspect_context(self) -> dict:
        try:
            """inspect_context."""
            ctx_core = self._get_context_core()
            if not ctx_core:
                raise_error("ContextCore not available", "E_INFRA_STARTUP")
            return ctx_core.get_context_inspector()

        except Exception as e:
            classify_and_raise(e, source="inference.inspect_context")
    async def store_fact(self, key: str, value: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """store_fact."""
            ctx_core = self._get_context_core()
            if not ctx_core:
                raise_error("ContextCore not available", "E_INFRA_STARTUP")
            ctx_core.store_fact(key, value)
            return success_response(data={"stored": key})

        except Exception as e:
            classify_and_raise(e, source="inference.store_fact")
    async def get_facts(self, query: str = "") -> dict:
        try:
            """get_facts."""
            ctx_core = self._get_context_core()
            if not ctx_core:
                raise_error("ContextCore not available", "E_INFRA_STARTUP")
            if query:
                return success_response(data={"facts": ctx_core.search_semantic(query)})
            return success_response(data={"facts": [{"key": k, **v} for k, v in ctx_core.semantic_memory.items()]})

        except Exception as e:
            classify_and_raise(e, source="inference.get_facts")
    async def reset_context(self, all: bool = False, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """reset_context."""
            self._context_core = None
            ctx_core = self._get_context_core()
            if not ctx_core:
                raise_error("ContextCore not available", "E_INFRA_STARTUP")
            if all:
                ctx_core.reset_all()
                safe_audit_log("inference.reset_context", resource="context", detail="scope=all")
            else:
                ctx_core.reset_session()
                safe_audit_log("inference.reset_context", resource="context", detail="scope=session")
            return success_response(data={"reset": "session" if not all else "all"})

        except Exception as e:
            classify_and_raise(e, source="inference.reset_context")
    async def chat(self, req: ChatRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> ChatResponse:
        """chat."""
        _chat_t0 = time.monotonic()
        from domains import get_chat_domain
        from startup_progress import STARTUP_PHASE
        import state as _chat_state

        if STARTUP_PHASE.get("phase") != "ready" or not _model_ready():
            raise_error("Model still loading — please wait.", "E_BAD_REQUEST", status_code=503)

        mem_err = _check_memory_pressure()
        if mem_err:
            raise_error(mem_err, "E_RESOURCE", status_code=503)

        try:

            _router = get_provider("default")
            _server = getattr(_router, '_server', None)
            if _server is not None:
                _cb = getattr(_server, '_circuit_breaker', None)
                if _cb is not None and _cb.state.value == "open":
                    raise_error("Model is degraded — circuit breaker open. Please wait or reload the model.", "E_BAD_REQUEST", status_code=503)
        except AppError as e:
            classify_and_raise(e, source="inference.chat")
        except Exception as e:
            logger.debug("Circuit breaker check failed: %s", e)

        user_msg = _extract_user_message(req.messages)
        if not user_msg:
            raise_error("No user message", "E_BAD_REQUEST", status_code=400)

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

        try:
            gen_params = _apply_meta_weights(
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                user_message=user_msg,
                user_id=req.user_id or "default",
            )
            result = await asyncio.wait_for(
                chat_domain.respond(
                    messages=messages,
                    model=_chat_state.model_type or req.model,
                    system_prompt=system_prompt,
                    temperature=gen_params["temperature"],
                    max_tokens=req.max_tokens,
                    session_id=req.session_id or "default",
                    user_id=req.user_id or "default",
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                message="Generation timed out. Try a shorter prompt or fewer tokens.",
                session_id=req.session_id or "default",
                done=True,
            )

        try:

            tokens = _count_tokens(result.text, _chat_state)
            _chat_elapsed_ms = round((time.monotonic() - _chat_t0) * 1000, 1)
            get_server_state().record_inference(
                tokens=tokens, elapsed_ms=_chat_elapsed_ms, model=_chat_state.model_type or req.model
            )
        except Exception as e:
            logger.warning("Failed to record inference metrics: %s", e)

        try:

            capture(
                user_msg or "",
                result.text,
                model=_chat_state.model_type or req.model,
                tokens_generated=_count_tokens(result.text, _chat_state),
                temperature=req.temperature,
                meta={"session_id": req.session_id or "default"},
            )
        except Exception as e:
            logger.warning("Failed to capture conversation: %s", e)

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
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """send_voice_message."""
        self._ensure_dirs()
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise_error("Only audio files accepted", "E_BAD_REQUEST", status_code=400)

        msg_id = str(uuid.uuid4())
        session_msg_dir = (self._VOICE_DIR / session_id).resolve()
        if not str(session_msg_dir).startswith(str(self._VOICE_DIR.resolve())):
            raise_error("Invalid session ID", "E_BAD_REQUEST", status_code=400)
        await asyncio.to_thread(session_msg_dir.mkdir, parents=True, exist_ok=True)
        ext = Path(file.filename or "audio.m4a").suffix or ".m4a"
        audio_path = session_msg_dir / f"{msg_id}{ext}"

        content = await file.read()
        await asyncio.to_thread(audio_path.write_bytes, content)

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

    async def get_voice_audio(self, session_id: str, message_id: str) -> dict:
        try:
            """get_voice_audio."""
            self._ensure_dirs()
            base = self._VOICE_DIR.resolve()
            audio_path = (self._VOICE_DIR / session_id / message_id).resolve()
            if not str(audio_path).startswith(str(base)):
                raise_error("Invalid path", "E_AUTH_FORBIDDEN", status_code=403)

            def _resolve():
                if audio_path.exists():
                    return audio_path
                for ext in [".m4a", ".wav", ".mp3", ".ogg", ".webm"]:
                    candidate = audio_path.parent / f"{audio_path.stem}{ext}"
                    if candidate.exists():
                        return candidate
                return None

            resolved = await asyncio.to_thread(_resolve)
            if resolved is None:
                raise_error("Audio not found", "E_NOT_FOUND", status_code=404)
            return FileResponse(str(resolved), media_type="audio/m4a")

        except Exception as e:
            classify_and_raise(e, source="inference.get_voice_audio")
    async def list_sessions(self, archived: Optional[bool] = None) -> dict:
        try:
            """list_sessions."""
            sessions = await asyncio.to_thread(self._build_session_metadata_index)
            if archived is not None:
                sessions = [s for s in sessions if s.get("archived", False) == archived]
            return success_response(data=sessions)

        except Exception as e:
            classify_and_raise(e, source="inference.list_sessions")
    async def search_sessions(self, q: str = "", limit: int = 20) -> dict:
        try:
            """search_sessions."""
            if not q.strip():
                return success_response(data=[], meta={"query": q, "total": 0})
            results = await asyncio.to_thread(_search_sessions_sync, q, limit)
            return success_response(data=results, meta={"query": q, "total": len(results)})

        except Exception as e:
            classify_and_raise(e, source="inference.search_sessions")
    async def get_current_session(self) -> dict:
        try:
            """get_current_session."""
            sessions = await asyncio.to_thread(self._build_session_cache)
            if not sessions:
                return success_response(data=None)
            return success_response(data=sessions[0])

        except Exception as e:
            classify_and_raise(e, source="inference.get_current_session")
    async def upsert_session(self, session_id: str, req: UpsertSessionRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """upsert_session."""
            existing = self._get_session(session_id)
            update_data = req.model_dump(exclude_none=True)
            for key, value in update_data.items():
                existing[key] = value
            self._save_session(session_id, existing)
            await self._flush_session_to_disk(session_id)
            return success_response(data={"session_id": session_id}, message="saved")

        except Exception as e:
            classify_and_raise(e, source="inference.upsert_session")
    async def create_session(self, req: CreateSessionRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """create_session."""
        try:
            session_id = req.session_id or str(uuid.uuid4())
            session_data = req.model_dump(exclude_none=True)
            session_data["session_id"] = session_id
            self._save_session(session_id, session_data)
            await self._flush_session_to_disk(session_id)
            safe_audit_log("inference.session_create", resource=session_id)
            return success_response(data={"session_id": session_id}, message="created")
        except Exception as exc:
            logger.warning("Create session failed: %s", exc, extra={"tag": "INF"})
            classify_and_raise(exc, source="create_session")

    async def get_session(self, session_id: str) -> dict:
        try:
            """get_session."""
            data = self._get_session(session_id)
            if not data.get("messages"):
                raise_error("Session not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=data)

        except Exception as e:
            classify_and_raise(e, source="inference.get_session")
    async def delete_session(self, session_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """delete_session."""
            if self._session_repo.delete(session_id):
                self._session_memory_cache.pop(session_id, None)
                self._session_dirty.discard(session_id)
                self._session_deleted.add(session_id)
                if len(self._session_deleted) > 1000:
                    self._session_deleted = set(list(self._session_deleted)[-500:])
                self._clear_session_kv(session_id)
                safe_audit_log("inference.session_delete", resource=session_id)
                return success_response(data={"session_id": session_id}, message="deleted")
            raise_error("Session not found", "E_NOT_FOUND", status_code=404)

        except Exception as e:
            classify_and_raise(e, source="inference.delete_session")
    def _clear_session_kv(self, session_id: str):
        """Drop cross-turn KV state for a deleted session.

        The KV cache is keyed by session_id inside the SloNet provider; a
        deleted session's cached keys/values would otherwise persist until
        TTL eviction and risk stale-context reuse if the id is recycled.
        """
        try:

            provider = get_provider("slonet-native")
            if provider is None:
                provider = get_provider("slonet")
            if provider is not None and hasattr(provider, "clear_session"):
                provider.clear_session(session_id)
        except Exception as exc:
            logger.warning("Failed to clear KV state for session %s: %s",
                           session_id, exc, extra={"tag": "KV"})

    async def chat_suggestions(self) -> dict:
        try:
            """chat_suggestions."""
            return success_response(data=[
                {"text": "What can you help me with?", "icon": "chat"},
                {"text": "Tell me about yourself", "icon": "user"},
                {"text": "Write a short poem", "icon": "pen"},
                {"text": "Explain quantum computing simply", "icon": "atom"},
                {"text": "Help me debug my code", "icon": "bug"},
                {"text": "Summarize a topic for me", "icon": "document"},
            ])

        except Exception as e:
            classify_and_raise(e, source="inference.chat_suggestions")
    async def list_model_providers(self) -> dict:
        try:
            """list_model_providers."""
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

        # ── Operations (CancelManager) ──

        except Exception as e:
            classify_and_raise(e, source="inference.list_model_providers")
    async def list_operations(self, type: Optional[str] = None) -> dict:
        try:
            """List all tracked operations (active + recently finished).

            Args:
                type: Optional filter by operation type (training, inference, download, etc.)

            Returns:
                Dict with 'operations' list and 'counts' by status.
            """

            mgr = get_cancel_manager()
            op_type = OpType(type) if type else None
            all_ops = mgr.list_all(op_type=op_type)
            return success_response(data={
                "operations": [op.to_dict() for op in all_ops],
                "counts": mgr.count(op_type=op_type),
            })

        except Exception as e:
            classify_and_raise(e, source="inference.list_operations")
    async def cancel_operation(self, op_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Cancel a single operation by ID.

            Args:
                op_id: The operation ID to cancel.

            Returns:
                Dict with cancel result.
            """

            mgr = get_cancel_manager()
            found = mgr.get(op_id)
            if not found:
                raise_error("Operation not found", "E_NOT_FOUND", status_code=404)
            if mgr.cancel(op_id):
                safe_audit_log("inference.operation_cancel", resource=op_id)
                return success_response(data=found.to_dict(), message="cancelled")
            raise_error(
                f"Cannot cancel operation in '{found.status.value}' state",
                "E_CANCEL_FAILED",
                status_code=409,
            )

        except Exception as e:
            classify_and_raise(e, source="inference.cancel_operation")
    async def cancel_all_operations(self, type: Optional[str] = None, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Cancel all active operations, optionally filtered by type.

            Args:
                type: Optional filter by operation type.

            Returns:
                Dict with list of cancelled operation IDs.
            """

            mgr = get_cancel_manager()
            op_type = OpType(type) if type else None
            cancelled = mgr.cancel_all(op_type=op_type)
            safe_audit_log("inference.operation_cancel_all", detail=f"count={len(cancelled)} type={type or 'all'}")
            return success_response(data={"cancelled": cancelled, "count": len(cancelled)})

        except Exception as e:
            classify_and_raise(e, source="inference.cancel_all_operations")
    async def purge_operations(self, max_age_s: float = 3600.0, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Remove finished operations older than max_age_s."""
            import time as _time
            _t0 = _time.monotonic()
            removed = get_cancel_manager().purge(max_age_s=max_age_s)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("inference.purge_operations", detail=f"removed={removed} max_age={max_age_s}s elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={"purged": removed, "elapsed_ms": round(_elapsed_ms, 1)})

        # ── Route registration ──

        except Exception as e:
            classify_and_raise(e, source="inference.purge_operations")

    async def chat_control(self, req: ChatControlRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """chat_control - Send control messages to active chat streams.

        Actions:
        - cancel: Cancel the active stream for this session
        - approve: Approve/deny a tool execution
        - context: Inject additional context into the stream
        """
        try:
            session_id = req.session_id

            if req.action == "cancel":
                # Find and cancel active inference operations for this session
                mgr = get_cancel_manager()
                active = mgr.list_active(op_type=OpType.INFERENCE)
                cancelled = False
                for op in active:
                    if session_id in op.get("label", ""):
                        mgr.cancel(op["id"])
                        cancelled = True
                        logger.info("Cancelled chat stream for session %s", session_id, extra={"tag": "INF"})
                        break
                if not cancelled:
                    logger.debug("No active stream found for session %s", session_id)
                return success_response(data={"cancelled": cancelled, "session_id": session_id})

            elif req.action == "approve":
                # Store approval for the streaming endpoint to pick up
                set_chat_control(session_id, {
                    "action": "approve",
                    "tool_name": req.tool_name,
                    "approved": req.approved if req.approved is not None else True,
                })
                logger.info("Stored tool approval for session %s: %s=%s", session_id, req.tool_name, req.approved)
                return success_response(data={"stored": True, "session_id": session_id})

            elif req.action == "context":
                # Store context injection for the streaming endpoint to pick up
                set_chat_control(session_id, {
                    "action": "context",
                    "context": req.context,
                })
                logger.info("Stored context injection for session %s", session_id)
                return success_response(data={"stored": True, "session_id": session_id})

            raise_error(f"Unknown action: {req.action}", code="E_BAD_REQUEST", status_code=400)

        except Exception as e:
            classify_and_raise(e, source="inference.chat_control")
    def _register_routes(self):
        r = self.router
        r.add_api_route("/inference/generate", self.generate, methods=["POST"], response_model=GenerateResponse)
        r.add_api_route("/inference/generate/stream", self.generate_stream, methods=["POST"])
        r.add_api_route("/info", self.get_info, methods=["GET"])
        r.add_api_route("/info/soul", self.get_info_soul, methods=["GET"])
        r.add_api_route("/", self.root, methods=["GET"])
        r.add_api_route("/chat/tools", self.list_chat_tools, methods=["GET"])
        r.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        r.add_api_route("/chat/control", self.chat_control, methods=["POST"])
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
        r.add_api_route("/chat/suggestions", self.chat_suggestions, methods=["GET"])
        r.add_api_route("/providers", self.list_model_providers, methods=["GET"])
        r.add_api_route("/operations", self.list_operations, methods=["GET"])
        r.add_api_route("/cancel/{op_id}", self.cancel_operation, methods=["POST"])
        r.add_api_route("/cancel-all", self.cancel_all_operations, methods=["POST"])
        r.add_api_route("/operations/purge", self.purge_operations, methods=["POST"])

_instance = InferenceRouter()
router = _instance.router

def set_vector_store_ref(ref) -> dict:
    """set_vector_store_ref."""
    return _instance.set_vector_store_ref(ref)

def flush_dirty_sessions() -> dict:
    """flush_dirty_sessions."""
    return _instance.flush_dirty_sessions()
