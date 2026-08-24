"""
Session Router - Chat session management.
Delegates to message_feedback in main.py for storage (in-process singleton).
"""
import asyncio
import json
import logging
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncIterator

from schemas.common import success_response, classify_and_raise, safe_audit_log
from config import ServerConfig

logger = logging.getLogger(__name__)

cfg = ServerConfig.from_env()


class SessionContext(BaseModel):
    system_prompt: Optional[str] = None
    knowledge: Optional[List[str]] = None
    messages: Optional[List[Dict[str, Any]]] = []


class SessionRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/session", tags=["session"])
        self._init_sse_helpers()
        self._register_routes()

    def _init_sse_helpers(self):
        try:
            from domains.api.sse_envelope import sse_event as _sse_event, sse_token, sse_error
            self._sse_event = _sse_event
            self._sse_token = sse_token
            self._sse_error = sse_error
        except ImportError:
            def _sse_event(stream, phase, status, data=None, meta=None, message=""):
                return "data: " + json.dumps({
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
            def sse_error(stream, phase, error, meta=None) -> dict:
                """sse_error."""
                return _sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
            self._sse_event = _sse_event
            self._sse_token = sse_token
            self._sse_error = sse_error

    def _register_routes(self):
        self.router.add_api_route("/{session_id}/context", self.set_session_context, methods=["POST"])
        self.router.add_api_route("/{session_id}/messages", self.get_session_messages, methods=["GET"])
        self.router.add_api_route("/{session_id}/inspector", self.get_session_inspector, methods=["GET"])
        self.router.add_api_route("/{session_id}/regenerate", self.regenerate_session, methods=["POST"])

    @staticmethod
    def get_router_session_context(session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Legacy helper – forwards to ``SessionCore``.

        Existing code (e.g., ``main.py``) still calls this function. It now
        proxies to the unified ``SessionCore`` implementation.
        """
        from domains.infrastructure.session_core import SessionCore
        return SessionCore.get_messages(session_id)

    async def set_session_context(self, session_id: str, ctx: SessionContext) -> dict:
        """Set session context (messages stored for regeneration)."""
        if ctx.messages:
            from domains.infrastructure.session_core import SessionCore
            result = SessionCore.store_context(session_id, ctx.messages)
            safe_audit_log("session.context_store", resource=session_id, detail=f"messages={len(ctx.messages)}")
            return success_response(data=result)
        return success_response(data={"session_id": session_id, "message_count": 0}, message="stored")

    async def get_session_messages(self, session_id: str) -> dict:
        """Return stored conversation messages for a session.

        Used by the UI to load a chat history.
        """
        from domains.infrastructure.session_core import SessionCore
        try:
            msgs = SessionCore.get_messages(session_id)
            return success_response(data={"session_id": session_id, "messages": msgs})
        except Exception as e:
            logger.warning("Get session messages failed: %s", e)
            classify_and_raise(e, source="session_get_messages")

    async def get_session_inspector(self, session_id: str) -> dict:
        """Return aggregated context state for the UI context inspector.

        Combines session context, knowledge stats, trait weights, manager
        modes, and workspace memory into a single response. Fetches all
        sources concurrently for low latency.
        """
        _inspector_start = time.time()
        try:
            def _fetch_messages():
                from domains.infrastructure.session_core import SessionCore
                return SessionCore.get_messages(session_id)

            def _fetch_feedback():
                from domains.feedback.message_feedback import get_message_feedback
                fb = get_message_feedback()
                return fb.get_stats()

            def _fetch_knowledge():
                knowledge = {"total_facts": 0, "topics": []}
                try:
                    from domains.learner.knowledge import get_knowledge_memory
                    km = get_knowledge_memory()
                    knowledge["total_facts"] = km.stats().get("total_facts", 0)
                    knowledge["topics"] = [t[0] for t in km.all_topics()[:10]]
                except Exception as e:
                    logger.debug("Knowledge memory unavailable in session detail: %s", e)
                return knowledge

            def _fetch_traits():
                traits, modes = {}, {}
                try:
                    from domains.context.managers import (get_trait_config, PersonalityManager,
                                                          MemoryManager, StyleManager, TaskManager)
                    config = get_trait_config()
                    traits = config.all()
                    modes = {
                        "personality": PersonalityManager(config).get_mode(),
                        "memory": MemoryManager(config).get_mode(),
                        "style": StyleManager(config).get_mode(),
                        "task": TaskManager(config).get_mode(),
                    }
                except Exception as e:
                    logger.debug("Trait config unavailable in session detail: %s", e)
                return traits, modes

            def _fetch_workspace():
                workspace = {"working_memory": [], "semantic_keys": [], "episodic_count": 0}
                try:
                    from domains.infrastructure.context_core import get_context_core
                    cc = get_context_core()
                    insp = cc.get_context_inspector()
                    workspace = {
                        "working_memory": insp.get("working_memory", []),
                        "semantic_keys": insp.get("semantic_keys", []),
                        "episodic_count": insp.get("episodic_count", 0),
                        "sensory_buffer_size": insp.get("sensory_buffer_size", 0),
                        "system_prompt": insp.get("system_prompt", "")[:300],
                    }
                except Exception as e:
                    logger.debug("Context core unavailable in session detail: %s", e)
                return workspace

            msgs, fb_stats, knowledge, (traits, modes), workspace = await asyncio.gather(
                asyncio.to_thread(_fetch_messages),
                asyncio.to_thread(_fetch_feedback),
                asyncio.to_thread(_fetch_knowledge),
                asyncio.to_thread(_fetch_traits),
                asyncio.to_thread(_fetch_workspace),
            )
            _elapsed_ms = round((time.time() - _inspector_start) * 1000)

            return success_response(data={
                "session": {
                    "id": session_id,
                    "message_count": len(msgs),
                    "messages": msgs[-10:],
                },
                "knowledge": knowledge,
                "traits": traits,
                "modes": modes,
                "feedback": {
                    "total": fb_stats.get("feedback_total", 0),
                    "thumbs_up": fb_stats.get("thumbs_up", 0),
                    "thumbs_down": fb_stats.get("thumbs_down", 0),
                },
                "workspace": workspace,
                "elapsed_ms": _elapsed_ms,
            })

        except Exception as e:
            logger.warning("Session inspector failed: %s", e)
            classify_and_raise(e, source="session_inspector")

    async def regenerate_session(self, session_id: str, request: Request) -> StreamingResponse:
        """Regenerate the last assistant response for a session."""
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _op_id = None
        try:
            mgr = get_cancel_manager()
            _op_id = mgr.register(OpType.INFERENCE, f"regenerate:{session_id}")
            mgr.start(_op_id)
        except Exception as exc:
            logger.debug("CancelManager registration failed for regenerate: %s", exc)

        async def generate() -> AsyncIterator[str]:
            """generate."""
            _regen_start = time.time()
            try:
                from domains.infrastructure.session_core import SessionCore
                msgs = SessionCore.get_messages(session_id)
                if not msgs:
                    yield self._sse_error("chat", "REGENERATE", "No session context found")
                    return

                from domains.models.provider import get_provider
                provider = get_provider("default")
                if provider is None:
                    yield self._sse_error("chat", "REGENERATE", "Model not loaded")
                    return

                yield self._sse_event("chat", "REGENERATE", "thinking",
                                      data={}, message="Regenerating...")

                full_response = ""
                _token_gen_start = time.time()
                _max_token_wait_s = cfg.generate_timeout
                _heartbeat_interval_s = 10.0
                _last_heartbeat = time.time()
                try:
                    async for token in provider.chat_stream(
                        msgs,
                        max_tokens=512,
                        temperature=0.8,
                        session_id=session_id,
                    ):
                        if await request.is_disconnected():
                            logger.info("Client disconnected from regenerate stream (request)", extra={"tag": "REQ", "context": {"session_id": session_id}})
                            return
                        if token:
                            _token_gen_start = time.time()
                            full_response += token
                            yield self._sse_token("chat", token)
                        else:
                            now = time.time()
                            if now - _last_heartbeat >= _heartbeat_interval_s:
                                yield ": heartbeat\n\n"
                                _last_heartbeat = now
                        elapsed_since_token = time.time() - _token_gen_start
                        if elapsed_since_token > _max_token_wait_s:
                            logger.warning("Regenerate stream stalled for %.1fs, aborting", elapsed_since_token, extra={"tag": "REQ"})
                            yield self._sse_error("chat", "TIMEOUT", f"Generation stalled for {elapsed_since_token:.0f}s")
                            return
                    yield self._sse_token("chat", "", done=True)
                    _regen_elapsed_ms = round((time.time() - _regen_start) * 1000)
                    safe_audit_log("session.regenerate", resource=session_id, detail=f"chars={len(full_response)} elapsed={_regen_elapsed_ms}ms")
                    logger.info("Regenerate completed: %d chars in %dms", len(full_response), _regen_elapsed_ms, extra={"tag": "REQ"})
                except GeneratorExit:
                    return
                except Exception as e:
                    logger.error("Regenerate stream error: %s", e, exc_info=True, extra={"tag": "REQ"})
                    yield self._sse_error("chat", "REGENERATE", f"Generation failed: {e}")
                    return

            except Exception as e:
                logger.error("Regenerate error: %s", e, exc_info=True, extra={"tag": "REQ"})
                yield self._sse_error("chat", "REGENERATE", str(e))
            finally:
                if _op_id:
                    try:
                        get_cancel_manager().finish(_op_id)
                    except Exception as exc:
                        logger.debug("CancelManager.finish failed for regenerate: %s", exc)

        return StreamingResponse(generate(), media_type="text/event-stream")


router = SessionRouter().router
