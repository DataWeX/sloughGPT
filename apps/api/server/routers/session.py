"""
Session Router - Chat session management.
Delegates to message_feedback in main.py for storage (in-process singleton).
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncIterator
import json

from schemas.common import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/session", tags=["session"])

try:
    from domains.api.sse_envelope import sse_event as _sse_event, sse_token, sse_error
except ImportError:
    def _sse_event(stream, phase, status, data=None, meta=None, message=""):
        return "data: " + json.dumps({
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

class SessionContext(BaseModel):
    system_prompt: Optional[str] = None
    knowledge: Optional[List[str]] = None
    messages: Optional[List[Dict[str, Any]]] = []

@router.post("/{session_id}/context")
async def set_session_context(session_id: str, ctx: SessionContext):
    """Set session context (messages stored for regeneration)."""
    if ctx.messages:
        from domains.infrastructure.session_core import SessionCore
        result = SessionCore.store_context(session_id, ctx.messages)
        return success_response(data=result)
    return success_response(data={"session_id": session_id, "message_count": 0}, message="stored")


def get_router_session_context(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """Legacy helper – forwards to ``SessionCore``.

    Existing code (e.g., ``main.py``) still calls this function. It now
    proxies to the unified ``SessionCore`` implementation.
    """
    from domains.infrastructure.session_core import SessionCore
    return SessionCore.get_messages(session_id)


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Return stored conversation messages for a session.

    Used by the UI to load a chat history.
    """
    from domains.infrastructure.session_core import SessionCore
    try:
        msgs = SessionCore.get_messages(session_id)
        return success_response(data={"session_id": session_id, "messages": msgs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/inspector")
async def get_session_inspector(session_id: str):
    """Return aggregated context state for the UI context inspector.

    Combines session context, knowledge stats, trait weights, manager
    modes, and workspace memory into a single response.
    """
    from domains.infrastructure.session_core import SessionCore
    from domains.feedback.message_feedback import get_message_feedback
    try:
        msgs = SessionCore.get_messages(session_id)
        fb = get_message_feedback()
        fb_stats = fb.get_stats()

        knowledge = {"total_facts": 0, "topics": []}
        try:
            from domains.learner.knowledge import get_knowledge_memory
            km = get_knowledge_memory()
            knowledge["total_facts"] = km.stats().get("total_facts", 0)
            knowledge["topics"] = [t[0] for t in km.all_topics()[:10]]
        except Exception as e:
            logger.debug("Knowledge memory unavailable in session detail: %s", e)

        traits = {}
        modes = {}
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

        return {
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
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/regenerate")
async def regenerate_session(session_id: str, request: Request) -> StreamingResponse:
    """Regenerate the last assistant response for a session.

    Loads stored session messages, calls the provider to regenerate
    the next assistant response, and streams the result via SSE.
    """
    async def generate() -> AsyncIterator[str]:
        try:
            from domains.infrastructure.session_core import SessionCore
            msgs = SessionCore.get_messages(session_id)
            if not msgs:
                yield sse_error("chat", "REGENERATE", "No session context found")
                return

            from domains.models.provider import get_provider
            provider = get_provider("default")
            if provider is None:
                yield sse_error("chat", "REGENERATE", "Model not loaded")
                return

            yield _sse_event("chat", "REGENERATE", "thinking",
                             data={}, message="Regenerating...")

            full_response = ""
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
                        full_response += token
                        yield sse_token("chat", token)
                yield sse_token("chat", "", done=True)
            except GeneratorExit:
                return
            except Exception as e:
                logger.error("Regenerate stream error: %s", e, exc_info=True, extra={"tag": "REQ"})
                yield sse_error("chat", "REGENERATE", f"Generation failed: {e}")
                return

        except Exception as e:
            logger.error("Regenerate error: %s", e, exc_info=True, extra={"tag": "REQ"})
            yield sse_error("chat", "REGENERATE", str(e))

    return StreamingResponse(generate(), media_type="text/event-stream")
