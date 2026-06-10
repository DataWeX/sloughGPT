"""
Session Router - Chat session management.
Delegates to message_feedback in main.py for storage (in-process singleton).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/session", tags=["session"])

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
        return result
    return {"status": "stored", "session_id": session_id, "message_count": 0}


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
        return {"status": "ok", "session_id": session_id, "messages": msgs}
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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
