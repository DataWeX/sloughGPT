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
