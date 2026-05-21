"""
Security Router - Audit logs and API key management
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/audit")
async def get_audit_logs(limit: int = 100, event_type: Optional[str] = None):
    """Get audit logs"""
    from main import audit_logger
    logs = audit_logger.logs[-limit:]
    if event_type:
        logs = [l for l in logs if l.get("event_type") == event_type]
    return {"logs": logs, "count": len(logs)}


@router.get("/keys")
async def get_keys():
    """Get API key info (not the keys themselves)"""
    from main import VALID_API_KEYS
    return {
        "count": len(VALID_API_KEYS),
        "configured": len(VALID_API_KEYS) > 0,
    }