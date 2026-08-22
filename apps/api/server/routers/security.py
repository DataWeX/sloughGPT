"""
Security Router - Audit logs and API key management
"""
import asyncio
from fastapi import APIRouter, Query
from typing import Optional

from schemas.common import success_response


class SecurityRouter:
    """Security Router - Audit logs and API key management."""

    def __init__(self):
        self.router = APIRouter(prefix="/security", tags=["security"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/audit", endpoint=self.get_audit_logs, methods=["GET"])
        self.router.add_api_route(path="/keys", endpoint=self.get_keys, methods=["GET"])

    async def get_audit_logs(
        self,
        limit: int = Query(default=100, ge=1, le=10000, description="Maximum number of log entries to return"),
        event_type: Optional[str] = Query(default=None, description="Filter by event type"),
        history: bool = Query(default=False, description="Read from persisted audit.log file"),
        before: Optional[str] = Query(default=None, description="ISO-8601 cursor for pagination"),
    ) -> dict:
        """Get audit logs.

        With ``history=true`` reads the persisted ``audit.log`` file (full trail
        across restarts, optional ``before`` ISO-8601 cursor for pagination);
        otherwise returns the in-memory session buffer.
        """
        from infrastructure.auth import get_audit_logger
        audit_logger = get_audit_logger()
        if history:
            logs = await asyncio.to_thread(audit_logger.file_query, limit=limit, event_type=event_type, before=before)
        else:
            logs = audit_logger.logs[-limit:]
            if event_type:
                logs = [l for l in logs if l.get("event_type") == event_type]
        return success_response(data={"logs": logs, "count": len(logs)})

    async def get_keys(self) -> dict:
        """Get API key info (not the keys themselves)"""
        from settings import get_security_settings
        sec = get_security_settings()
        return success_response(data={
            "count": len(sec.valid_api_keys),
            "configured": len(sec.valid_api_keys) > 0,
        })


router = SecurityRouter().router
