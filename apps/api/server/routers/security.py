"""
Security Router - Audit logs and API key management
"""
from fastapi import APIRouter
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

    async def get_audit_logs(self, limit: int = 100, event_type: Optional[str] = None):
        """Get audit logs"""
        from infrastructure.auth import get_audit_logger
        audit_logger = get_audit_logger()
        logs = audit_logger.logs[-limit:]
        if event_type:
            logs = [l for l in logs if l.get("event_type") == event_type]
        return success_response(data={"logs": logs, "count": len(logs)})

    async def get_keys(self):
        """Get API key info (not the keys themselves)"""
        from settings import get_security_settings
        sec = get_security_settings()
        return success_response(data={
            "count": len(sec.valid_api_keys),
            "configured": len(sec.valid_api_keys) > 0,
        })


router = SecurityRouter().router
