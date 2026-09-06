"""
Security Router - Audit logs and API key management

API keys are stored in MogDB with JSON sync for human readability.
Keys are hashed (SHA-256 truncated) — raw keys are only returned on creation.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, raise_error, success_response

logger = logging.getLogger("slo.routers.security")


def _get_key_manager():
    from routers.api_keys import ApiKeyManager
    return ApiKeyManager()


class SecurityRouter:
    """Security Router - Audit logs and API key management."""

    def __init__(self):
        self.router = APIRouter(prefix="/security", tags=["security"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/audit", endpoint=self.get_audit_logs, methods=["GET"])
        self.router.add_api_route(path="/keys", endpoint=self.list_keys, methods=["GET"])
        self.router.add_api_route(path="/keys", endpoint=self.create_key, methods=["POST"])
        self.router.add_api_route(path="/keys/validate", endpoint=self.validate_key, methods=["POST"])
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.get_key, methods=["GET"])
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.delete_key, methods=["DELETE"])
        self.router.add_api_route(path="/keys/{key_id}/rotate", endpoint=self.rotate_key, methods=["POST"])

    # ── Audit logs ──

    async def get_audit_logs(
        self,
        limit: int = Query(
            default=100, ge=1, le=10000, description="Maximum number of log entries to return"
        ),
        event_type: str | None = Query(default=None, description="Filter by event type"),
        history: bool = Query(default=False, description="Read from persisted audit.log file"),
        before: str | None = Query(default=None, description="ISO-8601 cursor for pagination"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get audit logs."""
        try:
            from infrastructure.auth import get_audit_logger

            audit_logger = get_audit_logger()
            if history:
                logs = await asyncio.to_thread(
                    audit_logger.file_query, limit=limit, event_type=event_type, before=before
                )
            else:
                logs = audit_logger.logs[-limit:]
                if event_type:
                    logs = [l for l in logs if l.get("event_type") == event_type]
            return success_response(data={"logs": logs, "count": len(logs)})
        except Exception as e:
            classify_and_raise(e, source="security.audit_logs")

    # ── API key management ──

    async def create_key(self, body: dict, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Create a new API key. The raw key is returned only in this response."""
        try:
            mgr = _get_key_manager()
            name = body.get("name", "")
            scopes = body.get("scopes", ["*"])
            expires_at = body.get("expires_at")
            key = mgr.create(name, scopes=scopes, expires_at=expires_at)
            return success_response(data=key)
        except Exception as e:
            classify_and_raise(e, source="security.create_key")

    async def list_keys(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """List all API keys (hashes only, not raw keys)."""
        try:
            mgr = _get_key_manager()
            keys = mgr.list()
            return success_response(data={"keys": keys, "count": len(keys)})
        except Exception as e:
            classify_and_raise(e, source="security.list_keys")

    async def get_key(self, key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get API key details by ID."""
        try:
            mgr = _get_key_manager()
            key = mgr.get(key_id)
            if key is None:
                raise_error("API key not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=key)
        except Exception as e:
            classify_and_raise(e, source="security.get_key")

    async def delete_key(self, key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Revoke an API key (soft delete)."""
        try:
            mgr = _get_key_manager()
            mgr.revoke(key_id)
            return success_response(data={"revoked": True})
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        except Exception as e:
            classify_and_raise(e, source="security.delete_key")

    async def rotate_key(self, key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Rotate an API key: revoke the old one, create a new one."""
        try:
            mgr = _get_key_manager()
            new_key = mgr.rotate(key_id)
            return success_response(data=new_key)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        except Exception as e:
            classify_and_raise(e, source="security.rotate_key")

    async def validate_key(self, body: dict) -> dict:
        """Validate an API key (public endpoint for auth checks)."""
        try:
            mgr = _get_key_manager()
            key = body.get("key", "")
            valid = mgr.validate(key)
            return success_response(data={"valid": valid})
        except Exception as e:
            classify_and_raise(e, source="security.validate_key")


router = SecurityRouter().router
