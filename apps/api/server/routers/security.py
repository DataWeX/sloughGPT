"""
Security Router - Audit logs and API key management

API keys are stored in MogDB with JSON sync for human readability.
Keys are hashed (SHA-256 truncated) — raw keys are only returned on creation.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import endpoint, raise_error, success_response

logger = logging.getLogger("slo.routers.security")


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default=["*"])
    expires_at: int | None = Field(default=None, description="Unix timestamp for key expiration")


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

    @staticmethod
    @endpoint("security.audit_logs")
    async def get_audit_logs(
        limit: int = Query(
            default=100, ge=1, le=10000, description="Maximum number of log entries to return"
        ),
        event_type: str | None = Query(default=None, description="Filter by event type"),
        history: bool = Query(default=False, description="Read from persisted audit.log file"),
        before: str | None = Query(default=None, description="ISO-8601 cursor for pagination"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        from infrastructure.auth import get_audit_logger

        audit_logger = get_audit_logger()

        workspace_id = ""
        if auth_user and auth_user.get("sub"):
            workspace_id = auth_user.get("workspace_id", "")

        if history:
            logs = await asyncio.to_thread(
                audit_logger.file_query,
                limit=limit,
                event_type=event_type,
                before=before,
                workspace_id=workspace_id,
            )
        else:
            logs = audit_logger.logs[-limit:]
            if event_type:
                logs = [l for l in logs if l.get("event_type") == event_type]
            if workspace_id:
                logs = [l for l in logs if l.get("workspace_id", "") == workspace_id]
        if auth_user.get("role") not in ("owner", "admin"):
            raise_error("Admin access required", code="auth/forbidden", status=403)
        return success_response(data={"logs": logs, "count": len(logs)})

    # ── API key management ──

    @staticmethod
    @endpoint("security.create_key")
    async def create_key(body: dict, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        name = body.get("name", "")
        scopes = body.get("scopes", ["*"])
        expires_at = body.get("expires_at")
        key = mgr.create(name, scopes=scopes, expires_at=expires_at)
        return success_response(data=key)

    @staticmethod
    @endpoint("security.list_keys")
    async def list_keys(auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        keys = mgr.list()
        return success_response(data={"keys": keys, "count": len(keys)})

    @staticmethod
    @endpoint("security.get_key")
    async def get_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        key = mgr.get(key_id)
        if key is None:
            raise_error("API key not found", "E_NOT_FOUND", status_code=404)
        if auth_user.get("sub") != key.get("user_id") and auth_user.get("role") != "admin":
            raise_error("Not authorized to manage this key", code="auth/forbidden", status=403)
        return success_response(data=key)

    @staticmethod
    @endpoint("security.delete_key")
    async def delete_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            mgr = _get_key_manager()
            key = mgr.get(key_id)
            if key is None:
                raise_error("API key not found", "E_NOT_FOUND", status_code=404)
            if auth_user.get("sub") != key.get("user_id") and auth_user.get("role") != "admin":
                raise_error("Not authorized to manage this key", code="auth/forbidden", status=403)
            mgr.revoke(key_id)
            return success_response(data={"revoked": True})
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)

    @staticmethod
    @endpoint("security.rotate_key")
    async def rotate_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            mgr = _get_key_manager()
            key = mgr.get(key_id)
            if key is None:
                raise_error("API key not found", "E_NOT_FOUND", status_code=404)
            if auth_user.get("sub") != key.get("user_id") and auth_user.get("role") != "admin":
                raise_error("Not authorized to manage this key", code="auth/forbidden", status=403)
            new_key = mgr.rotate(key_id)
            return success_response(data=new_key)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)

    @staticmethod
    @endpoint("security.validate_key")
    async def validate_key(body: dict, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        key = body.get("key", "")
        valid = mgr.validate(key)
        return success_response(data={"valid": valid})


router = SecurityRouter().router
