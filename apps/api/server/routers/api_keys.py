"""API key management with MogDB persistence and JSON sync."""
from __future__ import annotations

import hashlib
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import endpoint, raise_error, success_response

logger = logging.getLogger("slo.api_keys")


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class ApiKeyManager:
    """Manages API keys with MogDB storage and optional JSON sync."""

    def __init__(self, db=None, db_path: str | Path | None = None, sync_dir: str | Path | None = None):
        if db is not None:
            self._db = db
        else:
            from mogdb import MogDB

            if db_path is None:
                repo_root = Path(__file__).parent.parent.parent.parent
                db_path = repo_root / "data" / "api_keys_mogdb"
            if sync_dir is None:
                repo_root = Path(__file__).parent.parent.parent.parent
                sync_dir = repo_root / "data" / "api_keys_json"
            self._db = MogDB(str(db_path), sync_dir=str(sync_dir))
        self._collection = self._db.collection("api_keys")

    def create(
        self,
        name: str,
        scopes: list[str] | None = None,
        expires_at: int | None = None,
        workspace_id: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        key = f"slo_{secrets.token_urlsafe(32)}"
        doc = {
            "name": name,
            "key": key,
            "key_hash": _hash_key(key),
            "scopes": scopes or ["*"],
            "created_at": int(time.time()),
            "workspace_id": workspace_id,
            "user_id": user_id,
        }
        if expires_at is not None:
            doc["expires_at"] = expires_at
        doc_id = self._collection.insert_one(doc)
        doc["id"] = doc_id
        doc["revoked"] = False
        logger.info("Created API key '%s' (id=%s, ws=%s)", name, doc_id, workspace_id or "global")
        return doc

    def list(self, workspace_id: str = "", user_id: str = "") -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if workspace_id:
            query["workspace_id"] = workspace_id
        if user_id:
            query["user_id"] = user_id
        docs = self._collection.find(query)
        result = []
        for d in docs:
            entry = {
                "id": d["_id"],
                "name": d["name"],
                "key_hash": d["key_hash"],
                "scopes": d.get("scopes", ["*"]),
                "created_at": d["created_at"],
                "revoked": d.get("revoked", False),
                "workspace_id": d.get("workspace_id", ""),
                "user_id": d.get("user_id", ""),
            }
            if "expires_at" in d:
                entry["expires_at"] = d["expires_at"]
            result.append(entry)
        return result

    def get(self, key_id: str) -> Optional[dict[str, Any]]:
        doc = self._collection.find_one({"_id": key_id})
        if doc is None:
            return None
        return {
            "id": doc["_id"],
            "name": doc["name"],
            "key": doc["key"],
            "key_hash": doc["key_hash"],
            "scopes": doc.get("scopes", ["*"]),
            "created_at": doc["created_at"],
            "expires_at": doc.get("expires_at"),
            "revoked": doc.get("revoked", False),
            "workspace_id": doc.get("workspace_id", ""),
            "user_id": doc.get("user_id", ""),
        }

    def revoke(self, key_id: str) -> None:
        doc = self._collection.find_one({"_id": key_id})
        if doc is None:
            raise ValueError(f"API key not found: {key_id}")
        self._collection.update_one({"_id": key_id}, {"$set": {"revoked": True}})
        logger.info("Revoked API key '%s' (id=%s)", doc.get("name"), key_id)

    def rotate(self, key_id: str) -> dict[str, Any]:
        old = self._collection.find_one({"_id": key_id})
        if old is None:
            raise ValueError(f"API key not found: {key_id}")
        self.revoke(key_id)
        return self.create(
            old["name"],
            scopes=old.get("scopes", ["*"]),
            expires_at=old.get("expires_at"),
            workspace_id=old.get("workspace_id", ""),
            user_id=old.get("user_id", ""),
        )

    def validate(self, key: str) -> bool:
        key_hash = _hash_key(key)
        docs = self._collection.find({"key_hash": key_hash})
        for doc in docs:
            if doc.get("revoked", False):
                return False
            if "expires_at" in doc and doc["expires_at"] < time.time():
                return False
            return True
        return False


class ApiKeysRouter:
    """FastAPI router for API key CRUD operations."""

    def __init__(self, key_manager: ApiKeyManager | None = None):
        self._manager = key_manager or ApiKeyManager()
        self.router = APIRouter(prefix="/security", tags=["security"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/keys", endpoint=self.create_key, methods=["POST"])
        self.router.add_api_route(path="/keys", endpoint=self.list_keys, methods=["GET"])
        self.router.add_api_route(path="/keys/validate", endpoint=self.validate_key, methods=["POST"])
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.get_key, methods=["GET"])
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.delete_key, methods=["DELETE"])
        self.router.add_api_route(path="/keys/{key_id}/rotate", endpoint=self.rotate_key, methods=["POST"])

    def _get_workspace_user(self, auth_user: dict) -> tuple[str, str]:
        """Extract workspace_id and user_id from auth_user."""
        user_id = auth_user.get("sub", "") if auth_user else ""
        workspace_id = auth_user.get("workspace_id", "") if auth_user else ""
        return workspace_id, user_id

    @endpoint("api_keys.create")
    async def create_key(self, body: dict, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        name = body.get("name", "")
        scopes = body.get("scopes", ["*"])
        expires_at = body.get("expires_at")
        workspace_id, user_id = self._get_workspace_user(auth_user)
        key = self._manager.create(
            name, scopes=scopes, expires_at=expires_at,
            workspace_id=workspace_id, user_id=user_id,
        )
        return success_response(data=key)

    @endpoint("api_keys.list")
    async def list_keys(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        workspace_id, user_id = self._get_workspace_user(auth_user)
        keys = self._manager.list(workspace_id=workspace_id, user_id=user_id)
        return success_response(data={"keys": keys, "count": len(keys)})

    @endpoint("api_keys.get")
    async def get_key(self, key_id: str) -> dict:
        key = self._manager.get(key_id)
        if key is None:
            raise_error("API key not found", "E_NOT_FOUND", status_code=404)
        return success_response(data=key)

    @endpoint("api_keys.delete")
    async def delete_key(self, key_id: str) -> dict:
        try:
            self._manager.revoke(key_id)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        return success_response(data={"revoked": True})

    @endpoint("api_keys.rotate")
    async def rotate_key(self, key_id: str) -> dict:
        try:
            new_key = self._manager.rotate(key_id)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        return success_response(data=new_key)

    @endpoint("api_keys.validate")
    async def validate_key(self, body: dict) -> dict:
        key = body.get("key", "")
        valid = self._manager.validate(key)
        return success_response(data={"valid": valid})


_api_keys_router: ApiKeysRouter | None = None


def get_api_keys_router() -> ApiKeysRouter:
    global _api_keys_router
    if _api_keys_router is None:
        _api_keys_router = ApiKeysRouter()
    return _api_keys_router
