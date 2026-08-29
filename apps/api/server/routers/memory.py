"""Memory API router - inspect, search, store, and clear the auto-memory layer.

Thin HTTP wrapper over ``domains.memory.memory_service`` (endpoints are
adapters; all logic lives in core). Exposed so frontends and integrations can
manage the memory store the chat loop writes to automatically.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel, Field

from domains.memory.memory_service import get_memory_service
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, safe_audit_log, success_response, classify_and_raise

logger = logging.getLogger("slo.api.memory")


class StoreRequest(BaseModel):
    """Body for POST /memory/store."""

    content: str = Field(..., min_length=1)
    topic: str = "manual"
    source: str = "api"


class RememberRequest(BaseModel):
    """Body for POST /memory/remember."""

    user_message: str = Field(..., min_length=1)
    assistant_response: str = Field(..., min_length=1)


class ConfigRequest(BaseModel):
    """Body for POST /memory/config."""

    enabled: Optional[bool] = Field(default=None, description="Master switch for the memory layer.")
    archive_retention_days: Optional[float] = Field(default=None, description="Retention window in days for archive pruning; 0 prunes everything.")


class UpdateRequest(BaseModel):
    """Body for PATCH /memory/{item_id}."""

    content: str = Field(..., min_length=1, description="New fact text.")
    topic: Optional[str] = Field(default=None, description="Optional new topic label; keeps existing when omitted.")
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional importance score in [0, 1]; keeps existing when omitted.")


class MemoryRouter:
    """FastAPI router exposing the auto-memory layer over HTTP."""

    def __init__(self):
        self.router = APIRouter(prefix="/memory", tags=["memory"])
        self.router.add_api_route(
            "/stats", self.stats, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/list", self.list_memory, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/search", self.search, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/store", self.store, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/remember", self.remember, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/config", self.set_config, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/config", self.get_config, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/clear", self.clear, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/consolidate", self.consolidate, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/archive", self.archive, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/archive/stats", self.archive_stats, methods=["GET"], response_model=dict,
        )
        self.router.add_api_route(
            "/archive/prune", self.archive_prune, methods=["POST"], response_model=dict,
        )
        self.router.add_api_route(
            "/{item_id}", self.delete_item, methods=["DELETE"], response_model=dict,
        )
        self.router.add_api_route(
            "/{item_id}", self.update_item, methods=["PATCH"], response_model=dict,
        )

    def _service(self):
        """Resolve the process-wide MemoryService."""
        return get_memory_service()

    def stats(self) -> dict:
        """Return memory statistics (facts, topic buckets, enabled state)."""
        try:
            svc = self._service()
            stats = svc.stats() or {}
            stats["enabled"] = svc.enabled
            return stats
        except Exception as e:
            classify_and_raise(e, source="memory.stats")

    def list_memory(
        self,
        limit: int = Query(default=50, ge=1, le=1000, description="Maximum number of items to return"),
    ) -> dict:
        """List stored memory items, most recent first."""
        try:
            items = self._service().list_all(limit=limit)
            return success_response(data={"items": items, "total": len(items)})
        except Exception as e:
            classify_and_raise(e, source="memory.list")

    def search(
        self,
        q: str = Query(..., min_length=1, description="The lookup text"),
        limit: int = Query(default=5, ge=1, le=100, description="Maximum number of results"),
    ) -> dict:
        """Semantic-search stored memory."""
        try:
            results = self._service().retrieve(q.strip(), limit=limit)
            return success_response(data={"results": results, "total": len(results)})
        except Exception as e:
            classify_and_raise(e, source="memory.search")

    def store(self, req: StoreRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Persist one explicit fact."""
        try:
            content = req.content.strip()
            if not content:
                raise_error("content is required", "E_BAD_REQUEST", status_code=400)
            topic = req.topic or "manual"
            source = req.source or "api"
            stored = self._service().store(content, topic, source)
            safe_audit_log("memory.store", resource=topic, detail=f"stored={stored}")
            return success_response(data={"stored": stored, "content": content, "topic": topic, "source": source})
        except Exception as e:
            classify_and_raise(e, source="memory.store")

    def remember(self, req: RememberRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Persist one completed turn as durable memory."""
        try:
            user_message = req.user_message.strip()
            assistant_response = req.assistant_response.strip()
            if not user_message or not assistant_response:
                raise_error("user_message and assistant_response are required", "E_BAD_REQUEST", status_code=400)
            stored = self._service().remember(user_message, assistant_response)
            reason = "stored" if stored else "skipped (disabled, too short, or nothing new)"
            return success_response(data={"stored": stored, "reason": reason})
        except Exception as e:
            classify_and_raise(e, source="memory.remember")

    def set_config(self, req: ConfigRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Update runtime memory settings."""
        try:
            svc = self._service()
            if req.enabled is not None:
                svc.set_enabled(req.enabled)
            if req.archive_retention_days is not None:
                svc.set_archive_retention(req.archive_retention_days)
            safe_audit_log("memory.config", resource="memory", detail=f"enabled={req.enabled} retention={req.archive_retention_days}")
            return success_response(data=svc.config_snapshot())
        except Exception as e:
            classify_and_raise(e, source="memory.set_config")

    def get_config(self) -> dict:
        """Return the current runtime memory settings."""
        try:
            return success_response(data=self._service().config_snapshot())
        except Exception as e:
            classify_and_raise(e, source="memory.get_config")

    def delete_item(self, item_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Remove one stored memory item by entry id."""
        try:
            if not item_id or not item_id.strip():
                raise_error("item_id is required", "E_BAD_REQUEST", status_code=400)
            removed = self._service().delete([item_id.strip()])
            safe_audit_log("memory.delete", resource=item_id, detail=f"deleted={removed}")
            return success_response(data={"deleted": removed})
        except Exception as e:
            classify_and_raise(e, source="memory.delete")

    def update_item(self, item_id: str, req: UpdateRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Edit a stored memory item's text (and optionally its topic/importance)."""
        try:
            if not item_id or not item_id.strip():
                raise_error("item_id is required", "E_BAD_REQUEST", status_code=400)
            if not req.content or not req.content.strip():
                raise_error("content is required", "E_BAD_REQUEST", status_code=400)
            updated = self._service().update(item_id.strip(), req.content, topic=req.topic, importance=req.importance)
            safe_audit_log("memory.update", resource=item_id, detail=f"updated={updated}")
            return success_response(data={"updated": 1 if updated else 0, "duplicate": not updated})
        except Exception as e:
            classify_and_raise(e, source="memory.update")

    def clear(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Remove every stored memory item."""
        try:
            removed = self._service().clear()
            safe_audit_log("memory.clear", resource="all", detail=f"cleared={removed}")
            return success_response(data={"cleared": removed})
        except Exception as e:
            classify_and_raise(e, source="memory.clear")
    def consolidate(self, threshold: Optional[float] = None, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Merge near-duplicate facts, keeping the longest in each cluster."""
        try:
            from domains.memory.consolidation import plan_consolidation
            from domains.memory.memory_config import MemoryConfig
            svc = self._service()
            if threshold is None:
                threshold = MemoryConfig.get().consolidation_threshold
            threshold = float(threshold)
            facts = svc.list_all(limit=5000)
            plan = plan_consolidation(facts, threshold=threshold)
            removed = svc.delete(plan["remove_ids"]) if plan["remove_ids"] else 0
            safe_audit_log("memory.consolidate", resource="all", detail=f"removed={removed}, kept={len(plan['keep_ids'])}")
            return success_response(data={"removed": removed, "kept": len(plan["keep_ids"]), "threshold": threshold})
        except Exception as e:
            classify_and_raise(e, source="memory.consolidate")

    def archive(self, limit: Optional[int] = None) -> dict:
        """Return recent task-backed provenance archive records, newest first."""
        try:
            from domains.memory.task_memory import list_archive
            if limit is None:
                limit = 20
            limit = max(1, min(int(limit), 1000))
            records = list_archive(limit=limit)
            return success_response(data={"records": records, "total": len(records)})
        except Exception as e:
            classify_and_raise(e, source="memory.archive")

    def archive_stats(self) -> dict:
        """Summarize the task-backed provenance archive."""
        try:
            from domains.memory.task_memory import archive_stats
            return success_response(data=archive_stats())
        except Exception as e:
            classify_and_raise(e, source="memory.archive_stats")

    def archive_prune(self, retain_days: Optional[float] = None, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete archive records older than the retention window."""
        try:
            from domains.memory.task_memory import prune_archive
            removed = prune_archive(retain_days=retain_days)
            safe_audit_log("memory.archive_prune", resource="archive", detail=f"pruned={removed}")
            return success_response(data={"pruned": removed})
        except Exception as e:
            classify_and_raise(e, source="memory.archive_prune")


router = MemoryRouter().router
