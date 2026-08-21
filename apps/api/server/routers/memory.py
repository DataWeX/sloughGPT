"""Memory API router - inspect, search, store, and clear the auto-memory layer.

Thin HTTP wrapper over ``domains.memory.memory_service`` (endpoints are
adapters; all logic lives in core). Exposed so frontends and integrations can
manage the memory store the chat loop writes to automatically.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from domains.memory.memory_service import get_memory_service
from schemas.common import raise_error, safe_audit_log

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
        """
        Return memory statistics (facts, topic buckets, enabled state).

        Returns:
            dict: ``{enabled, total_facts, topics, visited_urls}``.

        Side effects:
            - none; read-only.
        """
        svc = self._service()
        stats = svc.stats() or {}
        stats["enabled"] = svc.enabled
        return stats

    def list_memory(
        self,
        limit: int = Query(default=50, ge=1, le=1000, description="Maximum number of items to return"),
    ) -> dict:
        """
        List stored memory items, most recent first.

        Args:
            limit: maximum number of items to return (default 50).

        Returns:
            dict: ``{items: [...], total: N}``.

        Side effects:
            - none; read-only.
        """
        items = self._service().list_all(limit=limit)
        return {"items": items, "total": len(items)}

    def search(
        self,
        q: str = Query(..., min_length=1, description="The lookup text"),
        limit: int = Query(default=5, ge=1, le=100, description="Maximum number of results"),
    ) -> dict:
        """
        Semantic-search stored memory.

        Args:
            q: the lookup text (query parameter).
            limit: maximum number of results (default 5).

        Returns:
            dict: ``{results: [...], total: N}``.

        Side effects:
            - none; read-only.
        """
        results = self._service().retrieve(q.strip(), limit=limit)
        return {"results": results, "total": len(results)}

    def store(self, req: StoreRequest) -> dict:
        """
        Persist one explicit fact.

        Args:
            req: ``{content, topic, source}`` request body.

        Returns:
            dict: ``{stored: bool, content, topic, source}``.

        Side effects:
            - writes the fact into the underlying knowledge store.
        """
        content = req.content.strip()
        if not content:
            raise_error("content is required", "E_BAD_REQUEST", status_code=400)
        topic = req.topic or "manual"
        source = req.source or "api"
        try:
            stored = self._service().store(content, topic, source)
        except Exception as e:
            logger.error("Memory store failed (topic=%s): %s", topic, e, exc_info=True)
            raise
        safe_audit_log("memory.store", resource=topic, detail=f"stored={stored}")
        return {"stored": stored, "content": content, "topic": topic, "source": source}

    def remember(self, req: RememberRequest) -> dict:
        """
        Persist one completed turn as durable memory.

        Args:
            req: ``{user_message, assistant_response}`` request body.

        Returns:
            dict: ``{stored: bool, reason: str}``.

        Side effects:
            - extracts and stores facts from the turn.
        """
        user_message = req.user_message.strip()
        assistant_response = req.assistant_response.strip()
        if not user_message or not assistant_response:
            raise_error("user_message and assistant_response are required", "E_BAD_REQUEST", status_code=400)
        try:
            stored = self._service().remember(user_message, assistant_response)
        except Exception as e:
            logger.error("Memory remember failed: %s", e, exc_info=True)
            raise
        reason = "stored" if stored else "skipped (disabled, too short, or nothing new)"
        return {"stored": stored, "reason": reason}

    def set_config(self, req: ConfigRequest) -> dict:
        """
        Update runtime memory settings.

        Args:
            req: body with optional ``enabled`` and/or
                ``archive_retention_days``. Omitted fields are left unchanged.

        Returns:
            dict: the full updated settings snapshot.

        Side effects:
            - mutates the process-wide ``MemoryConfig`` singleton; every
              subsequent memory call reflects the new state until the next
              update or process restart.
        """
        svc = self._service()
        if req.enabled is not None:
            svc.set_enabled(req.enabled)
        if req.archive_retention_days is not None:
            svc.set_archive_retention(req.archive_retention_days)
        safe_audit_log("memory.config", resource="memory", detail=f"enabled={req.enabled} retention={req.archive_retention_days}")
        return svc.config_snapshot()

    def get_config(self) -> dict:
        """
        Return the current runtime memory settings.

        Returns:
            dict: ``{enabled, min_chars, max_facts, store_path,
                sync_remember, consolidation_threshold,
                maintenance_interval_minutes, archive_retention_days}``.

        Side effects:
            - none; read-only.
        """
        return self._service().config_snapshot()

    def delete_item(self, item_id: str) -> dict:
        """
        Remove one stored memory item by entry id.

        Args:
            item_id: the vector-store entry id (from ``list``/``search``).

        Returns:
            dict: ``{deleted: 0|1}`` — 1 when the item was removed.

        Side effects:
            - removes the matching fact from the knowledge store (persisted).
        """
        if not item_id or not item_id.strip():
            raise_error("item_id is required", "E_BAD_REQUEST", status_code=400)
        removed = self._service().delete([item_id.strip()])
        safe_audit_log("memory.delete", resource=item_id, detail=f"deleted={removed}")
        return {"deleted": removed}

    def update_item(self, item_id: str, req: UpdateRequest) -> dict:
        """
        Edit a stored memory item's text (and optionally its topic/importance).

        Args:
            item_id: the vector-store entry id (from ``list``/``search``).
            req: new content, optional topic, and optional importance.

        Returns:
            dict: ``{updated: 0|1, duplicate: bool}`` — ``duplicate`` is True
                when the new text already exists as another fact.

        Side effects:
            - replaces the fact's text/embedding in the knowledge store.
        """
        if not item_id or not item_id.strip():
            raise_error("item_id is required", "E_BAD_REQUEST", status_code=400)
        if not req.content or not req.content.strip():
            raise_error("content is required", "E_BAD_REQUEST", status_code=400)
        updated = self._service().update(item_id.strip(), req.content, topic=req.topic, importance=req.importance)
        safe_audit_log("memory.update", resource=item_id, detail=f"updated={updated}")
        return {"updated": 1 if updated else 0, "duplicate": not updated}

    def clear(self) -> dict:
        """
        Remove every stored memory item.

        Returns:
            dict: ``{cleared: N}``.

        Side effects:
            - wipes the underlying knowledge store.
        """
        removed = self._service().clear()
        safe_audit_log("memory.clear", resource="all", detail=f"cleared={removed}")
        return {"cleared": removed}
    def consolidate(self, threshold: Optional[float] = None) -> dict:
        """
        Merge near-duplicate facts, keeping the longest in each cluster.

        Runs the same planning the ``memory.consolidate`` task uses: facts in
        the same topic whose n-gram cosine similarity is at or above the
        threshold are collapsed, deleting the shorter copies.

        Args:
            threshold: min n-gram cosine for near-dup merge; defaults to
                ``MemoryConfig.consolidation_threshold`` when omitted.

        Returns:
            dict: ``{removed: N, kept: N, threshold: float}``.

        Side effects:
            - deletes near-duplicate facts from the shared memory store.
        """
        from domains.memory.consolidation import plan_consolidation
        from domains.memory.memory_config import MemoryConfig
        svc = self._service()
        if threshold is None:
            threshold = MemoryConfig.get().consolidation_threshold
        threshold = float(threshold)
        facts = svc.list_all(limit=5000)
        plan = plan_consolidation(facts, threshold=threshold)
        try:
            removed = svc.delete(plan["remove_ids"]) if plan["remove_ids"] else 0
        except Exception as e:
            logger.error("Memory consolidation delete failed (threshold=%s): %s", threshold, e, exc_info=True)
            raise
        safe_audit_log("memory.consolidate", resource="all", detail=f"removed={removed}, kept={len(plan['keep_ids'])}")
        return {"removed": removed, "kept": len(plan["keep_ids"]), "threshold": threshold}

    def archive(self, limit: Optional[int] = None) -> dict:
        """
        Return recent task-backed provenance archive records, newest first.

        Args:
            limit: max records to return (default 20, clamped 1..1000).

        Returns:
            dict: ``{records: [...], total: N}``.

        Side effects:
            - none; read-only.
        """
        from domains.memory.task_memory import list_archive
        if limit is None:
            limit = 20
        limit = max(1, min(int(limit), 1000))
        records = list_archive(limit=limit)
        return {"records": records, "total": len(records)}

    def archive_stats(self) -> dict:
        """
        Summarize the task-backed provenance archive.

        Returns:
            dict: ``{path, records, bytes, task_types, oldest_ts, newest_ts}``.

        Side effects:
            - none; read-only.
        """
        from domains.memory.task_memory import archive_stats
        return archive_stats()

    def archive_prune(self, retain_days: Optional[float] = None) -> dict:
        """
        Delete archive records older than the retention window.

        Args:
            retain_days: retention window in days; records older than this are
                removed. ``0`` prunes everything. Defaults to
                ``MemoryConfig.archive_retention_days`` when omitted.

        Returns:
            dict: ``{pruned: N}`` — records removed.

        Side effects:
            - rewrites ``facts.jsonl`` keeping only records inside the window.
        """
        from domains.memory.task_memory import prune_archive
        try:
            removed = prune_archive(retain_days=retain_days)
        except Exception as e:
            logger.error("Archive prune failed (retain_days=%s): %s", retain_days, e, exc_info=True)
            raise
        safe_audit_log("memory.archive_prune", resource="archive", detail=f"pruned={removed}")
        return {"pruned": removed}


router = MemoryRouter().router
