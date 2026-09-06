"""
User Adapters Router - Per-user LoRA adapter management
"""

import logging
import threading
import time
from typing import Literal

from fastapi import APIRouter, Depends, Request
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.api.user_adapters")

# Response cache for list_adapters: avoids repeated MongoDB + O(n) disk stat calls.
# Keyed by a fixed sentinel; TTL 15s.
_list_cache: tuple[float, dict] | None = None
_LIST_CACHE_TTL = 15.0
_list_cache_lock = threading.Lock()


class AggregateBestRequest(BaseModel):
    top_k: int | None = 10
    min_feedback_count: int | None = 5
    output_name: str | None = "best_aggregated"


class AdapterUpdateRequest(BaseModel):
    rating: Literal["thumbs_up", "thumbs_down", "neutral"]
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)


class PruneAdaptersRequest(BaseModel):
    min_feedback_count: int = Field(default=1, ge=0, le=1000)
    max_age_days: int = Field(default=30, ge=1, le=3650)


class UserAdaptersRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/user-adapters", tags=["user-adapters"])
        self._register_routes()

    def _get_store(self):
        """Get the per-user LoRA store, raising 503 if not available."""
        try:
            from domains.feedback import get_per_user_lora

            return get_per_user_lora()
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)

    def _register_routes(self):
        self.router.add_api_route("", self.list_adapters, methods=["GET"])
        self.router.add_api_route("/quality", self.get_quality, methods=["GET"])
        self.router.add_api_route("/{user_id}", self.get_adapter, methods=["GET"])
        self.router.add_api_route("/{user_id}/update", self.update_adapter, methods=["POST"])
        self.router.add_api_route("/{user_id}/reset", self.reset_adapter, methods=["POST"])
        self.router.add_api_route("/merge", self.merge_adapters, methods=["POST"])
        self.router.add_api_route("/aggregate-best", self.aggregate_best, methods=["POST"])
        self.router.add_api_route("/{user_id}", self.delete_user_adapter, methods=["DELETE"])
        self.router.add_api_route("/prune", self.prune_low_quality_adapters, methods=["POST"])

    async def list_adapters(self) -> dict:
        """List all per-user LoRA adapters with aggregate statistics."""
        global _list_cache
        now = time.monotonic()
        with _list_cache_lock:
            if _list_cache and (now - _list_cache[0]) < _LIST_CACHE_TTL:
                return success_response(data=_list_cache[1])
        try:
            store = self._get_store()
            adapters = store.get_all_adapters()
            stats = store.get_stats()
            result = {"adapters": adapters, "stats": stats}
            with _list_cache_lock:
                _list_cache = (now, result)
            return success_response(data=result)
        except Exception as exc:
            logger.warning("List adapters failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.list")

    async def get_adapter(self, user_id: str) -> dict:
        """Retrieve a specific user's LoRA adapter metadata."""
        try:
            store = self._get_store()
            adapter = store.get_adapter(user_id)
            if adapter is None:
                return success_response(data={"user_id": user_id, "exists": False})
            return success_response(
                data={"user_id": user_id, "exists": True, "feedback_count": adapter.feedback_count}
            )
        except Exception as exc:
            logger.warning("Get adapter failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.get")

    async def update_adapter(
        self,
        user_id: str,
        req: AdapterUpdateRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Update a user's LoRA adapter with new feedback rating."""
        try:
            store = self._get_store()
            store.update_adapter(user_id, rating=req.rating)
            logger.info("Adapter updated (user=%s, rating=%s)", user_id, req.rating)
            safe_audit_log("adapter.update", resource=user_id, detail=f"rating={req.rating}")
            return success_response(data={"status": "updated", "user_id": user_id})
        except Exception as exc:
            logger.warning("Update adapter failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.update")

    async def reset_adapter(
        self, user_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Reset a user's LoRA adapter to its initial zero-weight state."""
        try:
            store = self._get_store()
            store.reset_user_adapter(user_id)
            logger.info("Adapter reset (user=%s)", user_id)
            safe_audit_log("adapter.reset", resource=user_id)
            return success_response(data={"status": "reset", "user_id": user_id})
        except Exception as exc:
            logger.warning("Reset adapter failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.reset")

    async def merge_adapters(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Merge all per-user LoRA adapters into a single combined adapter."""
        try:
            store = self._get_store()
            store.merge_all()
            logger.info("All adapters merged")
            safe_audit_log("adapter.merge", resource="all")
            return success_response(data={"status": "merged"})
        except Exception as exc:
            logger.warning("Merge adapters failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.merge")

    async def aggregate_best(
        self, req: AggregateBestRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Aggregate top-k best user adapters with auto-evaluation."""
        try:
            store = self._get_store()
            result = store.aggregate_best_adapters(
                top_k=req.top_k,
                min_feedback_count=req.min_feedback_count,
                output_name=req.output_name,
            )
            safe_audit_log(
                "adapter.aggregate",
                resource=req.output_name or "best_aggregated",
                user_count=result.get("user_count", 0),
                total_feedback=result.get("total_feedback", 0),
            )
            eval_result = result.get("eval", {})
            if "error" not in eval_result:
                return success_response(
                    data={
                        "status": "aggregated_with_eval",
                        "output_path": result.get("output_path", ""),
                        "user_count": result.get("user_count", 0),
                        "total_feedback": result.get("total_feedback", 0),
                        "eval": {
                            "verdict": eval_result.get("delta", {}).get("verdict", "unknown"),
                            "perplexity_delta": eval_result.get("delta", {}).get(
                                "perplexity_delta"
                            ),
                            "bleu_delta": eval_result.get("delta", {}).get("bleu_delta"),
                            "throughput_delta": eval_result.get("delta", {}).get(
                                "throughput_delta"
                            ),
                            "report": eval_result.get("report", ""),
                        },
                    }
                )
            return success_response(
                data={"status": "aggregated", "count": result.get("user_count", 0)}
            )
        except Exception as exc:
            logger.warning("Aggregate best failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.aggregate")

    async def get_quality(
        self, min_feedback_count: int = 3, max_age_days: int | None = None
    ) -> dict:
        """Retrieve quality metrics report for all eligible adapters."""
        try:
            store = self._get_store()
            return success_response(
                data=store.get_quality_report(
                    min_feedback_count=min_feedback_count,
                    max_age_days=max_age_days,
                )
            )
        except Exception as exc:
            logger.warning("Quality report failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.quality")

    async def delete_user_adapter(
        self, user_id: str, req: Request, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Delete a user's LoRA adapter."""
        try:
            store = self._get_store()
            store.delete_adapter(user_id)
            logger.info("Adapter deleted (user=%s)", user_id)
            safe_audit_log("adapter.delete", resource=user_id)
            return success_response(data={"status": "deleted", "user_id": user_id})
        except Exception as exc:
            logger.warning("Delete adapter failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.delete")

    async def prune_low_quality_adapters(
        self,
        request: PruneAdaptersRequest,
        req: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove adapters with too few feedback or too old."""
        try:
            store = self._get_store()
            deleted = store.prune_low_quality(
                min_feedback_count=request.min_feedback_count,
                max_age_days=request.max_age_days,
            )
            logger.info(
                "Pruned %d low-quality adapters (min_feedback=%d, max_age=%dd)",
                len(deleted),
                request.min_feedback_count,
                request.max_age_days,
            )
            safe_audit_log(
                "adapter.prune",
                resource="all",
                detail=f"deleted={len(deleted)}",
                deleted_users=deleted,
            )
            return success_response(
                data={
                    "status": "pruned",
                    "deleted_count": len(deleted),
                    "deleted_users": deleted,
                }
            )
        except Exception as exc:
            logger.warning("Prune adapters failed: %s", exc)
            classify_and_raise(exc, source="user_adapters.prune")


router = UserAdaptersRouter().router
