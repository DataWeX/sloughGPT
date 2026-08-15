"""
User Adapters Router - Per-user LoRA adapter management
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from schemas.common import success_response
from infrastructure.auth import require_auth_if_enabled, audit_user, get_audit_logger


class AggregateBestRequest(BaseModel):
    top_k: Optional[int] = 10
    min_feedback_count: Optional[int] = 5
    output_name: Optional[str] = "best_aggregated"


class AdapterUpdateRequest(BaseModel):
    rating: str
    quality_score: Optional[float] = None


class PruneAdaptersRequest(BaseModel):
    min_feedback_count: int = 1
    max_age_days: int = 30


class UserAdaptersRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/user-adapters", tags=["user-adapters"])
        self._register_routes()

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

    async def list_adapters(self):
        """List all user adapters"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            adapters = store.get_all_adapters()
            stats = store.get_stats()
            return success_response(data={"adapters": adapters, "stats": stats})
        except ImportError:
            raise HTTPException(status_code=503, detail="Per-user LoRA not available")

    async def get_adapter(self, user_id: str):
        """Get specific user's adapter"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            adapter = store.get_adapter(user_id)
            if adapter is None:
                return success_response(data={"user_id": user_id, "exists": False})
            return success_response(data={"user_id": user_id, "exists": True, "feedback_count": adapter.feedback_count})
        except ImportError:
            raise HTTPException(status_code=503, detail="Not available")

    async def update_adapter(self, user_id: str, req: AdapterUpdateRequest):
        """Update user's LoRA adapter"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.update_adapter(user_id, rating=req.rating)
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("adapter.update", resource=user_id, detail=f"rating={req.rating}")
            except Exception:
                pass
            return success_response(data={"status": "updated", "user_id": user_id})
        except ImportError:
            raise HTTPException(status_code=503, detail="Not available")

    async def reset_adapter(self, user_id: str):
        """Reset user's adapter"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.reset_user_adapter(user_id)
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("adapter.reset", resource=user_id)
            except Exception:
                pass
            return success_response(data={"status": "reset", "user_id": user_id})
        except ImportError:
            raise HTTPException(status_code=503, detail="Not available")

    async def merge_adapters(self):
        """Merge all adapters"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.merge_all()
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("adapter.merge", resource="all")
            except Exception:
                pass
            return success_response(data={"status": "merged"})
        except ImportError:
            raise HTTPException(status_code=503, detail="Not available")

    async def aggregate_best(self, req: AggregateBestRequest, auth_user: dict = Depends(require_auth_if_enabled)):
        """Aggregate top-k best user adapters with auto-evaluation."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            result = store.aggregate_best_adapters(
                top_k=req.top_k,
                min_feedback_count=req.min_feedback_count,
                output_name=req.output_name,
            )
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "adapter.aggregate",
                    resource=req.output_name or "best_aggregated",
                    extra={"user_count": result.get("user_count", 0), "total_feedback": result.get("total_feedback", 0)},
                )
            except Exception:
                pass
            eval_result = result.get("eval", {})
            if "error" not in eval_result:
                return success_response(data={
                    "status": "aggregated_with_eval",
                    "output_path": result.get("output_path", ""),
                    "user_count": result.get("user_count", 0),
                    "total_feedback": result.get("total_feedback", 0),
                    "eval": {
                        "verdict": eval_result.get("delta", {}).get("verdict", "unknown"),
                        "perplexity_delta": eval_result.get("delta", {}).get("perplexity_delta"),
                        "bleu_delta": eval_result.get("delta", {}).get("bleu_delta"),
                        "throughput_delta": eval_result.get("delta", {}).get("throughput_delta"),
                        "report": eval_result.get("report", ""),
                    },
                })
            return success_response(data={"status": "aggregated", "count": result.get("user_count", 0)})
        except ImportError:
            raise HTTPException(status_code=503, detail="Per-user LoRA not available")

    async def get_quality(self, min_feedback_count: int = 3, max_age_days: Optional[int] = None):
        """Get adapter quality metrics"""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            return success_response(
                data=store.get_quality_report(
                    min_feedback_count=min_feedback_count,
                    max_age_days=max_age_days,
                )
            )
        except ImportError:
            raise HTTPException(status_code=503, detail="Not available")

    async def delete_user_adapter(self, user_id: str, req: Request, auth_user: dict = Depends(require_auth_if_enabled)):
        """Delete a user's LoRA adapter."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.delete_adapter(user_id)
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("adapter.delete", user=audit_user(auth_user), resource=user_id)
            except Exception:
                pass
            return success_response(data={"status": "deleted", "user_id": user_id})
        except ImportError:
            raise HTTPException(status_code=503, detail="Per-user LoRA not available")

    async def prune_low_quality_adapters(self, request: PruneAdaptersRequest, req: Request):
        """Remove adapters with too few feedback or too old."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            deleted = store.prune_low_quality(
                min_feedback_count=request.min_feedback_count,
                max_age_days=request.max_age_days,
            )
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "adapter.prune",
                    resource="all",
                    detail=f"deleted={len(deleted)}",
                    extra={"deleted_users": deleted},
                )
            except Exception:
                pass
            return success_response(data={
                "status": "pruned",
                "deleted_count": len(deleted),
                "deleted_users": deleted,
            })
        except ImportError:
            raise HTTPException(status_code=503, detail="Per-user LoRA not available")


router = UserAdaptersRouter().router
