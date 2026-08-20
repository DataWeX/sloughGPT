"""
User Adapters Router - Per-user LoRA adapter management
"""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from typing import Optional

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log
from infrastructure.auth import require_auth_if_enabled, audit_user


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

    async def list_adapters(self) -> dict:
        """List all per-user LoRA adapters with aggregate statistics.

        Returns every registered adapter's metadata along with summary
        stats (total count, average feedback, etc.).

        Returns:
            Success envelope with adapters array and stats dict.

        Raises:
            503 if the per-user LoRA module is not available.
        """
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            adapters = store.get_all_adapters()
            stats = store.get_stats()
            return success_response(data={"adapters": adapters, "stats": stats})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.list")

    async def get_adapter(self, user_id: str) -> dict:
        """Retrieve a specific user's LoRA adapter metadata.

        Args:
            user_id: The unique user identifier.

        Returns:
            Success envelope with user_id, exists flag, and feedback_count
            if the adapter exists.

        Raises:
            503 if the per-user LoRA module is not available.
        """
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            adapter = store.get_adapter(user_id)
            if adapter is None:
                return success_response(data={"user_id": user_id, "exists": False})
            return success_response(data={"user_id": user_id, "exists": True, "feedback_count": adapter.feedback_count})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.get")

    async def update_adapter(self, user_id: str, req: AdapterUpdateRequest) -> dict:
        """Update a user's LoRA adapter with new feedback rating.

        Args:
            user_id: The unique user identifier.
            req: AdapterUpdateRequest with rating (thumbs_up/thumbs_down/neutral).

        Returns:
            Success envelope with status "updated" and user_id.

        Side effects:
            - Updates the adapter's feedback counter and weight deltas.
            - Writes an audit log entry for the update.

        Raises:
            503 if the per-user LoRA module is not available.
        """
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.update_adapter(user_id, rating=req.rating)
            safe_audit_log("adapter.update", resource=user_id, detail=f"rating={req.rating}")
            return success_response(data={"status": "updated", "user_id": user_id})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.update")

    async def reset_adapter(self, user_id: str) -> dict:
        """Reset a user's LoRA adapter to its initial zero-weight state.

        Clears all accumulated feedback deltas and resets the adapter's
        weight contributions to zero.

        Args:
            user_id: The unique user identifier.

        Returns:
            Success envelope with status "reset" and user_id.

        Side effects:
            - Clears the adapter's weight deltas in the store.
            - Writes an audit log entry for the reset.

        Raises:
            503 if the per-user LoRA module is not available.
        """
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.reset_user_adapter(user_id)
            safe_audit_log("adapter.reset", resource=user_id)
            return success_response(data={"status": "reset", "user_id": user_id})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.reset")

    async def merge_adapters(self) -> dict:
        """Merge all per-user LoRA adapters into a single combined adapter.

        Weight-deltas from all users are averaged and applied to produce
        a single merged adapter that represents the aggregate user feedback.

        Returns:
            Success envelope with status "merged".

        Side effects:
            - Combines all adapter weights in the store.
            - Writes an audit log entry for the merge.

        Raises:
            503 if the per-user LoRA module is not available.
        """
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.merge_all()
            safe_audit_log("adapter.merge", resource="all")
            return success_response(data={"status": "merged"})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.merge")

    async def aggregate_best(self, req: AggregateBestRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Aggregate top-k best user adapters with auto-evaluation."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            result = store.aggregate_best_adapters(
                top_k=req.top_k,
                min_feedback_count=req.min_feedback_count,
                output_name=req.output_name,
            )
            safe_audit_log("adapter.aggregate", resource=req.output_name or "best_aggregated", user_count=result.get("user_count", 0), total_feedback=result.get("total_feedback", 0))
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
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.aggregate")

    async def get_quality(self, min_feedback_count: int = 3, max_age_days: Optional[int] = None) -> dict:
        """Retrieve quality metrics report for all eligible adapters.

        Filters adapters by minimum feedback count and maximum age, then
        computes per-adapter quality scores based on feedback patterns.

        Args:
            min_feedback_count: Minimum number of feedback entries to include
                an adapter (default 3).
            max_age_days: Optional maximum age in days; adapters older than
                this are excluded.

        Returns:
            Success adapter quality metrics per adapter including
            score, feedback_count, and rating distribution.

        Raises:
            503 if the per-user LoRA module is not available.
        """
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
            raise_error("Per-user LoRA module not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.quality")

    async def delete_user_adapter(self, user_id: str, req: Request, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a user's LoRA adapter."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            store.delete_adapter(user_id)
            safe_audit_log("adapter.delete", resource=user_id)
            return success_response(data={"status": "deleted", "user_id": user_id})
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.delete")

    async def prune_low_quality_adapters(self, request: PruneAdaptersRequest, req: Request) -> dict:
        """Remove adapters with too few feedback or too old."""
        try:
            from domains.feedback import get_per_user_lora
            store = get_per_user_lora()
            deleted = store.prune_low_quality(
                min_feedback_count=request.min_feedback_count,
                max_age_days=request.max_age_days,
            )
            safe_audit_log("adapter.prune", resource="all", detail=f"deleted={len(deleted)}", deleted_users=deleted)
            return success_response(data={
                "status": "pruned",
                "deleted_count": len(deleted),
                "deleted_users": deleted,
            })
        except ImportError:
            raise_error("Per-user LoRA not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            classify_and_raise(exc, source="user_adapters.prune")


router = UserAdaptersRouter().router
