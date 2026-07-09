"""
User Adapters Router - Per-user LoRA adapter management
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from schemas.common import success_response

router = APIRouter(prefix="/user-adapters", tags=["user-adapters"])


class AggregateBestRequest(BaseModel):
    top_k: Optional[int] = 10
    min_feedback_count: Optional[int] = 5
    output_name: Optional[str] = "best_aggregated"


class AdapterUpdateRequest(BaseModel):
    rating: str
    quality_score: Optional[float] = None


@router.get("")
async def list_adapters():
    """List all user adapters"""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        adapters = store.get_all_adapters()
        stats = store.get_stats()
        return success_response(data={"adapters": adapters, "stats": stats})
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@router.get("/{user_id}")
async def get_adapter(user_id: str):
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


@router.post("/{user_id}/update")
async def update_adapter(user_id: str, req: AdapterUpdateRequest):
    """Update user's LoRA adapter"""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        store.update_adapter(user_id, rating=req.rating)
        return success_response(data={"status": "updated", "user_id": user_id})
    except ImportError:
        raise HTTPException(status_code=503, detail="Not available")


@router.post("/{user_id}/reset")
async def reset_adapter(user_id: str):
    """Reset user's adapter"""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        store.reset_adapter(user_id)
        return success_response(data={"status": "reset", "user_id": user_id})
    except ImportError:
        raise HTTPException(status_code=503, detail="Not available")


@router.post("/merge")
async def merge_adapters():
    """Merge all adapters"""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        store.merge_all()
        return success_response(data={"status": "merged"})
    except ImportError:
        raise HTTPException(status_code=503, detail="Not available")


@router.post("/aggregate-best")
async def aggregate_best(req: AggregateBestRequest):
    """Aggregate top-k best user adapters with auto-evaluation."""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        result = store.aggregate_best_adapters(
            top_k=req.top_k,
            min_feedback_count=req.min_feedback_count,
            output_name=req.output_name,
        )
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


@router.get("/quality")
async def get_quality():
    """Get adapter quality metrics"""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        return success_response(data=store.get_quality_report())
    except ImportError:
        raise HTTPException(status_code=503, detail="Not available")


class PruneAdaptersRequest(BaseModel):
    min_feedback_count: int = 1
    max_age_days: int = 30


@router.delete("/{user_id}")
async def delete_user_adapter(user_id: str, req: Request):
    """Delete a user's LoRA adapter."""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        store.delete_adapter(user_id)
        return success_response(data={"status": "deleted", "user_id": user_id})
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@router.post("/prune")
async def prune_low_quality_adapters(request: PruneAdaptersRequest, req: Request):
    """Remove adapters with too few feedback or too old."""
    try:
        from domains.feedback import get_per_user_lora
        store = get_per_user_lora()
        deleted = store.prune_low_quality(
            min_feedback_count=request.min_feedback_count,
            max_age_days=request.max_age_days,
        )
        return success_response(data={
            "status": "pruned",
            "deleted_count": len(deleted),
            "deleted_users": deleted,
        })
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")
