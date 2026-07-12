"""
Registry Router - Model registry management
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from schemas.common import success_response

router = APIRouter(prefix="/registry", tags=["registry"])

_registry = {}


class ModelRecord(BaseModel):
    model_id: str
    metrics: Optional[dict] = None
    metadata: Optional[dict] = None


@router.get("/models")
async def list_models():
    """List registered models"""
    return success_response(data={"models": list(_registry.values()), "count": len(_registry)})


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail="Model not found")
    return _registry[model_id]


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete model from registry"""
    if model_id in _registry:
        del _registry[model_id]
        return success_response(data={"status": "deleted", "model_id": model_id})
    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/models/{model_id}/record")
async def record_model(model_id: str, record: ModelRecord):
    """Record model metrics"""
    _registry[model_id] = {"model_id": record.model_id, "metrics": record.metrics or {}, "metadata": record.metadata or {}}
    return success_response(data={"status": "recorded", "model_id": model_id})


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get model metrics"""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail="Model not found")
    return success_response(data=_registry[model_id].get("metrics", {}))


@router.get("/best")
async def get_best_model():
    """Get best performing model"""
    if not _registry:
        return success_response(data={"model_id": None})
    best = max(_registry.values(), key=lambda m: m.get("metrics", {}).get("score", 0))
    return success_response(data={"model_id": best["model_id"], "metrics": best.get("metrics", {})})


@router.get("/stats")
async def get_registry_stats():
    """Get registry statistics"""
    return success_response(data={"total_models": len(_registry)})
