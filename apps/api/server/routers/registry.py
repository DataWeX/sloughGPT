"""
Registry Router - Proxies to the real ModelRegistry (domains.infrastructure.model_registry).

Previously used an in-memory dict that lost data on restart and was disconnected
from the actual model serving layer. Now delegates to get_model_registry() so all
registry operations reflect the real state of loaded models.
"""
from fastapi import APIRouter, HTTPException
from schemas.common import success_response

router = APIRouter(prefix="/registry", tags=["registry"])


def _get_registry():
    from domains.infrastructure.model_registry import get_model_registry
    return get_model_registry()


@router.get("/models")
async def list_models():
    """List registered models from the live registry."""
    reg = _get_registry()
    models = reg.list_models()
    return success_response(data={"models": models, "count": len(models)})


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model details from the live registry."""
    reg = _get_registry()
    models = reg.list_models()
    found = next((m for m in models if m.get("model_id") == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")
    return success_response(data=found)


@router.get("/best")
async def get_best_model():
    """Get best performing model by metrics."""
    reg = _get_registry()
    health = reg.health_summary()
    return success_response(data=health)


@router.get("/stats")
async def get_registry_stats():
    """Get registry statistics."""
    reg = _get_registry()
    health = reg.health_summary()
    return success_response(data=health)
