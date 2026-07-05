"""
Config Router - MVC View layer
"""
from fastapi import APIRouter

from schemas.config import GenerationConfig, ConfigUpdate
from controllers.config import get_config_controller

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/generation")
async def get_generation_config():
    """Get current generation config"""
    ctrl = get_config_controller()
    return ctrl.get_generation_config()


@router.put("/generation")
@router.patch("/generation")
async def update_generation_config(req: ConfigUpdate):
    """Update generation config"""
    ctrl = get_config_controller()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    return ctrl.update_generation_config(**updates)
