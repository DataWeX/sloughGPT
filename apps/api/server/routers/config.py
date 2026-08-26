"""
Config Router - MVC View layer
"""
import logging
import time as _time
from fastapi import APIRouter, Depends

from schemas.config import ConfigUpdate
from infrastructure.auth import require_auth_if_enabled
from controllers.config import get_config_controller
from schemas.common import success_response, classify_and_raise, safe_audit_log

logger = logging.getLogger("slo.routers.config")


class ConfigRouter:
    """Router for application configuration."""

    def __init__(self):
        self.router = APIRouter(prefix="/config", tags=["config"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/generation", self.get_generation_config, methods=["GET"])
        self.router.add_api_route("/generation", self.update_generation_config, methods=["PUT"], operation_id="update_generation_config_put")
        self.router.add_api_route("/generation", self.update_generation_config, methods=["PATCH"], operation_id="update_generation_config_patch")

    async def get_generation_config(self) -> dict:
        """Return the current generation configuration (temperature, top_p, etc)."""
        try:
            ctrl = get_config_controller()
            result = ctrl.get_generation_config()
            return success_response(data=result)
        except Exception as e:
            classify_and_raise(e, source="config.get_generation")

    async def update_generation_config(self, req: ConfigUpdate) -> dict:
        """Update the generation configuration with partial field changes."""
        try:
            ctrl = get_config_controller()
            updates = {k: v for k, v in req.model_dump().items() if v is not None}
            result = ctrl.update_generation_config(**updates)
            safe_audit_log("config.generation.save", resource="generation", detail=str(updates), **{k: str(v) for k, v in updates.items()})
            return success_response(data=result)
        except Exception as e:
            classify_and_raise(e, source="config.update_generation")


router = ConfigRouter().router
