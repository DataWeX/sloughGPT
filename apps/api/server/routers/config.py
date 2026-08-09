"""
Config Router - MVC View layer
"""
from fastapi import APIRouter

from schemas.config import ConfigUpdate
from controllers.config import get_config_controller
from schemas.common import success_response


class ConfigRouter:
    """Router for application configuration."""

    def __init__(self):
        self.router = APIRouter(prefix="/config", tags=["config"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/generation", self.get_generation_config, methods=["GET"])
        self.router.add_api_route("/generation", self.update_generation_config, methods=["PUT", "PATCH"])

    async def get_generation_config(self):
        """Get current generation config"""
        ctrl = get_config_controller()
        return success_response(data=ctrl.get_generation_config())

    async def update_generation_config(self, req: ConfigUpdate):
        """Update generation config"""
        ctrl = get_config_controller()
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        result = ctrl.update_generation_config(**updates)
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "config.generation.save",
                resource="generation",
                detail=str(updates),
                extra={k: str(v) for k, v in updates.items()},
            )
        except Exception:
            pass
        return success_response(data=result)


router = ConfigRouter().router
