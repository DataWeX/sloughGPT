"""
Config Router - MVC View layer
"""
from fastapi import APIRouter

from schemas.config import ConfigUpdate
from controllers.config import get_config_controller
from schemas.common import success_response, safe_audit_log


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
        """Return the current generation configuration (temperature, top_p, etc).

        Returns:
            Success envelope containing all generation config fields
            managed by ConfigController (temperature, top_p, top_k,
            max_tokens, repetition_penalty, do_sample).

        Side effects:
            Reads from the ConfigController which may load config from
            disk on first access.
        """
        ctrl = get_config_controller()
        return success_response(data=ctrl.get_generation_config())

    async def update_generation_config(self, req: ConfigUpdate) -> dict:
        """Update the generation configuration with partial field changes.

        Args:
            req: ConfigUpdate with optional temperature, top_p, top_k,
                max_tokens, repetition_penalty, and do_sample fields.
                Only non-None fields are applied.

        Returns:
            Success envelope containing the full updated generation
            config after merging with the current values.

        Side effects:
            Persists the updated config via ConfigController.
            Logs an audit entry with the changed fields and their values.
        """
        ctrl = get_config_controller()
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        result = ctrl.update_generation_config(**updates)
        safe_audit_log("config.generation.save", resource="generation", detail=str(updates), **{k: str(v) for k, v in updates.items()})
        return success_response(data=result)


router = ConfigRouter().router
