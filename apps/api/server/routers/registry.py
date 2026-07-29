"""
Registry Router - Proxies to the real ModelRegistry (domains.infrastructure.model_registry).

Previously used an in-memory dict that lost data on restart and was disconnected
from the actual model serving layer. Now delegates to get_model_registry() so all
registry operations reflect the real state of loaded models.
"""
from fastapi import APIRouter, HTTPException
from schemas.common import success_response


class RegistryRouter:
    """Registry Router - Proxies to the real ModelRegistry."""

    def __init__(self):
        self.router = APIRouter(prefix="/registry", tags=["registry"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/models", endpoint=self.list_models, methods=["GET"])
        self.router.add_api_route(path="/models/{model_id}", endpoint=self.get_model, methods=["GET"])
        self.router.add_api_route(path="/best", endpoint=self.get_best_model, methods=["GET"])
        self.router.add_api_route(path="/stats", endpoint=self.get_registry_stats, methods=["GET"])

    def _get_registry(self):
        from domains.infrastructure.model_registry import get_model_registry
        return get_model_registry()

    async def list_models(self):
        """List registered models from the live registry."""
        reg = self._get_registry()
        models = reg.list_models()
        return success_response(data={"models": models, "count": len(models)})

    async def get_model(self, model_id: str):
        """Get model details from the live registry."""
        reg = self._get_registry()
        models = reg.list_models()
        found = next((m for m in models if m.get("model_id") == model_id), None)
        if not found:
            raise HTTPException(status_code=404, detail="Model not found")
        return success_response(data=found)

    async def get_best_model(self):
        """Get best performing model by metrics."""
        reg = self._get_registry()
        health = reg.health_summary()
        return success_response(data=health)

    async def get_registry_stats(self):
        """Get registry statistics."""
        reg = self._get_registry()
        health = reg.health_summary()
        return success_response(data=health)


router = RegistryRouter().router
