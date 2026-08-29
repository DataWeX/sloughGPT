"""
Registry Router - Proxies to the real ModelRegistry (domains.infrastructure.model_registry).

Previously used an in-memory dict that lost data on restart and was disconnected
from the actual model serving layer. Now delegates to get_model_registry() so all
registry operations reflect the real state of loaded models.
"""
from fastapi import APIRouter
from schemas.common import raise_error, success_response, safe_audit_log, classify_and_raise
import time as _time


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

    async def list_models(self) -> dict:
        """List registered models from the live registry."""
        try:
            _t0 = _time.monotonic()
            reg = self._get_registry()
            models = reg.list_models()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("registry.list", resource="models", detail=f"elapsed={_elapsed_ms:.0f}ms count={len(models)}")
            return success_response(data={"models": models, "count": len(models)})
        except Exception as e:
            classify_and_raise(e, source="registry.list")

    async def get_model(self, model_id: str) -> dict:
        """Get model details from the live registry."""
        try:
            _t0 = _time.monotonic()
            reg = self._get_registry()
            models = reg.list_models()
            found = next((m for m in models if m.get("model_id") == model_id), None)
            if not found:
                raise_error("Model not found", "E_NOT_FOUND", status_code=404)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("registry.get", resource=model_id, detail=f"elapsed={_elapsed_ms:.0f}ms")
            return success_response(data=found)
        except Exception as e:
            classify_and_raise(e, source="registry.get")

    async def get_best_model(self) -> dict:
        """Get best performing model by metrics."""
        try:
            _t0 = _time.monotonic()
            reg = self._get_registry()
            health = reg.health_summary()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("registry.best", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
            return success_response(data=health)
        except Exception as e:
            classify_and_raise(e, source="registry.best")

    async def get_registry_stats(self) -> dict:
        """Retrieve aggregate statistics for the model registry."""
        try:
            _t0 = _time.monotonic()
            reg = self._get_registry()
            health = reg.health_summary()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("registry.stats", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
            return success_response(data=health)
        except Exception as e:
            classify_and_raise(e, source="registry.stats")


router = RegistryRouter().router
