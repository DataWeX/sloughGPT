"""ModelStack — one-time model + overlay stacks (knowledge/dataset/cache)."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, endpoint, success_response

from domain.infrastructure.model_stack import ModelBase, ModelStack, StackLayer
from domain.training import get_cache_root

logger = logging.getLogger(__name__)

# In-memory singleton per process (persist optionally via file)
_stack: ModelStack | None = None


def _get_stack() -> ModelStack:
    global _stack
    if _stack is None:
        # Default base: first model in cache or dummy
        cache_models = Path.home() / ".cache" / "sloughgpt" / "models"
        dummy = ModelBase(model_id="base", path=cache_models / "base.soul")
        _stack = ModelStack(dummy)
    return _stack


class ModelStackRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/model-stack", tags=["model-stack"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.get_stack, methods=["GET"])
        self.router.add_api_route("/base", self.set_base, methods=["POST"])
        self.router.add_api_route("/push", self.push_layer, methods=["POST"])
        self.router.add_api_route("/{name}", self.remove_layer, methods=["DELETE"])
        self.router.add_api_route("/clear", self.clear_layers, methods=["POST"])

    @endpoint("model_stack.get_stack")
    async def get_stack(self) -> dict:
        try:
            stack = _get_stack()
            return success_response(data=stack.to_dict())
        except Exception as e:
            classify_and_raise(e, source="model_stack.get_stack")

    @endpoint("model_stack.set_base")
    async def set_base(
        self, payload: dict, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Set one-time base model. File is chmod 444 and never deleted."""
        try:
            model_id = payload.get("model_id") or payload.get("name") or "base"
            path = Path(payload.get("path") or payload.get("model_path") or "")
            if not path:
                from domain.shared import find_repo_root

                path = find_repo_root() / "models" / f"{model_id}.soul"
            base = ModelBase(model_id=model_id, path=path, sha256=payload.get("sha256"))
            global _stack
            # Preserve existing layers when swapping base
            layers = _get_stack()._layers if _stack else []
            _stack = ModelStack(base)
            for l in layers:
                try:
                    _stack.push(l)
                except Exception:
                    pass
            return success_response(data=_stack.to_dict())
        except Exception as e:
            classify_and_raise(e, source="model_stack.set_base")

    @endpoint("model_stack.push_layer")
    async def push_layer(self, payload: dict) -> dict:
        """Push a knowledge/dataset/cache layer. Body: {kind, name, path, workspace_id}."""
        try:
            kind = payload.get("kind", "knowledge")
            name = payload["name"]
            path = Path(payload["path"])
            # Resolve cache-relative paths: if name exists in cache, use that
            if not path.exists():
                cand = get_cache_root() / name
                if cand.exists():
                    path = cand
            layer = StackLayer(
                kind=kind, name=name, path=path, workspace_id=payload.get("workspace_id", "")
            )
            _get_stack().push(layer)
            return success_response(data=_get_stack().to_dict())
        except Exception as e:
            classify_and_raise(e, source="model_stack.push_layer")

    @endpoint("model_stack.remove_layer")
    async def remove_layer(self, name: str) -> dict:
        try:
            ok = _get_stack().remove(name)
            return success_response(data={"removed": ok, "stack": _get_stack().to_dict()})
        except Exception as e:
            classify_and_raise(e, source="model_stack.remove_layer")

    @endpoint("model_stack.clear_layers")
    async def clear_layers(self, payload: dict | None = None) -> dict:
        try:
            kind = (payload or {}).get("kind")
            n = _get_stack().clear(kind)
            return success_response(data={"cleared": n, "stack": _get_stack().to_dict()})
        except Exception as e:
            classify_and_raise(e, source="model_stack.clear_layers")


router = ModelStackRouter().router
