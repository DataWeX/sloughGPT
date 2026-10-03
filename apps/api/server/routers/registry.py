"""
Registry Router - Proxies to the real ModelRegistry (domains.infrastructure.model_registry).

Previously used an in-memory dict that lost data on restart and was disconnected
from the actual model serving layer. Now delegates to get_model_registry() so all
registry operations reflect the real state of loaded models.
"""

from __future__ import annotations

import time as _time

from fastapi import APIRouter, Query
from schemas.common import (
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)


class RegistryRouter:
    """Registry Router - Proxies to the real ModelRegistry."""

    def __init__(self):
        self.router = APIRouter(prefix="/registry", tags=["registry"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/models", endpoint=self.list_models, methods=["GET"])
        self.router.add_api_route(
            path="/models/{model_id}", endpoint=self.get_model, methods=["GET"]
        )
        self.router.add_api_route(path="/best", endpoint=self.get_best_model, methods=["GET"])
        self.router.add_api_route(path="/stats", endpoint=self.get_registry_stats, methods=["GET"])
        self.router.add_api_route(path="/artifacts", endpoint=self.list_artifacts, methods=["GET"])
        self.router.add_api_route(path="/verify", endpoint=self.verify_artifacts, methods=["GET"])

    def _get_registry(self):
        from domain.infrastructure.model_registry import get_model_registry

        return get_model_registry()

    @endpoint("registry.list")
    async def list_models(self) -> dict:
        """List registered models from the live registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        models = reg.list_models()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.list",
            resource="models",
            detail=f"elapsed={_elapsed_ms:.0f}ms count={len(models)}",
        )
        return success_response(data={"models": models, "count": len(models)})

    @endpoint("registry.get")
    async def get_model(self, model_id: str) -> dict:
        """Get model details from the live registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        models = reg.list_models()
        found = next((m for m in models if m.get("model_id") == model_id), None)
        if not found:
            raise_error("Model not found", "E_NOT_FOUND", status_code=404)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.get", resource=model_id, detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=found)

    @endpoint("registry.best")
    async def get_best_model(self) -> dict:
        """Get best performing model by metrics."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        health = reg.health_summary()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.best", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=health)

    @endpoint("registry.stats")
    async def get_registry_stats(self) -> dict:
        """Retrieve aggregate statistics for the model registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        health = reg.health_summary()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.stats", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=health)

    @endpoint("registry.artifacts")
    async def list_artifacts(
        self,
        kind: str | None = Query(None, description="Filter by kind"),
        q: str | None = Query(None, description="Substring search on id/name"),
    ) -> dict:
        """List artifacts across all kinds (Stage 1: read-only index)."""
        import asyncio as _aio

        from domain.infrastructure._internal import artifact_registry as _ar

        if kind is not None and kind not in _ar.ARTIFACT_KINDS:
            raise_error(
                f"Unknown kind {kind!r} (expected one of {', '.join(_ar.ARTIFACT_KINDS)})",
                "E_BAD_REQUEST",
                status_code=400,
            )
        _t0 = _time.monotonic()
        artifacts = await _aio.to_thread(_ar.list_artifacts, kind, q)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.artifacts",
            resource=kind or "all",
            detail=f"elapsed={_elapsed_ms:.0f}ms count={len(artifacts)}",
        )
        return success_response(data={"artifacts": artifacts, "count": len(artifacts)})

    @endpoint("registry.verify")
    async def verify_artifacts(
        self,
        kind: str | None = Query(None, description="Filter by kind"),
        q: str | None = Query(None, description="Substring search on id/name"),
    ) -> dict:
        """Hash/size integrity check over artifacts (read-only, hashed in thread)."""
        import asyncio as _aio

        from domain.infrastructure._internal import artifact_registry as _ar

        if kind is not None and kind not in _ar.ARTIFACT_KINDS:
            raise_error(
                f"Unknown kind {kind!r} (expected one of {', '.join(_ar.ARTIFACT_KINDS)})",
                "E_BAD_REQUEST",
                status_code=400,
            )
        _t0 = _time.monotonic()
        report = await _aio.to_thread(_ar.verify, kind, q)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.verify",
            resource=kind or "all",
            detail=f"elapsed={_elapsed_ms:.0f}ms summary={report['summary']}",
        )
        return success_response(data=report)


router = RegistryRouter().router

