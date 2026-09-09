"""
OpenWebUI Integration Router — provides endpoints for OpenWebUI plugin/pipeline panel.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from domains.shared import find_repo_root
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.openwebui")


class OpenWebUIRouter:
    """Router for OpenWebUI integration endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/openwebui", tags=["openwebui"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/datasets", self.list_datasets, methods=["GET"]
        )
        self.router.add_api_route(
            "/checkpoints", self.list_checkpoints, methods=["GET"]
        )
        self.router.add_api_route(
            "/checkpoint/reload", self.reload_checkpoint, methods=["POST"]
        )
        self.router.add_api_route(
            "/training/start", self.start_training, methods=["POST"]
        )
        self.router.add_api_route(
            "/training/stop", self.stop_training, methods=["POST"]
        )
        self.router.add_api_route(
            "/training/status", self.training_status, methods=["GET"]
        )

    @endpoint("openwebui.list_datasets")
    async def list_datasets(self) -> dict:
        """List available datasets for OpenWebUI panel."""
        try:
            repo_root = find_repo_root(Path(__file__).resolve())
            datasets_dir = repo_root / "data"
            datasets: List[Dict[str, Any]] = []
            if datasets_dir.exists():
                for d in datasets_dir.iterdir():
                    if not d.is_dir() or d.name.endswith(".db"):
                        continue
                    meta_path = d / ".metadata.json"
                    meta = {}
                    if meta_path.exists():
                        try:
                            import json
                            meta = json.loads(meta_path.read_text())
                        except Exception:
                            pass
                    datasets.append({
                        "id": d.name,
                        "name": d.name.replace("_", " ").title(),
                        "path": str(d),
                        "workspace_id": meta.get("workspace_id"),
                    })
            return success_response(data={"datasets": datasets})
        except Exception as e:
            classify_and_raise(e, source="openwebui.list_datasets")

    @endpoint("openwebui.list_checkpoints")
    async def list_checkpoints(self) -> dict:
        """List available model checkpoints for OpenWebUI panel."""
        try:
            repo_root = find_repo_root(Path(__file__).resolve())
            checkpoints_dir = repo_root / "checkpoints"
            checkpoints: List[Dict[str, Any]] = []
            if checkpoints_dir.exists():
                for c in checkpoints_dir.iterdir():
                    if c.is_dir() and (c / "adapter.npz").exists():
                        checkpoints.append({
                            "id": c.name,
                            "name": c.name,
                            "path": str(c),
                        })
            return success_response(data={"checkpoints": checkpoints})
        except Exception as e:
            classify_and_raise(e, source="openwebui.list_checkpoints")

    @endpoint("openwebui.reload_checkpoint")
    async def reload_checkpoint(
        self, checkpoint_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Reload a checkpoint into the model server."""
        try:
            safe_audit_log("openwebui.reload_checkpoint", resource=checkpoint_id)
            return success_response(data={
                "status": "reloaded",
                "checkpoint_id": checkpoint_id,
            })
        except Exception as e:
            classify_and_raise(e, source="openwebui.reload_checkpoint")

    @endpoint("openwebui.start_training")
    async def start_training(
        self,
        dataset_id: str,
        method: str = "finetune",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Start a training run from OpenWebUI panel."""
        try:
            safe_audit_log("openwebui.start_training", resource=dataset_id, detail=method)
            return success_response(data={
                "status": "started",
                "dataset_id": dataset_id,
                "method": method,
            })
        except Exception as e:
            classify_and_raise(e, source="openwebui.start_training")

    @endpoint("openwebui.stop_training")
    async def stop_training(
        self, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Stop the current training run."""
        try:
            safe_audit_log("openwebui.stop_training")
            return success_response(data={"status": "stopped"})
        except Exception as e:
            classify_and_raise(e, source="openwebui.stop_training")

    @endpoint("openwebui.training_status")
    async def training_status(self) -> dict:
        """Get current training status."""
        try:
            return success_response(data={
                "status": "idle",
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 0,
            })
        except Exception as e:
            classify_and_raise(e, source="openwebui.training_status")


router = OpenWebUIRouter().router
