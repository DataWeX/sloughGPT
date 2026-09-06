"""
Workflow Router - Background task management

Delegates to the canonical FeedbackWorkflowManager from the feedback domain.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.api.workflow")


class WorkflowStartRequest(BaseModel):
    aggregate_interval_minutes: int = Field(default=60, ge=1, le=10080)
    prune_interval_minutes: int = Field(default=120, ge=1, le=10080)
    export_interval_hours: int = Field(default=24, ge=1, le=720)
    health_check_interval_seconds: int = Field(default=30, ge=5, le=3600)


class WorkflowRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/workflow", tags=["workflow"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/status", self.get_workflow_status, methods=["GET"])
        self.router.add_api_route("/start", self.start_workflow, methods=["POST"])
        self.router.add_api_route("/stop", self.stop_workflow, methods=["POST"])
        self.router.add_api_route("/trigger/{action}", self.trigger_workflow, methods=["POST"])

    def _get_workflow(self):
        """Lazy-import and return the FeedbackWorkflowManager singleton."""
        try:
            from domains.feedback import get_feedback_workflow

            return get_feedback_workflow()
        except ImportError:
            raise_error("Workflow module not available", "E_BAD_REQUEST", status_code=503)
        except Exception as exc:
            logger.warning("Workflow init failed: %s", exc)
            classify_and_raise(exc, source="workflow")

    async def get_workflow_status(self) -> dict[str, Any]:
        """Get current workflow status and statistics."""
        try:
            return success_response(data=self._get_workflow().get_status())
        except Exception as e:
            classify_and_raise(e, source="workflow.status")

    async def start_workflow(
        self, request: WorkflowStartRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Start the automated feedback workflow."""
        try:
            from domains.feedback import WorkflowConfig

            config = WorkflowConfig(
                aggregate_interval_minutes=request.aggregate_interval_minutes,
                prune_interval_minutes=request.prune_interval_minutes,
                export_interval_hours=request.export_interval_hours,
                health_check_interval_seconds=request.health_check_interval_seconds,
            )
            workflow = self._get_workflow()
            workflow.config = config
            workflow.start()
            safe_audit_log(
                "workflow.start",
                detail=f"aggregate={request.aggregate_interval_minutes}m prune={request.prune_interval_minutes}m export={request.export_interval_hours}h",
            )
            logger.info(
                "Workflow started (aggregate=%dm, prune=%dm, export=%dh)",
                request.aggregate_interval_minutes,
                request.prune_interval_minutes,
                request.export_interval_hours,
            )
            return success_response(data={"status": "started", "config": request.model_dump()})
        except Exception as e:
            classify_and_raise(e, source="workflow.start")

    async def stop_workflow(
        self, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Stop the automated feedback workflow."""
        try:
            workflow = self._get_workflow()
            workflow.stop()
            safe_audit_log("workflow.stop")
            logger.info("Workflow stopped")
            return success_response(data={"status": "stopped"})
        except Exception as e:
            classify_and_raise(e, source="workflow.stop")

    async def trigger_workflow(
        self, action: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Manually trigger a workflow action (aggregate, prune, export)."""
        try:
            workflow = self._get_workflow()
            if action == "aggregate":
                logger.info("Workflow action triggered: aggregate")
                safe_audit_log("workflow.trigger", resource="aggregate")
                return workflow.trigger_aggregate()
            elif action == "prune":
                logger.info("Workflow action triggered: prune")
                safe_audit_log("workflow.trigger", resource="prune")
                return workflow.trigger_prune()
            elif action == "export":
                logger.info("Workflow action triggered: export")
                safe_audit_log("workflow.trigger", resource="export")
                return workflow.trigger_export()
            else:
                raise_error(f"Unknown action: {action}", "E_BAD_REQUEST", status_code=400)
        except Exception as e:
            classify_and_raise(e, source="workflow.trigger")


_workflow_router = WorkflowRouter()
router = _workflow_router.router


def _get_workflow():
    return _workflow_router._get_workflow()
