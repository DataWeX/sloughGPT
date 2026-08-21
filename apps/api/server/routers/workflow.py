"""
Workflow Router - Background task management

Delegates to the canonical FeedbackWorkflowManager from the feedback domain.
"""
import logging
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter

from schemas.common import raise_error, success_response, classify_and_raise

logger = logging.getLogger("slo.api.workflow")


class WorkflowStartRequest(BaseModel):
    aggregate_interval_minutes: int = 60
    prune_interval_minutes: int = 120
    export_interval_hours: int = 24
    health_check_interval_seconds: int = 30


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
            classify_and_raise(exc, source="workflow")

    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        return success_response(data=self._get_workflow().get_status())

    async def start_workflow(self, request: WorkflowStartRequest) -> Dict[str, Any]:
        """Start the automated feedback workflow."""
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
        logger.info("Workflow started (aggregate=%dm, prune=%dm, export=%dh)", request.aggregate_interval_minutes, request.prune_interval_minutes, request.export_interval_hours)
        return success_response(data={"status": "started", "config": request.model_dump()})

    async def stop_workflow(self) -> Dict[str, Any]:
        """Stop the automated feedback workflow."""
        workflow = self._get_workflow()
        workflow.stop()
        logger.info("Workflow stopped")
        return success_response(data={"status": "stopped"})

    async def trigger_workflow(self, action: str) -> Dict[str, Any]:
        """Manually trigger a workflow action (aggregate, prune, export)."""
        workflow = self._get_workflow()
        if action == "aggregate":
            logger.info("Workflow action triggered: aggregate")
            return workflow.trigger_aggregate()
        elif action == "prune":
            logger.info("Workflow action triggered: prune")
            return workflow.trigger_prune()
        elif action == "export":
            logger.info("Workflow action triggered: export")
            return workflow.trigger_export()
        else:
            raise_error(f"Unknown action: {action}", "E_BAD_REQUEST", status_code=400)


_workflow_router = WorkflowRouter()
router = _workflow_router.router


def _get_workflow():
    return _workflow_router._get_workflow()
