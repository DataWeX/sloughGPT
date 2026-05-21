"""
Workflow Router - Background task management

Delegates to the canonical FeedbackWorkflowManager from the feedback domain.
"""
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/workflow", tags=["workflow"])


class WorkflowStartRequest(BaseModel):
    aggregate_interval_minutes: int = 60
    prune_interval_minutes: int = 120
    export_interval_hours: int = 24
    health_check_interval_seconds: int = 30


def _get_workflow():
    """Lazy-import and return the FeedbackWorkflowManager singleton."""
    try:
        from domains.feedback import get_feedback_workflow
        return get_feedback_workflow()
    except ImportError:
        raise HTTPException(status_code=503, detail="Workflow module not available")


@router.get("/status")
async def get_workflow_status() -> Dict[str, Any]:
    """Get current workflow status and statistics."""
    return _get_workflow().get_status()


@router.post("/start")
async def start_workflow(request: WorkflowStartRequest) -> Dict[str, Any]:
    """Start the automated feedback workflow."""
    from domains.feedback import WorkflowConfig
    config = WorkflowConfig(
        aggregate_interval_minutes=request.aggregate_interval_minutes,
        prune_interval_minutes=request.prune_interval_minutes,
        export_interval_hours=request.export_interval_hours,
        health_check_interval_seconds=request.health_check_interval_seconds,
    )
    workflow = _get_workflow()
    workflow.start()
    return {"status": "started", "config": request.model_dump()}


@router.post("/stop")
async def stop_workflow() -> Dict[str, Any]:
    """Stop the automated feedback workflow."""
    workflow = _get_workflow()
    workflow.stop()
    return {"status": "stopped"}


@router.post("/trigger/{action}")
async def trigger_workflow(action: str) -> Dict[str, Any]:
    """Manually trigger a workflow action (aggregate, prune, export)."""
    workflow = _get_workflow()
    if action == "aggregate":
        return workflow.trigger_aggregate()
    elif action == "prune":
        return workflow.trigger_prune()
    elif action == "export":
        return workflow.trigger_export()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")