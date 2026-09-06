"""Training control endpoints — status, start, pause, resume, stop, reset, is-running."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from .controller import get_training_controller
from .jobs import training_jobs

logger = logging.getLogger("slo")

router = APIRouter(tags=["training", "control"])


def _signal_current_job(pause: bool | None = None, cancel: bool = False) -> dict[str, Any]:
    """Signal the controller's current job with cooperative control events."""
    controller = get_training_controller()
    jid = getattr(controller, "current_job_id", None)
    if not jid:
        return {}
    job = training_jobs.get(jid)
    if not job:
        return {}
    signaled: dict[str, Any] = {}
    if pause is not None:
        ev = job.get("_pause_event")
        if ev is not None:
            if pause:
                ev.set()
                signaled["pause"] = "requested"
            else:
                ev.clear()
                signaled["resume"] = "requested"
    if cancel:
        ev = job.get("_cancel_event")
        if ev is not None:
            ev.set()
            signaled["cancel"] = "requested"
        try:
            from domains.infrastructure.cancel_manager import get_cancel_manager

            get_cancel_manager().cancel(jid)
        except Exception as e:
            logger.warning("CancelManager.cancel failed for %s: %s", jid, e)
    return signaled


@router.get("/training/status")
async def get_training_status():
    """Get comprehensive training system status."""
    controller = get_training_controller()
    status = controller.get_status()

    running_jobs = [
        {"id": jid, "name": job.get("name"), "progress": job.get("progress", 0)}
        for jid, job in training_jobs.items()
        if job.get("status") == "running"
    ]

    status["running_jobs"] = running_jobs
    status["total_tracked_jobs"] = len(training_jobs)

    return status


@router.post("/training/control/start")
async def control_start_training():
    """Request to start training."""
    controller = get_training_controller()

    if controller.is_running():
        return {
            "success": False,
            "message": "Training is already running",
            **controller.get_status(),
        }

    if controller.is_paused():
        return {
            "success": False,
            "message": "Training is paused. Use /training/control/resume to continue.",
            **controller.get_status(),
        }

    return {
        "success": True,
        "message": "Ready to start training",
        **controller.get_status(),
    }


@router.post("/training/control/pause")
async def control_pause_training():
    """Pause current training."""
    controller = get_training_controller()
    result = controller.pause()

    if result["success"]:
        signaled = _signal_current_job(pause=True)
        logger.info(
            "Training pause requested: %s", signaled or "no tracked job", extra={"tag": "TRAIN"}
        )

    return result


@router.post("/training/control/resume")
async def control_resume_training():
    """Resume paused training."""
    controller = get_training_controller()
    result = controller.resume()

    if result["success"]:
        signaled = _signal_current_job(pause=False)
        logger.info("Training resumed: %s", signaled or "no tracked job", extra={"tag": "TRAIN"})

    return result


@router.post("/training/control/stop")
async def control_stop_training():
    """Stop current training."""
    controller = get_training_controller()
    result = controller.stop()

    if result["success"]:
        for jid, job in training_jobs.items():
            if job.get("status") == "running":
                job["status"] = "stopping"
        signaled = _signal_current_job(cancel=True)
        logger.info(
            "Training stop requested: %s", signaled or "no tracked job", extra={"tag": "TRAIN"}
        )

    return result


@router.post("/training/control/reset")
async def control_reset_training():
    """Reset training controller to idle state."""
    controller = get_training_controller()
    return controller.reset()


@router.get("/training/is-running")
async def is_training_running():
    """Quick check if training is currently running."""
    controller = get_training_controller()
    return {
        "is_running": controller.is_running(),
        "is_paused": controller.is_paused(),
        "is_idle": controller.is_idle(),
        "state": controller.state.value,
        "current_job": controller.current_job_id,
    }
