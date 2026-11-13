"""Training control endpoints — status, start, pause, resume, stop, reset, is-running.

Descriptor-projected (card 20261104_151036): seven ``ToolSpec`` contracts →
``create_router`` projections instead of hand-laid decorators. Raw wire — the
handlers' dicts are served unwrapped (``envelope=False``), byte-identical to
the hand-built routes, which published 200 only (no parameters → no 422).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from infrastructure.contract import RouteSpec, create_router

from domain.agents import ToolSpec

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
            from domain.infrastructure.cancel_manager import get_cancel_manager

            get_cancel_manager().cancel(jid)
        except Exception as e:
            logger.warning("CancelManager.cancel failed for %s: %s", jid, e)
    return signaled


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


async def control_resume_training():
    """Resume paused training."""
    controller = get_training_controller()
    result = controller.resume()

    if result["success"]:
        signaled = _signal_current_job(pause=False)
        logger.info("Training resumed: %s", signaled or "no tracked job", extra={"tag": "TRAIN"})

    return result


async def control_stop_training():
    """Stop current training."""
    controller = get_training_controller()
    result = controller.stop()

    if result["success"]:
        for _jid, job in training_jobs.items():
            if job.get("status") == "running":
                job["status"] = "stopping"
        signaled = _signal_current_job(cancel=True)
        logger.info(
            "Training stop requested: %s", signaled or "no tracked job", extra={"tag": "TRAIN"}
        )

    return result


async def control_reset_training():
    """Reset training controller to idle state."""
    controller = get_training_controller()
    return controller.reset()


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


# ── Descriptor projections (card 20261104_151036) ──────────────────────
# One ToolSpec per capability; the module router keeps the tags the hand-built
# decorators rode, so the served tag list stays byte-identical.

SPEC_STATUS = ToolSpec(
    name="training.control.status",
    description="Get comprehensive training system status.",
    parameters=[],
    execute=get_training_status,
    idempotent=True,
    version="1",
)

SPEC_START = ToolSpec(
    name="training.control.start",
    description="Request to start training.",
    parameters=[],
    execute=control_start_training,
    idempotent=False,
    version="1",
)

SPEC_PAUSE = ToolSpec(
    name="training.control.pause",
    description="Pause current training.",
    parameters=[],
    execute=control_pause_training,
    idempotent=False,
    version="1",
)

SPEC_RESUME = ToolSpec(
    name="training.control.resume",
    description="Resume paused training.",
    parameters=[],
    execute=control_resume_training,
    idempotent=False,
    version="1",
)

SPEC_STOP = ToolSpec(
    name="training.control.stop",
    description="Stop current training.",
    parameters=[],
    execute=control_stop_training,
    idempotent=False,
    version="1",
)

SPEC_RESET = ToolSpec(
    name="training.control.reset",
    description="Reset training controller to idle state.",
    parameters=[],
    execute=control_reset_training,
    idempotent=False,
    version="1",
)

SPEC_IS_RUNNING = ToolSpec(
    name="training.control.is_running",
    description="Quick check if training is currently running.",
    parameters=[],
    execute=is_training_running,
    idempotent=True,
    version="1",
)

router.include_router(
    create_router(
        SPEC_STATUS,
        RouteSpec(path="/training/status", envelope=False),
        get_training_status,
    )
)
router.include_router(
    create_router(
        SPEC_START,
        RouteSpec(path="/training/control/start", envelope=False),
        control_start_training,
    )
)
router.include_router(
    create_router(
        SPEC_PAUSE,
        RouteSpec(path="/training/control/pause", envelope=False),
        control_pause_training,
    )
)
router.include_router(
    create_router(
        SPEC_RESUME,
        RouteSpec(path="/training/control/resume", envelope=False),
        control_resume_training,
    )
)
router.include_router(
    create_router(
        SPEC_STOP,
        RouteSpec(path="/training/control/stop", envelope=False),
        control_stop_training,
    )
)
router.include_router(
    create_router(
        SPEC_RESET,
        RouteSpec(path="/training/control/reset", envelope=False),
        control_reset_training,
    )
)
router.include_router(
    create_router(
        SPEC_IS_RUNNING,
        RouteSpec(path="/training/is-running", envelope=False),
        is_training_running,
    )
)
