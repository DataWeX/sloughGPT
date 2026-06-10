"""
Health Router - MVC View layer
"""
from fastapi import APIRouter

from schemas.health import (
    HealthResponse, 
    DetailedHealthResponse, 
    LivenessResponse, 
    ReadinessResponse
)
from controllers.health import get_health_controller
from startup_progress import STARTUP_PHASE

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health():
    """Basic health check"""
    ctrl = get_health_controller()
    return ctrl.get_basic_health()


@router.get("/live", response_model=LivenessResponse)
async def liveness():
    """Kubernetes liveness probe"""
    ctrl = get_health_controller()
    return ctrl.get_liveness()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness():
    """Kubernetes readiness probe"""
    ctrl = get_health_controller()
    return ctrl.get_readiness()


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health():
    """Detailed health with system metrics"""
    ctrl = get_health_controller()
    return ctrl.get_detailed_health()


@router.get("/startup-progress")
async def startup_progress():
    """Return current server startup phase so the frontend can show
    meaningful progress during the 90s PyTorch cold-import window."""
    return STARTUP_PHASE


@router.get("/model")
async def model_health():
    """Model health status: perplexity trend, drift detection, benchmark history.

    Uses the ModelHealthMonitor to track model quality over time.
    Requires a loaded HF model with a SloLSTM-based architecture.
    """
    try:
        from domains.feedback.model_health import get_health_monitor
        mon = get_health_monitor()
        import state as server_state
        if server_state.model is not None and mon._model is None:
            mon.set_model(server_state.model, server_state.tokenizer)
        stats = mon.get_stats()
        return {"status": "ok", **stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}