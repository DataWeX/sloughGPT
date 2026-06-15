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


@router.get("/debug")
async def debug_info():
    """Lightweight debug snapshot for the frontend debug overlay.

    Returns a compact subset of /health/detailed suitable for frequent polling.
    Includes request/error history, path latencies, inference metrics, and request rate.
    """
    ctrl = get_health_controller()
    detailed = ctrl.get_detailed_health()
    return {
        "model_loaded": detailed.get("model_loaded", False),
        "model_type": detailed.get("model_type"),
        "soul": detailed.get("soul"),
        "uptime_seconds": detailed.get("uptime_seconds", 0),
        "request_count": detailed.get("request_count", 0),
        "error_count": detailed.get("error_count", 0),
        "inference_count": detailed.get("inference_count", 0),
        "total_tokens": detailed.get("total_tokens", 0),
        "tokens_per_sec": detailed.get("tokens_per_sec", 0),
        "avg_tokens_per_request": detailed.get("avg_tokens_per_request", 0),
        "avg_latency_ms": detailed.get("avg_latency_ms", 0),
        "requests_per_minute": detailed.get("requests_per_minute", 0),
        "health_score": detailed.get("health_score", {}),
        "model_metrics": detailed.get("model_metrics", []),
        "model_events": detailed.get("model_events", []),
        "health_history": detailed.get("health_history", []),
        "memory_history": detailed.get("memory_history", []),
        "rate_violations": detailed.get("rate_violations", []),
        "path_latencies": detailed.get("path_latencies", []),
        "recent_errors": detailed.get("recent_errors", []),
        "cpu_percent": detailed.get("system", {}).get("cpu_percent"),
        "memory_percent": detailed.get("system", {}).get("memory_percent"),
        "gpu_backend": detailed.get("gpu", {}).get("backend"),
    }


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


@router.get("/summary")
async def health_summary():
    """Compact health summary for the status bar.

    Returns only what the UI needs: score, status, model, soul, key counters.
    Lightweight — no history arrays, no path breakdowns.
    """
    ctrl = get_health_controller()
    detailed = ctrl.get_detailed_health()
    hs = detailed.get("health_score", {})
    return {
        "score": hs.get("score", 0),
        "status": hs.get("status", "unknown"),
        "model_loaded": detailed.get("model_loaded", False),
        "model_type": detailed.get("model_type"),
        "soul": detailed.get("soul"),
        "uptime_seconds": detailed.get("uptime_seconds", 0),
        "request_count": detailed.get("request_count", 0),
        "error_count": detailed.get("error_count", 0),
        "tokens_per_sec": detailed.get("tokens_per_sec", 0),
        "cpu_percent": detailed.get("system", {}).get("cpu_percent"),
        "memory_percent": detailed.get("system", {}).get("memory_percent"),
    }