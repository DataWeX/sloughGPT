"""
Health Router - MVC View layer

Provides REST health endpoints plus a live SSE stream (GET /health/stream)
that pushes health snapshots every 3 seconds for real-time UI updates.
"""
import asyncio
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from controllers.health import get_health_controller
from startup_progress import STARTUP_PHASE
from schemas.common import success_response

router = APIRouter(prefix="/health", tags=["health"])

HEALTH_STREAM_INTERVAL = 3.0


@router.get("")
async def health():
    """Basic health check"""
    ctrl = get_health_controller()
    return success_response(data=ctrl.get_basic_health())


@router.get("/live")
async def liveness():
    """Kubernetes liveness probe"""
    ctrl = get_health_controller()
    return success_response(data=ctrl.get_liveness())


@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe"""
    ctrl = get_health_controller()
    return success_response(data=ctrl.get_readiness())


@router.get("/detailed")
async def detailed_health():
    """Detailed health with system metrics"""
    ctrl = get_health_controller()
    return success_response(data=ctrl.get_detailed_health())


@router.get("/startup-progress")
async def startup_progress():
    """Return current server startup phase so the frontend can show
    meaningful progress during the startup sequence."""
    return success_response(data=STARTUP_PHASE)


@router.get("/debug")
async def debug_info():
    """Lightweight debug snapshot for the frontend debug overlay.

    Returns a compact subset of /health/detailed suitable for frequent polling.
    Includes request/error history, path latencies, inference metrics, and request rate.
    """
    ctrl = get_health_controller()
    detailed = ctrl.get_detailed_health()
    return success_response(data={
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
    })


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
        return success_response(data={"status": "ok", **stats})
    except Exception as e:
        return success_response(data={"status": "error", "message": str(e)})


@router.get("/summary")
async def health_summary():
    """Compact health summary for the status bar.

    Returns the flow-based health verdict: score, status, human-readable
    summary, and per-check diagnoses. No raw numbers in user-facing text.
    """
    ctrl = get_health_controller()
    detailed = ctrl.get_detailed_health()
    hs = detailed.get("health_score", {})
    return success_response(data={
        "score": hs.get("score", 0),
        "status": hs.get("status", "unknown"),
        "summary": hs.get("summary", ""),
        "diagnoses": hs.get("diagnoses", []),
        "model_loaded": detailed.get("model_loaded", False),
        "model_loading": detailed.get("model_loading", False),
        "model_type": detailed.get("model_type"),
        "soul": detailed.get("soul"),
        "uptime_seconds": detailed.get("uptime_seconds", 0),
        "request_count": detailed.get("request_count", 0),
        "error_count": detailed.get("error_count", 0),
        "tokens_per_sec": detailed.get("tokens_per_sec", 0),
        "cpu_percent": detailed.get("system", {}).get("cpu_percent"),
        "memory_percent": detailed.get("system", {}).get("memory_percent"),
    })


def _build_health_snapshot(ctrl) -> dict:
    """Build a compact health snapshot for the SSE stream."""
    detailed = ctrl.get_detailed_health()
    basic = ctrl.get_basic_health()
    hs = detailed.get("health_score", {})
    return {
        "stream": "health",
        "phase": "HEALTH",
        "status": "working",
        "data": {
            "model_loaded": detailed.get("model_loaded", False),
            "model_loading": detailed.get("model_loading", False),
            "model_type": detailed.get("model_type"),
            "soul": detailed.get("soul"),
            "is_inferencing": basic.get("is_inferencing", False),
            "inference_count": basic.get("inference_count", 0),
            "uptime_seconds": detailed.get("uptime_seconds", 0),
            "request_count": detailed.get("request_count", 0),
            "error_count": detailed.get("error_count", 0),
            "tokens_per_sec": detailed.get("tokens_per_sec", 0),
            "avg_latency_ms": detailed.get("avg_latency_ms", 0),
            "cpu_percent": detailed.get("system", {}).get("cpu_percent"),
            "memory_percent": detailed.get("system", {}).get("memory_percent"),
            "health_score": hs.get("score", 0),
            "health_status": hs.get("status", "unknown"),
            "health_summary": hs.get("summary", ""),
            "diagnoses": hs.get("diagnoses", []),
            "num_parameters": detailed.get("num_parameters"),
            "quantization": detailed.get("quantization"),
            "training_pool": detailed.get("training_pool"),
        },
        "meta": {"ts": time.time()},
        "message": hs.get("summary", ""),
    }


@router.get("/stream")
async def health_stream(request: Request):
    """SSE stream that pushes health snapshots every 3 seconds.

    Clients receive standard SSE envelopes with stream=health, phase=HEALTH,
    status=working, and a data payload containing the full health snapshot.
    The connection stays open until the client disconnects.
    """
    ctrl = get_health_controller()

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            try:
                snapshot = _build_health_snapshot(ctrl)
                envelope = {
                    "stream": "health",
                    "phase": "HEALTH",
                    "status": "working",
                    "data": snapshot,
                    "meta": {},
                    "message": "",
                }
                yield "data: " + json.dumps(envelope, default=str) + "\n\n"
            except Exception:
                pass
            await asyncio.sleep(HEALTH_STREAM_INTERVAL)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
