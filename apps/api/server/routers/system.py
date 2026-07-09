"""
System Router - Host metrics, system information, lifecycle status, and output stream.
"""
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
import asyncio
import psutil
import platform
import time

from schemas.common import success_response

router = APIRouter(prefix="/system", tags=["system"])

_metrics_cache = {"data": None, "ts": 0.0}
_METRICS_TTL = 2.0  # seconds


@router.get("/metrics")
async def get_metrics():
    """Get system metrics (cached for 2s)"""
    now = time.monotonic()
    if _metrics_cache["data"] is None or (now - _metrics_cache["ts"]) > _METRICS_TTL:
        _metrics_cache["data"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        _metrics_cache["ts"] = now
    return success_response(data=_metrics_cache["data"])


@router.get("/info")
async def get_info():
    """Get system info"""
    return success_response(data={
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
    })


@router.get("/disk")
async def get_disk():
    """Get disk usage"""
    disk = psutil.disk_usage('/')
    return success_response(data={
        "total_gb": disk.total / (1024**3),
        "used_gb": disk.used / (1024**3),
        "free_gb": disk.free / (1024**3),
        "percent": disk.percent,
    })


@router.get("/lifecycle")
async def get_lifecycle_status():
    """Get the current lifecycle manager state.

    Returns the lifecycle phase, active startup profile, uptime,
    in-flight request count, hook statistics, health gate status,
    and a preview of which hooks the active profile would run.
    """
    try:
        from domains.infrastructure.lifecycle import get_lifecycle_manager
        mgr = get_lifecycle_manager()
        return success_response(data=mgr.get_results())
    except Exception as exc:
        return success_response(data={
            "phase": "unavailable",
            "profile": "unknown",
            "error": str(exc),
        })


# ── Output stream ───────────────────────────────────────────────────────

@router.get("/stream")
async def stream_output(request: Request, tail: int = Query(50, ge=0, le=500)):
    """SSE stream of all server output (logs, training progress, etc.).

    Sends recent history first, then streams new lines as they arrive.
    Each event: {"text": "...", "level": "info|error|warning", "source": "...", "ts": 1234.5}
    """
    from domains.infrastructure.output_buffer import get_server_buffer

    buf = get_server_buffer()
    sub = buf.subscribe("http-" + str(id(request)))

    async def generate():
        try:
            for line in buf.tail(tail):
                yield f"data: {line.to_sse()}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                lines = sub.read(timeout=0.2)
                for line in lines:
                    yield f"data: {line.to_sse()}\n\n"
                await asyncio.sleep(0)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            buf.unsubscribe(sub.name)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/output")
async def tail_output(n: int = Query(100, ge=1, le=1000)):
    """Get last N lines of server output."""
    from domains.infrastructure.output_buffer import get_server_buffer
    buf = get_server_buffer()
    return success_response(data={"lines": buf.tail_dicts(n), "size": buf.count, "seq": buf.seq})
