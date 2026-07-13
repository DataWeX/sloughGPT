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


# ── Training executor ─────────────────────────────────────────────────

@router.get("/executor")
async def get_executor_status():
    """Get TrainingExecutor pool status and job list.

    Returns active/max worker counts, total tracked jobs, and metadata
    for every job (newest first).  Job metadata includes status, timing,
    tree_id, and error if failed.
    """
    from domains.training.executor import _instance
    if _instance is None:
        return success_response(data={
            "initialized": False,
            "active_jobs": 0,
            "max_workers": 0,
            "total_tracked": 0,
            "jobs": [],
        })
    return success_response(data={
        "initialized": True,
        "active_jobs": _instance.active_count(),
        "max_workers": _instance._max_workers,
        "total_tracked": len(_instance._jobs),
        "jobs": _instance.list_jobs(),
    })


@router.get("/executor/{job_id}")
async def get_executor_job(job_id: str):
    """Get metadata for a single training job by ID."""
    from domains.training.executor import _instance
    if _instance is None:
        return success_response(data={"error": "executor not initialized"})
    status = _instance.status(job_id)
    if status is None:
        return success_response(data={"error": f"job {job_id} not found"})
    return success_response(data=status)


@router.get("/executor/{job_id}/result")
async def get_executor_job_result(job_id: str):
    """Get shape/dtype summary for a completed job's trained weights.

    Returns weight names, shapes, dtypes, and byte sizes.
    Actual arrays are not returned (too large for HTTP).
    """
    from domains.training.executor import _instance
    if _instance is None:
        return success_response(data={"error": "executor not initialized"})
    summary = _instance.result_summary(job_id)
    if summary is None:
        info = _instance.status(job_id)
        if info is None:
            return success_response(data={"error": f"job {job_id} not found"})
        return success_response(data={"error": "job not completed or has no weight result"})
    return success_response(data=summary)


@router.post("/executor/purge")
async def purge_executor_jobs(max_age_s: float = Query(3600.0, gt=0)):
    """Remove completed/failed/cancelled jobs older than max_age_s."""
    from domains.training.executor import _instance
    if _instance is None:
        return success_response(data={"purged": 0})
    purged = _instance.purge_completed(max_age_s=max_age_s)
    return success_response(data={"purged": purged})


@router.post("/executor/{job_id}/cancel")
async def cancel_executor_job(job_id: str):
    """Request cancellation for a training job.

    For queued jobs the future is cancelled outright.  For running jobs
    a flag is set; the training function must check
    ``executor.is_cancelled(job_id)`` periodically.
    """
    from domains.training.executor import _instance
    if _instance is None:
        return success_response(data={"cancelled": False, "reason": "executor not initialized"})
    cancelled = _instance.cancel(job_id)
    return success_response(data={"cancelled": cancelled})


# ── Inference pool ────────────────────────────────────────────────────

@router.get("/inference-pool")
async def get_inference_pool_status():
    """Get InferencePool status."""
    from infrastructure.inference_pool import InferencePool
    try:
        pool = await InferencePool.get_instance()
        return success_response(data={
            "initialized": True,
            "max_workers": pool._max_workers,
            "queue_timeout": pool._queue_timeout,
        })
    except Exception as exc:
        return success_response(data={
            "initialized": False,
            "error": str(exc),
        })
