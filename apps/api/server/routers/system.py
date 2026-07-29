"""
System Router - Host metrics, system information, lifecycle status, and output stream.
"""
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
import asyncio
import psutil
import platform
import time

from schemas.common import success_response
from infrastructure.auth import require_auth_if_enabled


class SystemRouter:
    def __init__(self):
        self._metrics_cache = {"data": None, "ts": 0.0}
        self._METRICS_TTL = 2.0
        self.router = APIRouter(prefix="/system", tags=["system"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/metrics", self.get_metrics, methods=["GET"])
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/disk", self.get_disk, methods=["GET"])
        self.router.add_api_route("/lifecycle", self.get_lifecycle_status, methods=["GET"])
        self.router.add_api_route("/stream", self.stream_output, methods=["GET"])
        self.router.add_api_route("/output", self.tail_output, methods=["GET"])
        self.router.add_api_route("/executor", self.get_executor_status, methods=["GET"])
        self.router.add_api_route("/executor/{job_id}", self.get_executor_job, methods=["GET"])
        self.router.add_api_route("/executor/{job_id}/result", self.get_executor_job_result, methods=["GET"])
        self.router.add_api_route("/executor/purge", self.purge_executor_jobs, methods=["POST"])
        self.router.add_api_route("/executor/{job_id}/cancel", self.cancel_executor_job, methods=["POST"])
        self.router.add_api_route("/inference-pool", self.get_inference_pool_status, methods=["GET"])

    async def get_metrics(self):
        """Get system metrics (cached for 2s)"""
        now = time.monotonic()
        if self._metrics_cache["data"] is None or (now - self._metrics_cache["ts"]) > self._METRICS_TTL:
            try:
                from domains.infrastructure.resource_manager import get_resource_manager
                rm = get_resource_manager()
                logical = rm.topology.logical_cores
                physical = rm.topology.physical_cores
            except Exception:
                logical = psutil.cpu_count(logical=True) or 1
                physical = psutil.cpu_count(logical=False) or 1
            self._metrics_cache["data"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count_logical": logical,
                "cpu_count_physical": physical,
            }
            self._metrics_cache["ts"] = now
        return success_response(data=self._metrics_cache["data"])

    async def get_info(self):
        """Get system info"""
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            cpu_count = rm.topology.logical_cores
        except Exception:
            cpu_count = psutil.cpu_count()
        return success_response(data={
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": cpu_count,
        })

    async def get_disk(self):
        """Get disk usage"""
        disk = psutil.disk_usage('/')
        return success_response(data={
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent,
        })

    async def get_lifecycle_status(self):
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

    async def stream_output(self, request: Request, tail: int = Query(50, ge=0, le=500)):
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

    async def tail_output(self, n: int = Query(100, ge=1, le=1000)):
        """Get last N lines of server output."""
        from domains.infrastructure.output_buffer import get_server_buffer
        buf = get_server_buffer()
        return success_response(data={"lines": buf.tail_dicts(n), "size": buf.count, "seq": buf.seq})

    async def get_executor_status(self):
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

    async def get_executor_job(self, job_id: str):
        """Get metadata for a single training job by ID."""
        from domains.training.executor import _instance
        if _instance is None:
            return success_response(data={"error": "executor not initialized"})
        status = _instance.status(job_id)
        if status is None:
            return success_response(data={"error": f"job {job_id} not found"})
        return success_response(data=status)

    async def get_executor_job_result(self, job_id: str):
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

    async def purge_executor_jobs(
        self,
        max_age_s: float = Query(3600.0, gt=0),
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Remove completed/failed/cancelled jobs older than max_age_s."""
        from domains.training.executor import _instance
        if _instance is None:
            return success_response(data={"purged": 0})
        purged = _instance.purge_completed(max_age_s=max_age_s)
        return success_response(data={"purged": purged})

    async def cancel_executor_job(self, job_id: str, auth_user: dict = Depends(require_auth_if_enabled)):
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

    async def get_inference_pool_status(self):
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


router = SystemRouter().router
