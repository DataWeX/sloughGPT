"""
System Router - Host metrics, system information, lifecycle status, and output stream.
"""
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
import asyncio
import logging
import psutil
import platform
import time

from schemas.common import success_response, raise_error, safe_audit_log, classify_and_raise
from infrastructure.auth import require_auth_if_enabled
from typing import AsyncGenerator

logger = logging.getLogger("slo.routers.system")

logger = logging.getLogger("slo.api.system")


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

    async def get_metrics(self) -> dict:
        """Get system metrics (cached for 2s)."""
        try:
            now = time.monotonic()
            if self._metrics_cache["data"] is None or (now - self._metrics_cache["ts"]) > self._METRICS_TTL:
                import asyncio

                def _sample():
                    try:
                        from domains.infrastructure.resource_manager import get_resource_manager
                        rm = get_resource_manager()
                        logical = rm.topology.logical_cores
                        physical = rm.topology.physical_cores
                    except Exception as exc:
                        logger.debug("system: resource_manager unavailable for metrics: %s", exc)
                        logical = psutil.cpu_count(logical=True) or 1
                        physical = psutil.cpu_count(logical=False) or 1
                    return {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": psutil.virtual_memory().percent,
                        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                        "cpu_count_logical": logical,
                        "cpu_count_physical": physical,
                    }

                self._metrics_cache["data"] = await asyncio.to_thread(_sample)
                self._metrics_cache["ts"] = now
            return success_response(data=self._metrics_cache["data"])
        except Exception as e:
            classify_and_raise(e, source="system.metrics")

    async def get_info(self) -> dict:
        """Retrieve host system information including platform and CPU details."""
        try:
            import asyncio

            def _read():
                try:
                    from domains.infrastructure.resource_manager import get_resource_manager
                    rm = get_resource_manager()
                    cpu_count = rm.topology.logical_cores
                except Exception as exc:
                    logger.debug("system: resource_manager unavailable for info: %s", exc)
                    cpu_count = psutil.cpu_count()
                return {
                    "platform": platform.system(),
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "processor": platform.processor(),
                    "cpu_count": cpu_count,
                }

            return success_response(data=await asyncio.to_thread(_read))
        except Exception as e:
            classify_and_raise(e, source="system.info")

    async def get_disk(self) -> dict:
        """Retrieve disk usage statistics for the root filesystem."""
        try:
            import asyncio

            def _read():
                disk = psutil.disk_usage('/')
                return {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                }

            return success_response(data=await asyncio.to_thread(_read))
        except Exception as e:
            classify_and_raise(e, source="system.disk")

    async def get_lifecycle_status(self) -> dict:
        """Get the current lifecycle manager state."""
        try:
            from domains.infrastructure.lifecycle import get_lifecycle_manager
            mgr = get_lifecycle_manager()
            return success_response(data=mgr.get_results())
        except Exception as exc:
            classify_and_raise(exc, source="system.lifecycle")

    async def stream_output(self, request: Request, tail: int = Query(50, ge=0, le=500)) -> AsyncGenerator[str, None]:
        try:
            """SSE stream of all server output (logs, training progress, etc.).

            Sends recent history first, then streams new lines as they arrive.
            Each event: {"text": "...", "level": "info|error|warning", "source": "...", "ts": 1234.5}
            """
            from domains.infrastructure.output_buffer import get_server_buffer

            buf = get_server_buffer()
            sub = buf.subscribe("http-" + str(id(request)))

            async def generate() -> AsyncGenerator[str, None]:
                """generate."""
                try:
                    for line in buf.tail(tail):
                        yield f"data: {line.to_sse()}\n\n"
                    while True:
                        if await request.is_disconnected():
                            break
                        lines = await sub.async_read(timeout=0.2)
                        for line in lines:
                            yield f"data: {line.to_sse()}\n\n"
                except (asyncio.CancelledError, GeneratorExit):
                    pass
                finally:
                    buf.unsubscribe(sub.name)

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="system.stream_output")
    async def tail_output(self, n: int = Query(100, ge=1, le=1000)) -> dict:
        """Get last N lines of server output."""
        try:
            from domains.infrastructure.output_buffer import get_server_buffer
            buf = get_server_buffer()
            return success_response(data={"lines": buf.tail_dicts(n), "size": buf.count, "seq": buf.seq})
        except Exception as e:
            classify_and_raise(e, source="system.tail_output")

    async def get_executor_status(self) -> dict:
        """Get TrainingExecutor pool status and job list."""
        try:
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
        except Exception as e:
            classify_and_raise(e, source="system.executor_status")

    async def get_executor_job(self, job_id: str) -> dict:
        """Get metadata for a single training job by ID."""
        try:
            from domains.training.executor import _instance
            if _instance is None:
                raise_error("executor not initialized", "E_INFRA_STARTUP")
            status = _instance.status(job_id)
            if status is None:
                raise_error(f"job {job_id} not found", "E_NOT_FOUND")
            return success_response(data=status)
        except Exception as e:
            classify_and_raise(e, source="system.executor_job")

    async def get_executor_job_result(self, job_id: str) -> dict:
        """Get shape/dtype summary for a completed job's trained weights."""
        try:
            from domains.training.executor import _instance
            if _instance is None:
                raise_error("executor not initialized", "E_INFRA_STARTUP")
            summary = _instance.result_summary(job_id)
            if summary is None:
                info = _instance.status(job_id)
                if info is None:
                    raise_error(f"job {job_id} not found", "E_NOT_FOUND")
                raise_error("job not completed or has no weight result", "E_DOMAIN")
            return success_response(data=summary)
        except Exception as e:
            classify_and_raise(e, source="system.executor_job_result")

    async def purge_executor_jobs(
        self,
        max_age_s: float = Query(3600.0, gt=0),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove completed/failed/cancelled jobs older than max_age_s."""
        try:
            from domains.training.executor import _instance
            if _instance is None:
                return success_response(data={"purged": 0})
            purged = _instance.purge_completed(max_age_s=max_age_s)
            safe_audit_log("executor.purge", resource="executor", detail=f"purged={purged} max_age_s={max_age_s}")
            return success_response(data={"purged": purged})
        except Exception as e:
            classify_and_raise(e, source="system.executor_purge")

    async def cancel_executor_job(self, job_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Request cancellation for a training job."""
        try:
            from domains.training.executor import _instance
            if _instance is None:
                return success_response(data={"cancelled": False, "reason": "executor not initialized"})
            cancelled = _instance.cancel(job_id)
            safe_audit_log("executor.cancel", resource=job_id, detail=f"cancelled={cancelled}")
            return success_response(data={"cancelled": cancelled})
        except Exception as e:
            classify_and_raise(e, source="system.executor_cancel")

    async def get_inference_pool_status(self) -> dict:
        """Retrieve the InferencePool worker pool status."""
        try:
            from infrastructure.inference_pool import InferencePool
            pool = await InferencePool.get_instance()
            return success_response(data={
                "initialized": True,
                "max_workers": pool._max_workers,
                "queue_timeout": pool._queue_timeout,
            })
        except Exception as exc:
            classify_and_raise(exc, source="system.inference_pool")


router = SystemRouter().router
