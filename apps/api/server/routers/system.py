"""
System Router - Host metrics, system information, lifecycle status, and output stream.
"""

import asyncio
import logging
import platform
import threading
import time
from collections.abc import AsyncGenerator

import psutil
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

logger = logging.getLogger("slo.routers.system")


def _daemon_summary(state: dict, policy) -> dict:
    """Compact view of the enforcement loop for the UI — a summary, not a state dump.

    ``active`` means the state file is fresh relative to the policy interval, which is
    the signal the UI needs to distinguish "managed" from "policy set, nobody enforcing".
    """
    if not state:
        return {"present": False, "active": False, "explain": None}
    updated_at = state.get("updated_at") or 0
    try:
        age = max(0.0, time.time() - float(updated_at))
    except (TypeError, ValueError):
        age = None
    ttl = max(180.0, float(getattr(policy, "interval_seconds", 60.0)) * 3)
    action = state.get("action") or {}
    result = state.get("result") or {}
    return {
        "present": True,
        "active": age is not None and age <= ttl,
        "pid": state.get("pid"),
        "age_seconds": round(age, 1) if age is not None else None,
        "owned": bool(state.get("owned")),
        "dry_run": bool(state.get("dry_run")),
        "last_action": action.get("action"),
        "last_value": action.get("value"),
        "last_reason": action.get("reason"),
        "last_outcome": result.get("reason"),
        "explain": state.get("explain"),
    }


class SystemRouter:
    def __init__(self):
        self._metrics_cache = {"data": None, "ts": 0.0}
        self._metrics_lock = threading.Lock()
        self._METRICS_TTL = 2.0
        self.router = APIRouter(prefix="/system", tags=["system"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/metrics", self.get_metrics, methods=["GET"])
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/disk", self.get_disk, methods=["GET"])
        self.router.add_api_route("/battery", self.get_battery, methods=["GET"])
        self.router.add_api_route("/battery/limit", self.set_battery_limit, methods=["POST"])
        self.router.add_api_route("/battery/policy", self.set_battery_policy, methods=["PUT"])
        self.router.add_api_route("/lifecycle", self.get_lifecycle_status, methods=["GET"])
        self.router.add_api_route(
            "/stream", self.stream_output, methods=["GET"], response_model=None
        )
        self.router.add_api_route("/output", self.tail_output, methods=["GET"])
        self.router.add_api_route("/executor", self.get_executor_status, methods=["GET"])
        self.router.add_api_route("/executor/{job_id}", self.get_executor_job, methods=["GET"])
        self.router.add_api_route(
            "/executor/{job_id}/result", self.get_executor_job_result, methods=["GET"]
        )
        self.router.add_api_route("/executor/purge", self.purge_executor_jobs, methods=["POST"])
        self.router.add_api_route(
            "/executor/{job_id}/cancel", self.cancel_executor_job, methods=["POST"]
        )
        self.router.add_api_route(
            "/inference-pool", self.get_inference_pool_status, methods=["GET"]
        )

    @endpoint("system.get_metrics")
    async def get_metrics(self) -> dict:
        """Get system metrics (cached for 2s)."""
        try:
            now = time.monotonic()
            with self._metrics_lock:
                if (
                    self._metrics_cache["data"] is not None
                    and (now - self._metrics_cache["ts"]) <= self._METRICS_TTL
                ):
                    return success_response(data=self._metrics_cache["data"])

            def _sample():
                try:
                    from domain.infrastructure.resource_manager import get_resource_manager

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

            data = await asyncio.to_thread(_sample)
            with self._metrics_lock:
                self._metrics_cache["data"] = data
                self._metrics_cache["ts"] = now
            return success_response(data=data)
        except Exception as e:
            classify_and_raise(e, source="system.metrics")

    @endpoint("system.get_info")
    async def get_info(self) -> dict:
        """Retrieve host system information including platform and CPU details."""
        try:

            def _read():
                try:
                    from domain.infrastructure.resource_manager import get_resource_manager

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

    @endpoint("system.get_disk")
    async def get_disk(self) -> dict:
        """Retrieve disk usage statistics for the root filesystem."""
        try:

            def _read():
                disk = psutil.disk_usage("/")
                return {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                }

            return success_response(data=await asyncio.to_thread(_read))
        except Exception as e:
            classify_and_raise(e, source="system.disk")

    @endpoint("system.get_battery")
    async def get_battery(self) -> dict:
        """Read charge state, charge-cap capability, longevity advice, and policy.

        Reads are pure, so this is safe to poll. ``control.supported`` is false on
        machines whose kernel does not expose a charge threshold (VMs, containers,
        many chassis) — that is a capability report, not an error. ``daemon`` reports
        what the enforcement loop last did, if it has ever run.
        """

        def _read():
            from chargectl import (
                BatteryReader,
                daemon_state_path,
                default_policy_path,
                default_sys_base,
                explain,
                load_policy,
                optimize_hint,
                probe,
                read_state,
            )

            base = default_sys_base()
            status = BatteryReader(sys_base=base).read()
            capability = probe(base)
            policy, policy_error = load_policy()
            state_path = daemon_state_path()
            return {
                "status": status.as_dict(),
                "control": capability.as_dict(),
                "advice": optimize_hint(status),
                "policy": {
                    **policy.as_dict(),
                    "file": str(default_policy_path()),
                    "error": policy_error,
                    "explain": explain(policy, capability),
                },
                "daemon": {
                    **_daemon_summary(read_state(state_path), policy),
                    "state_file": str(state_path),
                },
            }

        try:
            return success_response(data=await asyncio.to_thread(_read))
        except Exception as e:
            classify_and_raise(e, source="system.battery")

    @endpoint("system.set_battery_policy")
    async def set_battery_policy(
        self,
        enabled: bool | None = Query(None),
        floor: int | None = Query(None, ge=1, le=100),
        ceiling: int | None = Query(None, ge=1, le=100),
        mode: str | None = Query(None),
        interval: float | None = Query(None, ge=1),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Persist the charge policy the daemon enforces.

        Never fails hard: an invalid band comes back as ``ok: false`` with the reason
        and the untouched policy, so the UI can show it inline instead of a 4xx.
        """

        def _write():
            from chargectl import (
                default_sys_base,
                explain,
                load_policy,
                normalize,
                probe,
                save_policy,
            )

            base, load_error = load_policy()
            policy, error = normalize(
                floor=floor,
                ceiling=ceiling,
                mode=mode,
                enabled=enabled,
                interval_seconds=interval,
                base=base,
            )
            if error:
                return {"ok": False, "error": error, "policy": base.as_dict()}
            path = save_policy(policy)
            return {
                "ok": True,
                "error": None,
                "load_error": load_error,
                "policy": policy.as_dict(),
                "file": str(path),
                "explain": explain(policy, probe(default_sys_base())),
            }

        try:
            result = await asyncio.to_thread(_write)
            saved = result["policy"]
            safe_audit_log(
                "system.battery_policy",
                resource="battery",
                detail=(
                    f"ok={result['ok']} enabled={saved.get('enabled')} "
                    f"band={saved.get('band')} mode={saved.get('mode')}"
                ),
            )
            return success_response(data=result)
        except Exception as e:
            classify_and_raise(e, source="system.battery_policy")

    @endpoint("system.set_battery_limit")
    async def set_battery_limit(
        self,
        percent: int = Query(80, ge=1, le=100),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Cap charge at ``percent`` to prevent overcharge (100 lifts the cap).

        Never fails hard: an unsupported or read-only kernel returns
        ``applied: false`` with a human reason.
        """

        def _write():
            from chargectl import set_limit

            return set_limit(percent)

        try:
            result = await asyncio.to_thread(_write)
            safe_audit_log(
                "system.battery_limit",
                resource="battery",
                detail=f"percent={percent} applied={result.applied} reason={result.reason}",
            )
            return success_response(data=result.as_dict())
        except Exception as e:
            classify_and_raise(e, source="system.battery_limit")

    @endpoint("system.get_lifecycle_status")
    async def get_lifecycle_status(self) -> dict:
        """Get the current lifecycle manager state."""
        try:
            from domain.infrastructure import get_lifecycle_manager

            mgr = get_lifecycle_manager()
            return success_response(data=mgr.get_results())
        except Exception:
            return success_response(data={"phase": "unavailable"})

    @endpoint("system.stream_output")
    async def stream_output(
        self, request: Request, tail: int = Query(50, ge=0, le=500)
    ) -> AsyncGenerator[str, None]:
        try:
            """SSE stream of all server output (logs, training progress, etc.).

            Sends recent history first, then streams new lines as they arrive.
            Each event: {"text": "...", "level": "info|error|warning", "source": "...", "ts": 1234.5}
            """
            from domain.infrastructure._internal.output_buffer import get_server_buffer

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
                    pass  # Expected: client disconnected or stream cancelled
                finally:
                    buf.unsubscribe(sub.name)

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="system.stream_output")

    @endpoint("system.tail_output")
    async def tail_output(self, n: int = Query(100, ge=1, le=1000)) -> dict:
        """Get last N lines of server output."""
        from domain.infrastructure._internal.output_buffer import get_server_buffer

        buf = get_server_buffer()
        return success_response(
            data={"lines": buf.tail_dicts(n), "size": buf.count, "seq": buf.seq}
        )

    @endpoint("system.get_executor_status")
    async def get_executor_status(self) -> dict:
        """Get TrainingExecutor pool status and job list."""
        from domain.training.engine import get_training_engine

        return success_response(data=get_training_engine().executor_status().data)

    @endpoint("system.get_executor_job")
    async def get_executor_job(self, job_id: str) -> dict:
        """Get metadata for a single training job by ID."""
        from domain.training.engine import get_training_engine

        result = get_training_engine().executor_job(job_id)
        if not result.success:
            raise_error(result.error, result.metadata.get("code", "E_NOT_FOUND"))
        return success_response(data=result.data)

    @endpoint("system.get_executor_job_result")
    async def get_executor_job_result(self, job_id: str) -> dict:
        """Get shape/dtype summary for a completed job's trained weights."""
        from domain.training.engine import get_training_engine

        result = get_training_engine().executor_job_result(job_id)
        if not result.success:
            raise_error(result.error, result.metadata.get("code", "E_DOMAIN"))
        return success_response(data=result.data)

    @endpoint("system.purge_executor_jobs")
    async def purge_executor_jobs(
        self,
        max_age_s: float = Query(3600.0, gt=0),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove completed/failed/cancelled jobs older than max_age_s."""
        try:
            from domain.training.engine import get_training_engine

            result = get_training_engine().purge_executor_jobs(max_age_s=max_age_s)
            purged = result.data.get("purged", 0)
            safe_audit_log(
                "executor.purge",
                resource="executor",
                detail=f"purged={purged} max_age_s={max_age_s}",
            )
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="system.executor_purge")

    @endpoint("system.cancel_executor_job")
    async def cancel_executor_job(
        self, job_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Request cancellation for a training job."""
        try:
            from domain.training.engine import get_training_engine

            result = get_training_engine().cancel_executor_job(job_id)
            safe_audit_log(
                "executor.cancel",
                resource=job_id,
                detail=f"cancelled={result.data.get('cancelled', False)}",
            )
            return success_response(data=result.data)
        except Exception as e:
            classify_and_raise(e, source="system.executor_cancel")

    @endpoint("system.get_inference_pool_status")
    async def get_inference_pool_status(self) -> dict:
        """Retrieve the InferencePool worker pool status."""
        try:
            from infrastructure.inference_pool import InferencePool

            pool = await InferencePool.get_instance()
            return success_response(
                data={
                    "initialized": True,
                    "max_workers": pool._max_workers,
                    "queue_timeout": pool._queue_timeout,
                }
            )
        except Exception:
            return success_response(
                data={"initialized": False, "max_workers": 0, "queue_timeout": 0}
            )


router = SystemRouter().router
