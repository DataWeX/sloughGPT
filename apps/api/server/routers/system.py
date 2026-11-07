"""
System Router - Host metrics, system information, lifecycle status, and output stream.

Also serves the error-logging (/errors) and session (/session) surfaces, merged
from routers/errors.py + routers/session.py in card 20261101_154058 — one
router module behind the /monitoring journey.
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

from domain.infrastructure.output_buffer import get_server_buffer
from domain.logging import LogTag

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
            from domain.infrastructure import get_server_buffer

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
        from domain.infrastructure import get_server_buffer

        buf = get_server_buffer()
        return success_response(
            data={"lines": buf.tail_dicts(n), "size": buf.count, "seq": buf.seq}
        )

    @endpoint("system.get_executor_status")
    async def get_executor_status(self) -> dict:
        """Get TrainingExecutor pool status and job list."""
        from domain.training import peek_training_executor

        _instance = peek_training_executor()

        if _instance is None:
            return success_response(
                data={
                    "initialized": False,
                    "active_jobs": 0,
                    "max_workers": 0,
                    "total_tracked": 0,
                    "jobs": [],
                }
            )
        return success_response(
            data={
                "initialized": True,
                "active_jobs": _instance.active_count(),
                "max_workers": _instance.max_workers,
                "total_tracked": _instance.job_count,
                "jobs": _instance.list_jobs(),
            }
        )

    @endpoint("system.get_executor_job")
    async def get_executor_job(self, job_id: str) -> dict:
        """Get metadata for a single training job by ID."""
        from domain.training import peek_training_executor

        _instance = peek_training_executor()

        if _instance is None:
            raise_error("executor not initialized", "E_INFRA_STARTUP")
        status = _instance.status(job_id)
        if status is None:
            raise_error(f"job {job_id} not found", "E_NOT_FOUND")
        return success_response(data=status)

    @endpoint("system.get_executor_job_result")
    async def get_executor_job_result(self, job_id: str) -> dict:
        """Get shape/dtype summary for a completed job's trained weights."""
        from domain.training import peek_training_executor

        _instance = peek_training_executor()

        if _instance is None:
            raise_error("executor not initialized", "E_INFRA_STARTUP")
        summary = _instance.result_summary(job_id)
        if summary is None:
            info = _instance.status(job_id)
            if info is None:
                raise_error(f"job {job_id} not found", "E_NOT_FOUND")
            raise_error("job not completed or has no weight result", "E_DOMAIN")
        return success_response(data=summary)

    @endpoint("system.purge_executor_jobs")
    async def purge_executor_jobs(
        self,
        max_age_s: float = Query(3600.0, gt=0),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove completed/failed/cancelled jobs older than max_age_s."""
        try:
            from domain.training import peek_training_executor

            _instance = peek_training_executor()

            if _instance is None:
                return success_response(data={"purged": 0})
            purged = _instance.purge_completed(max_age_s=max_age_s)
            safe_audit_log(
                "executor.purge",
                resource="executor",
                detail=f"purged={purged} max_age_s={max_age_s}",
            )
            return success_response(data={"purged": purged})
        except Exception as e:
            classify_and_raise(e, source="system.executor_purge")

    @endpoint("system.cancel_executor_job")
    async def cancel_executor_job(
        self, job_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Request cancellation for a training job."""
        try:
            from domain.training import peek_training_executor

            _instance = peek_training_executor()

            if _instance is None:
                return success_response(
                    data={"cancelled": False, "reason": "executor not initialized"}
                )
            cancelled = _instance.cancel(job_id)
            safe_audit_log("executor.cancel", resource=job_id, detail=f"cancelled={cancelled}")
            return success_response(data={"cancelled": cancelled})
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


# ── Error logging (merged from routers/errors.py) ─────────────────────
# Verbatim move: schemas, ErrorsRouter, the singleton and its re-export
# aliases. Only `router = _errors_instance.router` was dropped — the
# aggregate facade below registers that SAME singleton router.

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ERRORS = 500
# stdlib LogRecord attributes — passing any of these via `extra` makes
# Logger.makeRecord raise KeyError, so frontend context keys with these
# names are dropped on ingest instead of 500ing the batch.
_INGEST_RESERVED_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ERROR_DB_PATH = os.path.join(_REPO_ROOT, "data", "errors_mogdb")
_ERROR_SYNC_PATH = os.path.join(_REPO_ROOT, "data", "errors_json")


def _get_error_db():
    from mogdb import MogDB

    return MogDB(_ERROR_DB_PATH, sync_dir=_ERROR_SYNC_PATH)


# ── Pydantic models ──────────────────────────────────────────────────────


class ErrorEntry(BaseModel):
    message: str = Field(max_length=5000)
    source: str = Field(default="web", max_length=100)
    stack: str | None = Field(default=None, max_length=20000)
    url: str | None = Field(default=None, max_length=2000)
    line: int | None = None
    col: int | None = None
    timestamp: str | None = None
    metadata: dict | None = None


class ErrorBatch(BaseModel):
    errors: list[ErrorEntry] = Field(max_length=100)


class FrontendLogEntry(BaseModel):
    level: str = Field(default="info", max_length=20)
    logger: str = Field(default="web", max_length=100)
    message: str = Field(default="", max_length=5000)
    timestamp: float = 0.0
    context: dict | None = None
    exception: str | None = Field(default=None, max_length=20000)


class FrontendLogBatch(BaseModel):
    logs: list[FrontendLogEntry] = Field(max_length=100)


# ── Router class ─────────────────────────────────────────────────────────


class ErrorsRouter:
    """OOP router for error-logging endpoints."""

    _DEDUP_WINDOW_S = 10

    def __init__(self):
        self.router = APIRouter(prefix="/errors", tags=["errors"])
        self._errors_lock = threading.Lock()
        self._error_buffer: list[dict] = []
        self._error_count_since_clear = 0
        self._dedup_map: dict[str, float] = {}
        self._storage_healthy = True

        self._error_buffer = self._load_from_mogdb()
        self._error_count_since_clear = 0

        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/log", self.log_errors, methods=["POST"])
        self.router.add_api_route("/logs/ingest", self.ingest_frontend_logs, methods=["POST"])
        self.router.add_api_route("/recent", self.get_recent_errors, methods=["GET"])
        self.router.add_api_route("/grouped", self.get_grouped_errors, methods=["GET"])
        self.router.add_api_route("/trends", self.get_error_trends, methods=["GET"])
        self.router.add_api_route("/export", self.export_errors, methods=["GET"])
        self.router.add_api_route("/clear", self.clear_errors, methods=["DELETE"])
        self.router.add_api_route("/unread", self.unread_count, methods=["GET"])
        self.router.add_api_route("/log", self.get_opencode_log, methods=["GET"])
        self.router.add_api_route(
            "/stream", self.error_stream, methods=["GET"], response_model=None
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _load_from_mogdb(self) -> list[dict]:
        """Load recent errors from MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            docs = col.find(sort=[("_created", -1)], limit=MAX_ERRORS)
            self._storage_healthy = True
            return docs
        except Exception as e:
            logger.warning("failed to load errors from mogdb: %s", e)
            self._storage_healthy = False
            return []

    def _persist_to_mogdb(self, records: list[dict]):
        """Persist error records to MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            col.insert_many(records)
            self._storage_healthy = True
        except Exception as e:
            logger.warning("failed to persist errors to mogdb: %s", e)
            self._storage_healthy = False

    def _clear_mogdb(self):
        """Clear all errors from MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            col.delete_many({})
            self._storage_healthy = True
        except Exception as e:
            logger.warning("failed to clear mogdb errors: %s", e)
            self._storage_healthy = False

    def _fingerprint(self, message: str) -> str:
        normalized = re.sub(r"\d+", "N", message.lower())
        normalized = re.sub(r"[a-f0-9]{8,}", "ID", normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    # ── Route handlers ──────────────────────────────────────────────────

    @endpoint("errors.log")
    async def log_errors(
        self,
        batch: ErrorBatch,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Log one or more client-side JavaScript errors for server-side monitoring."""
        now_ts = datetime.now(UTC)
        now_iso = now_ts.isoformat()
        now_epoch = now_ts.timestamp()
        client_host = request.client.host if request.client else "unknown"

        logged = 0
        batch_fps: set = set()
        records_to_persist: list = []
        with self._errors_lock:
            for entry in batch.errors:
                message = entry.message or ""
                fp = self._fingerprint(message)

                last_seen = self._dedup_map.get(fp)
                if (
                    fp not in batch_fps
                    and last_seen is not None
                    and (now_epoch - last_seen) < self._DEDUP_WINDOW_S
                ):
                    for rec in reversed(self._error_buffer):
                        if rec.get("fingerprint") == fp:
                            rec["count"] = rec.get("count", 1) + 1
                            rec["timestamp"] = entry.timestamp or now_iso
                            break
                    continue

                batch_fps.add(fp)
                self._dedup_map[fp] = now_epoch
                if len(self._dedup_map) > 500:
                    cutoff = now_epoch - self._DEDUP_WINDOW_S * 10
                    self._dedup_map = {k: v for k, v in self._dedup_map.items() if v > cutoff}

                error_record = {
                    "id": uuid.uuid4().hex[:12],
                    "message": entry.message,
                    "source": entry.source or "web",
                    "stack": entry.stack,
                    "url": entry.url,
                    "line": entry.line,
                    "col": entry.col,
                    "client_host": client_host,
                    "timestamp": entry.timestamp or now_iso,
                    "metadata": entry.metadata or {},
                    "fingerprint": fp,
                    "count": 1,
                }
                self._error_buffer.append(error_record)
                self._error_count_since_clear += 1
                records_to_persist.append(error_record)

                is_extension = any(
                    x in (entry.url or "")
                    for x in ("chrome-extension://", "moz-extension://", "extension://")
                )
                is_vague = entry.message in (
                    "error",
                    "Script error.",
                    "Non-Error promise rejection",
                )
                log_fn = logger.debug if (is_extension or is_vague) else logger.error
                log_fn(
                    "CLIENT ERROR [%s] %s | %s:%s %s",
                    error_record["id"],
                    entry.message[:120],
                    entry.url or "?",
                    entry.line or 0,
                    entry.col or 0,
                )
                logged += 1

            while len(self._error_buffer) > MAX_ERRORS:
                self._error_buffer.pop(0)

        if records_to_persist:
            await asyncio.to_thread(self._persist_to_mogdb, records_to_persist)

        return success_response(data={"status": "ok", "logged": logged})

    @endpoint("errors.ingest")
    async def ingest_frontend_logs(
        self, batch: FrontendLogBatch, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Ingest frontend logs through the core logging pipeline."""
        user_id = (auth_user or {}).get("id") or (auth_user or {}).get("username") or "anonymous"

        for entry in batch.logs:
            context: dict = {}
            if entry.context:
                context.update(entry.context)
            if entry.exception:
                context["exception"] = entry.exception

            raw_tag = context.pop("tag", None)
            tag = raw_tag if raw_tag and raw_tag in LogTag._value2member_map_ else "WEB"
            # Drop keys that would collide with stdlib LogRecord attributes
            # (makeRecord raises KeyError on overwrite) — they carry no
            # structured value for our formatters anyway.
            for reserved in _INGEST_RESERVED_KEYS:
                context.pop(reserved, None)
            level = getattr(logging, entry.level.upper(), logging.INFO)
            logger_name = (
                f"slo.web.{entry.logger}"
                if entry.logger and not entry.logger.startswith("slo.")
                else entry.logger or "slo.web"
            )

            # Spread context as top-level extras so formatters render each
            # key once as `k=v`.  The logger name already identifies the
            # source — no nested `context` dict or `source` echo.
            _log = logging.getLogger(logger_name)
            _log.log(
                level,
                entry.message,
                extra={"tag": tag, **context, "user": user_id},
            )

        return success_response(data={"status": "ok", "ingested": len(batch.logs)})

    @endpoint("errors.recent")
    async def get_recent_errors(self, limit: int = 50, offset: int = 0) -> dict:
        """Get recent errors from the buffer."""
        with self._errors_lock:
            total = len(self._error_buffer)
            start = max(0, total - offset - limit)
            end = max(0, total - offset)
            errors = list(reversed(self._error_buffer[start:end]))
            unread_count = self._error_count_since_clear
        return success_response(
            data={
                "errors": errors,
                "unread_count": unread_count,
                "total": total,
                "offset": offset,
                "limit": limit,
                "storage_healthy": self._storage_healthy,
            }
        )

    @endpoint("errors.grouped")
    async def get_grouped_errors(self) -> dict:
        """Get errors grouped by fingerprint."""
        with self._errors_lock:
            buffer_snapshot = list(self._error_buffer)
        groups: dict[str, dict] = {}
        for entry in reversed(buffer_snapshot):
            fp = entry.get("fingerprint") or self._fingerprint(entry.get("message", ""))
            count = entry.get("count", 1)
            if fp in groups:
                groups[fp]["count"] += count
                groups[fp]["latest"] = entry.get("timestamp", "")
            else:
                groups[fp] = {
                    "fingerprint": fp,
                    "message": entry.get("message", "")[:200],
                    "source": entry.get("source", ""),
                    "count": count,
                    "latest": entry.get("timestamp", ""),
                    "sample_id": entry.get("id", ""),
                    "sample_url": entry.get("url", ""),
                    "sample_line": entry.get("line"),
                }
        result = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
        return success_response(data={"groups": result, "total_groups": len(result)})

    @endpoint("errors.trends")
    async def get_error_trends(self, hours: int = 24) -> dict:
        """Get error counts per hour for the last N hours."""
        now = datetime.now(UTC)
        buckets: dict[str, int] = {}

        for h in range(hours):
            t = now.replace(minute=0, second=0, microsecond=0)
            from datetime import timedelta

            t = t - timedelta(hours=h)
            key = t.strftime("%Y-%m-%dT%H:00")
            buckets[key] = 0

        with self._errors_lock:
            buffer_snapshot = list(self._error_buffer)
        for entry in buffer_snapshot:
            ts = entry.get("timestamp", "")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                key = t.strftime("%Y-%m-%dT%H:00")
                if key in buckets:
                    buckets[key] += 1
            except (ValueError, TypeError):
                logger.debug("Skipping unparseable timestamp in error trends: %s", ts)

        try:
            db = _get_error_db()
            col = db.collection("errors")
            all_errors = col.find()
            for rec in all_errors:
                ts = rec.get("timestamp", "")
                if not ts:
                    continue
                try:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    key = t.strftime("%Y-%m-%dT%H:00")
                    if key in buckets:
                        buckets[key] += 1
                except (ValueError, TypeError):
                    logger.debug("Skipping unparseable timestamp in mogdb trends: %s", ts)
        except Exception as e:
            logger.warning("failed to read trends from mogdb: %s", e)

        result = [{"hour": k, "count": v} for k, v in sorted(buckets.items())]
        return success_response(data={"trends": result, "hours": hours})

    @endpoint("errors.export")
    async def export_errors(self, limit: int = 500) -> dict:
        """Export errors as JSON."""
        with self._errors_lock:
            errors = list(reversed(self._error_buffer[-limit:]))
            total = len(self._error_buffer)
        return JSONResponse(
            content={"errors": errors, "total": total, "exported": len(errors)},
            headers={
                "Content-Disposition": f'attachment; filename="errors-{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}.json"'
            },
        )

    @endpoint("errors.clear")
    async def clear_errors(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Clear all errors."""
        with self._errors_lock:
            self._error_buffer.clear()
            self._error_count_since_clear = 0
            self._dedup_map.clear()
        await asyncio.to_thread(self._clear_mogdb)
        safe_audit_log("errors.clear", resource="all")
        return success_response(data={"status": "ok", "cleared": True})

    @endpoint("errors.unread")
    async def unread_count(self) -> dict:
        """Get unread error count."""
        with self._errors_lock:
            count = self._error_count_since_clear
        return success_response(data={"unread_count": count})

    @endpoint("errors.stream")
    async def error_stream(self, request: Request) -> StreamingResponse:
        """SSE endpoint pushing real-time error events to the frontend."""
        import time as _time

        from fastapi.responses import StreamingResponse

        buf = get_server_buffer()
        subscriber = buf.subscribe("error-stream")
        seen_seq = buf.seq

        async def generate():
            nonlocal seen_seq
            try:
                while True:
                    if await request.is_disconnected():
                        break

                    lines = await subscriber.async_read(timeout=1.0)

                    now = _time.time()
                    if not hasattr(generate, "_last_yield"):
                        generate._last_yield = now
                    if now - generate._last_yield >= 30.0:
                        yield ": heartbeat\n\n"
                        generate._last_yield = now

                    for line in lines:
                        if line.level in ("error", "critical"):
                            ctx = line.context or {}
                            corr_id = ctx.get("corr") or ctx.get("correlation_id") or ""
                            http_method = ctx.get("method", "")
                            http_path = ctx.get("path", "")
                            http_status = ctx.get("status")
                            dur_ms = ctx.get("dur_ms")

                            event = {
                                "stream": "errors",
                                "phase": "ERROR",
                                "status": "working",
                                "data": {
                                    "message": line.text,
                                    "level": line.level,
                                    "source": line.source,
                                    "tag": line.tag,
                                    "context": ctx,
                                    "ts": line.timestamp,
                                    "correlation_id": corr_id,
                                    "http_method": http_method,
                                    "http_path": http_path,
                                    "http_status": http_status,
                                    "duration_ms": dur_ms,
                                },
                                "meta": {"ts": _time.time()},
                            }
                            yield "data: " + json.dumps(event, default=str) + "\n\n"
                            generate._last_yield = _time.time()

                    with self._errors_lock:
                        current_count = len(self._error_buffer)

                    if current_count > seen_seq:
                        with self._errors_lock:
                            new_errors = list(self._error_buffer[seen_seq:current_count])
                        seen_seq = current_count
                        for err in new_errors:
                            event = {
                                "stream": "errors",
                                "phase": "CLIENT_ERROR",
                                "status": "working",
                                "data": {
                                    "id": err.get("id", ""),
                                    "message": err.get("message", ""),
                                    "source": err.get("source", ""),
                                    "stack": err.get("stack"),
                                    "url": err.get("url"),
                                    "line": err.get("line"),
                                    "col": err.get("col"),
                                    "fingerprint": err.get("fingerprint", ""),
                                    "count": err.get("count", 1),
                                    "timestamp": err.get("timestamp", ""),
                                    "metadata": err.get("metadata", {}),
                                },
                                "meta": {"ts": _time.time()},
                            }
                            yield "data: " + json.dumps(event, default=str) + "\n\n"
                            generate._last_yield = _time.time()

            except Exception as e:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "stream": "errors",
                            "phase": "ERROR",
                            "status": "error",
                            "data": {"error": str(e)},
                            "message": str(e),
                        }
                    )
                    + "\n\n"
                )
            finally:
                buf.unsubscribe("error-stream")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @endpoint("errors.opencode_log")
    async def get_opencode_log(self) -> dict:
        """Get opencode CLI and UI error logs."""
        home = os.path.expanduser("~")
        entries: list[dict] = []

        cli_log_path = os.path.join(home, ".opencode-error-log.json")
        try:
            if os.path.exists(cli_log_path):

                def _read_cli_log():
                    with open(cli_log_path) as f:
                        return json.load(f)

                data = await asyncio.to_thread(_read_cli_log)
                for rec in data:
                    entries.append(
                        {
                            "timestamp": rec.get("timestamp", ""),
                            "command": rec.get("command", ""),
                            "category": "cli",
                            "pattern": rec.get("pattern", ""),
                            "snippet": rec.get("snippet", ""),
                            "cwd": rec.get("cwd", ""),
                        }
                    )
        except Exception as e:
            logger.warning("failed to read cli log: %s", e)

        ui_log_path = os.path.join(home, ".opencode-ui-error-log.json")
        try:
            if os.path.exists(ui_log_path):

                def _read_ui_log():
                    with open(ui_log_path) as f:
                        return json.load(f)

                data = await asyncio.to_thread(_read_ui_log)
                for rec in data:
                    entries.append(
                        {
                            "timestamp": rec.get("timestamp", ""),
                            "command": rec.get("command", ""),
                            "category": rec.get("category", "unknown"),
                            "pattern": rec.get("pattern", ""),
                            "snippet": rec.get("snippet", ""),
                            "cwd": rec.get("cwd", ""),
                        }
                    )
        except Exception as e:
            logger.warning("failed to read ui log: %s", e)

        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return success_response(data={"entries": entries[:200], "total": len(entries)})


# ── Singleton + backward-compat re-exports ──────────────────────────────────

_errors_instance = ErrorsRouter()
_error_buffer = _errors_instance._error_buffer
_error_count_since_clear = _errors_instance._error_count_since_clear
_dedup_map = _errors_instance._dedup_map


async def clear_errors() -> dict:
    """clear_errors — async backward-compat re-export."""
    return await _errors_instance.clear_errors()


# ── Session management (merged from routers/session.py) ──────────────
# Verbatim move; SessionRouter is constructed once, at the aggregate
# facade below, preserving the old one-instance-per-import semantics.


class SessionContext(BaseModel):
    system_prompt: str | None = None
    knowledge: list[str] | None = None
    messages: list[dict[str, Any]] | None = []


class SessionRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/session", tags=["session"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/{session_id}/context", self.set_session_context, methods=["POST"]
        )
        self.router.add_api_route(
            "/{session_id}/messages", self.get_session_messages, methods=["GET"]
        )
        self.router.add_api_route(
            "/{session_id}/inspector", self.get_session_inspector, methods=["GET"]
        )

    @staticmethod
    def get_router_session_context(session_id: str) -> list[dict[str, Any]] | None:
        """Legacy helper – forwards to ``SessionCore``.

        Existing code (e.g., ``main.py``) still calls this function. It now
        proxies to the unified ``SessionCore`` implementation.
        """
        from domain.infrastructure.session_core import SessionCore

        return SessionCore.get_messages(session_id)

    @endpoint("session.set_session_context")
    async def set_session_context(
        self,
        session_id: str,
        ctx: SessionContext,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Set session context (messages stored for regeneration)."""
            if ctx.messages:
                from domain.infrastructure.session_core import SessionCore

                result = SessionCore.store_context(session_id, ctx.messages)
                safe_audit_log(
                    "session.context_store",
                    resource=session_id,
                    detail=f"messages={len(ctx.messages)}",
                )
                return success_response(data=result)
            return success_response(
                data={"session_id": session_id, "message_count": 0}, message="stored"
            )

        except Exception as e:
            classify_and_raise(e, source="session.set_session_context")

    @endpoint("session.get_session_messages")
    async def get_session_messages(self, session_id: str) -> dict:
        """Return stored conversation messages for a session.

        Used by the UI to load a chat history.
        """
        from domain.infrastructure.session_core import SessionCore

        try:
            msgs = SessionCore.get_messages(session_id)
            return success_response(data={"session_id": session_id, "messages": msgs})
        except Exception as e:
            logger.warning("Get session messages failed: %s", e)
            classify_and_raise(e, source="session_get_messages")

    @endpoint("session.get_session_inspector")
    async def get_session_inspector(self, session_id: str) -> dict:
        """Return aggregated context state for the UI context inspector.

        Combines session context, knowledge stats, trait weights, manager
        modes, and workspace memory into a single response. Fetches all
        sources concurrently for low latency.
        """
        _inspector_start = time.time()
        try:

            def _fetch_messages():
                from domain.infrastructure.session_core import SessionCore

                return SessionCore.get_messages(session_id)

            def _fetch_feedback():
                from domain.feedback import get_message_feedback

                fb = get_message_feedback()
                return fb.get_stats()

            def _fetch_knowledge():
                knowledge = {"total_facts": 0, "topics": []}
                try:
                    from domain.knowledge import get_knowledge_memory

                    km = get_knowledge_memory()
                    knowledge["total_facts"] = km.stats().get("total_facts", 0)
                    knowledge["topics"] = [t[0] for t in km.all_topics()[:10]]
                except Exception as e:
                    logger.debug("Knowledge memory unavailable in session detail: %s", e)
                return knowledge

            def _fetch_traits():
                traits, modes = {}, {}
                try:
                    from domain.context import (
                        MemoryManager,
                        PersonalityManager,
                        StyleManager,
                        TaskManager,
                        get_trait_config,
                    )

                    config = get_trait_config()
                    traits = config.all()
                    modes = {
                        "personality": PersonalityManager(config).get_mode(),
                        "memory": MemoryManager(config).get_mode(),
                        "style": StyleManager(config).get_mode(),
                        "task": TaskManager(config).get_mode(),
                    }
                except Exception as e:
                    logger.debug("Trait config unavailable in session detail: %s", e)
                return traits, modes

            def _fetch_workspace():
                workspace = {"working_memory": [], "semantic_keys": [], "episodic_count": 0}
                try:
                    from domain.infrastructure.context_core import get_context_core

                    cc = get_context_core()
                    insp = cc.get_context_inspector()
                    workspace = {
                        "working_memory": insp.get("working_memory", []),
                        "semantic_keys": insp.get("semantic_keys", []),
                        "episodic_count": insp.get("episodic_count", 0),
                        "sensory_buffer_size": insp.get("sensory_buffer_size", 0),
                        "system_prompt": insp.get("system_prompt", "")[:300],
                    }
                except Exception as e:
                    logger.debug("Context core unavailable in session detail: %s", e)
                return workspace

            msgs, fb_stats, knowledge, (traits, modes), workspace = await asyncio.gather(
                asyncio.to_thread(_fetch_messages),
                asyncio.to_thread(_fetch_feedback),
                asyncio.to_thread(_fetch_knowledge),
                asyncio.to_thread(_fetch_traits),
                asyncio.to_thread(_fetch_workspace),
            )
            _elapsed_ms = round((time.time() - _inspector_start) * 1000)

            return success_response(
                data={
                    "session": {
                        "id": session_id,
                        "message_count": len(msgs),
                        "messages": msgs[-10:],
                    },
                    "knowledge": knowledge,
                    "traits": traits,
                    "modes": modes,
                    "feedback": {
                        "total": fb_stats.get("feedback_total", 0),
                        "thumbs_up": fb_stats.get("thumbs_up", 0),
                        "thumbs_down": fb_stats.get("thumbs_down", 0),
                    },
                    "workspace": workspace,
                    "elapsed_ms": _elapsed_ms,
                }
            )

        except Exception as e:
            logger.warning("Session inspector failed: %s", e)
            classify_and_raise(e, source="session_inspector")


# ── Aggregate facade ──────────────────────────────────────────────────────
# Boot contract: one module-level router per module (ROUTER_MODULES
# manifest). Sub-router prefixes bake at add-time, so /system/*, /errors/*
# and /session/* paths are byte-identical to the old three-module layout.

router = APIRouter()
router.include_router(SystemRouter().router)
router.include_router(_errors_instance.router)
router.include_router(SessionRouter().router)
