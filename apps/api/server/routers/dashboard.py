"""
Dashboard Router — SSE live stream and REST endpoint for the CLI monitor.

Aggregates health data, active processes, and recent events into a single
stream consumed by ``sloughgpt monitor``.

Endpoints:
    GET /dashboard/stream  — SSE: full snapshot every 2 seconds
    GET /dashboard/events  — REST: last N events as JSON
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from schemas.common import success_response

logger = logging.getLogger("slo.dashboard")


def _get_active_processes() -> dict:
    """Collect active process status from all subsystems."""
    processes = {}

    # Training jobs
    try:
        from training.jobs import training_jobs

        for job_id, job in training_jobs.items():
            status = job.get("status", "unknown")
            if status in ("running", "queued", "starting"):
                progress = job.get("progress", 0)
                step = job.get("current_step", "")
                total = job.get("total_steps", "")
                model = job.get("model", job.get("dataset", ""))
                label = job.get("name", job_id[:8])
                detail = f"{model}" if model else ""
                if progress > 0:
                    detail += f" {progress}%"
                if step and total:
                    detail += f" step {step}/{total}"
                processes[f"train:{job_id[:8]}"] = {
                    "type": "training",
                    "status": status,
                    "label": label,
                    "detail": detail.strip(),
                    "progress": progress,
                }
    except Exception as e:
        logger.debug("Failed to collect training processes: %s", e)
    try:
        import state as server_state

        proc = server_state._self_train_proc
        if proc is not None:
            ret = proc.poll()
            if ret is None:
                processes["self-train"] = {
                    "type": "self-train",
                    "status": "running",
                    "label": "self-train",
                    "detail": f"pid {proc.pid}",
                    "progress": 0,
                }
            else:
                processes["self-train"] = {
                    "type": "self-train",
                    "status": "exited",
                    "label": "self-train",
                    "detail": f"exit code {ret}",
                    "progress": 100 if ret == 0 else 0,
                }
    except Exception as e:
        logger.debug("Failed to collect self-train status: %s", e)

    # Auto-train turbo
    try:
        from domains.training.service import get_turbo_lock, get_turbo_state

        with get_turbo_lock():
            turbo = dict(get_turbo_state())
        turbo_status = turbo.get("status", "idle")
        if turbo_status not in ("idle", "stopped"):
            epoch = turbo.get("epoch", "")
            total_epochs = turbo.get("epochs", "")
            loss = turbo.get("loss", "")
            detail = ""
            if epoch and total_epochs:
                detail = f"epoch {epoch}/{total_epochs}"
            if loss:
                detail += f" loss {loss}" if detail else f"loss {loss}"
            progress = turbo.get("progress", 0)
            processes["auto-train"] = {
                "type": "auto-train",
                "status": turbo_status,
                "label": "auto-train",
                "detail": detail.strip(),
                "progress": progress,
            }
    except Exception as e:
        logger.debug("Failed to collect auto-train status: %s", e)

    # Active downloads
    try:
        from domains.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        downloads = mgr.list_downloads()
        for model_id, dl in downloads.items():
            status = dl.get("status", "unknown")
            if status in ("complete", "failed", "cancelled"):
                continue
            pct = dl.get("percentage", 0)
            speed = dl.get("speed_mb_per_sec", 0)
            eta = dl.get("eta_seconds", 0)
            current_file = dl.get("current_file", "")
            detail = f"{pct:.0f}%"
            if speed > 0:
                detail += f" {speed:.1f}MB/s"
            if eta > 0:
                detail += f" ETA {eta:.0f}s"
            if current_file:
                # Show just filename, not full path
                fname = current_file.rsplit("/", 1)[-1] if "/" in current_file else current_file
                if len(fname) > 20:
                    fname = fname[:18] + "..."
                detail += f" {fname}"
            progress = pct
            processes[f"dl:{model_id[:20]}"] = {
                "type": "download",
                "status": status,
                "label": model_id[:14],
                "detail": detail,
                "progress": progress,
            }
    except Exception as e:
        logger.debug("Failed to collect download status: %s", e)

    return processes


def _get_health_summary() -> dict:
    """Fast health summary from existing sources."""
    try:
        import psutil
        import state as server_state

        model_loaded = server_state.model is not None or server_state.provider is not None
        model_type = getattr(server_state, "model_type", None) or ""
        uptime = server_state.uptime_seconds
        req_count = server_state.request_count
        err_count = server_state.error_count
        tps = server_state.get_tokens_per_second()
        avg_lat = server_state.get_avg_latency()
        rpm = server_state.get_requests_per_minute()

        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)

        return {
            "model_loaded": model_loaded,
            "model_type": model_type,
            "uptime_seconds": round(uptime),
            "request_count": req_count,
            "error_count": err_count,
            "tokens_per_sec": tps,
            "avg_latency_ms": avg_lat,
            "requests_per_minute": rpm,
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(mem.percent, 1),
            "memory_used_mb": round(mem.used / 1024 / 1024),
        }
    except Exception as exc:
        import logging

        logging.getLogger("slo.dashboard").warning(
            "Health data collection failed: %s", exc, exc_info=True
        )
        return {"model_loaded": False, "error": f"health unavailable: {exc}"}


def _build_snapshot() -> dict:
    """Build a single dashboard snapshot."""
    from domains.infrastructure.event_buffer import get_event_buffer

    return {
        "stream": "dashboard",
        "phase": "DASHBOARD",
        "status": "working",
        "data": {
            "health": _get_health_summary(),
            "processes": _get_active_processes(),
            "events": get_event_buffer().recent(15),
        },
        "meta": {"ts": time.time()},
    }


class DashboardRouter:
    """Routes for ``/dashboard/*`` endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/dashboard", tags=["dashboard"])
        self.SNAPSHOT_INTERVAL = 2.0
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/stream", self.dashboard_stream, methods=["GET"])
        self.router.add_api_route("/events", self.dashboard_events, methods=["GET"])

    async def dashboard_stream(self, request: Request) -> StreamingResponse:
        """SSE endpoint pushing dashboard snapshots every 2 seconds."""

        async def generate() -> AsyncGenerator[str, None]:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snapshot = await asyncio.to_thread(_build_snapshot)
                    yield "data: " + json.dumps(snapshot, default=str) + "\n\n"
                except Exception as e:
                    logger.warning("Dashboard snapshot failed: %s", e)
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "stream": "dashboard",
                                "phase": "ERROR",
                                "status": "error",
                                "data": {"error": str(e)},
                                "message": str(e),
                            }
                        )
                        + "\n\n"
                    )
                await asyncio.sleep(self.SNAPSHOT_INTERVAL)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def dashboard_events(self, n: int = 20) -> dict:
        """Return the last N dashboard events as JSON."""
        try:
            from domains.infrastructure.event_buffer import get_event_buffer

            events = get_event_buffer().recent(n)
            return success_response(data={"events": events, "count": len(events)})
        except Exception as e:
            classify_and_raise(e, source="dashboard.events")


router = DashboardRouter().router
