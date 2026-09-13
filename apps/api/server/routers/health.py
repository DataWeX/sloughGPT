"""
Health Router — REST health endpoints and SSE live stream.

Endpoints:
    GET /health          — Basic health: model state, device, inference stats, lifecycle
    GET /health/live     — Kubernetes liveness probe
    GET /health/ready    — Kubernetes readiness probe
    GET /health/detailed — Full system health: CPU, memory, GPU, registry, trends
    GET /health/startup-progress — Current startup phase string
    GET /health/debug    — Debug subset: model state, metrics, histories, errors
    GET /health/model    — Model health monitor stats (perplexity, loss trends)
    GET /health/summary  — Condensed health score with diagnoses
    GET /health/stream   — SSE: pushes full snapshot every 3 seconds

All endpoints return the standard response envelope
``{"status": "success", "data": {...}}`` via ``success_response()``.

Side effects:
    - Controller methods read from ServerState, psutil, ModelRegistry
    - ``/health/detailed`` records trend snapshots on each call
    - ``/health/stream`` holds the connection open until client disconnect
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from controllers.health import get_health_controller
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from schemas.common import endpoint, success_response
from startup_progress import STARTUP_PHASE

logger = logging.getLogger(__name__)


class HealthRouter:
    """Routes for ``/health/*`` endpoints.

    Delegates all business logic to ``HealthController``. Each handler
    calls the controller in a thread pool (via ``asyncio.to_thread``) to
    avoid blocking the event loop on psutil reads or registry queries.
    """

    def __init__(self):
        self.router = APIRouter(prefix="/health", tags=["health"])
        self.HEALTH_STREAM_INTERVAL = 3.0
        self._register_routes()

    def _register_routes(self):
        """Bind all health endpoints to their handler methods."""
        self.router.add_api_route("", self.health, methods=["GET"])
        self.router.add_api_route("/live", self.liveness, methods=["GET"])
        self.router.add_api_route("/ready", self.readiness, methods=["GET"])
        self.router.add_api_route("/detailed", self.detailed_health, methods=["GET"])
        self.router.add_api_route("/startup-progress", self.startup_progress, methods=["GET"])
        self.router.add_api_route("/startup-status", self.startup_status, methods=["GET"])
        self.router.add_api_route("/startup-history", self.startup_history, methods=["GET"])
        self.router.add_api_route("/startup-diagnostics", self.startup_diagnostics, methods=["GET"])
        self.router.add_api_route("/startup-config", self.startup_config, methods=["GET"])
        self.router.add_api_route("/startup-health", self.startup_health, methods=["GET"])
        self.router.add_api_route("/startup-compare", self.startup_compare, methods=["GET"])
        self.router.add_api_route("/startup-rollback", self.startup_rollback, methods=["GET"])
        self.router.add_api_route(
            "/startup-stream", self.startup_stream, methods=["GET"], response_model=None
        )
        self.router.add_api_route("/debug", self.debug_info, methods=["GET"])
        self.router.add_api_route("/model", self.model_health, methods=["GET"])
        self.router.add_api_route("/summary", self.health_summary, methods=["GET"])
        self.router.add_api_route(
            "/stream", self.health_stream, methods=["GET"], response_model=None
        )
        self.router.add_api_route("/services", self.services_health, methods=["GET"])

    @endpoint("health.health")
    async def health(self) -> dict:
        """Basic health status.

        Returns model loaded state, device, inference count, lifecycle
        phase, and resource allocation. This is the primary health check
        for load balancers and monitoring dashboards.

        During cold start, ``get_basic_health`` can block on first-time
        module imports (psutil, controllers.models, lifecycle) when the
        model-load thread is also importing. A 5-second timeout with a
        lightweight fallback prevents the CLI's 90s startup timer from
        firing on a healthy server.

        Returns:
            Envelope with ``model_loaded``, ``model_type``, ``device``,
            ``is_inferencing``, ``lifecycle``, ``resource_allocation``.
        """
        ctrl = get_health_controller()
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(ctrl.get_basic_health),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception):
            # Fast fallback during cold start — read lightweight sources only
            import state as server_state

            data = {
                "status": "healthy",
                "model_loaded": server_state.model is not None
                or server_state.provider is not None,
                "model_type": getattr(server_state, "model_type", None),
                "lifecycle": {"phase": STARTUP_PHASE.get("phase", "initializing")},
                "model_loading": STARTUP_PHASE.get("phase") == "loading_model",
            }
        return success_response(data=data)

    @endpoint("health.liveness")
    async def liveness(self) -> dict:
        """Kubernetes liveness probe.

        Returns alive=True when the process is running. Does NOT check
        model readiness — use ``/health/ready`` for that.

        Returns:
            Envelope with ``status: "alive"``.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_liveness))

    @endpoint("health.readiness")
    async def readiness(self) -> dict:
        """Kubernetes readiness probe.

        Returns ready=True when the service can accept traffic. Returns
        ready=False during startup before routers are registered.

        Returns:
            Envelope with ``status: "ready"``.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_readiness))

    @endpoint("health.detailed_health")
    async def detailed_health(self) -> dict:
        """Full system health with system metrics and GPU info.

        Includes everything from ``/health`` plus CPU/memory percentages,
        GPU backend and VRAM, registry health, training pool status,
        health score with diagnoses, trend histories, and recent errors.

        Cached for 2 seconds to avoid redundant psutil reads under
        concurrent polling.

        Returns:
            Envelope with ``system``, ``gpu``, ``registry``,
            ``health_score``, ``health_history``, ``memory_history``,
            ``recent_errors``, and all basic health fields.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_detailed_health))

    @endpoint("health.startup_progress")
    async def startup_progress(self) -> dict:
        """Current startup phase with staged loader status.

        Returns the lifecycle phase string (e.g. "running", "starting",
        "draining") from the startup progress tracker, plus the staged
        loader stage, timing information, hook statuses, and errors.

        Returns:
            Envelope with the current phase string and staged loader info.
        """
        from infrastructure.staged_loader import get_staged_loader

        loader = get_staged_loader()
        data = dict(STARTUP_PHASE)
        data["staged_loader"] = loader.get_status()
        return success_response(data=data)

    @endpoint("health.startup_status")
    async def startup_status(self) -> dict:
        """Detailed startup diagnostics.

        Returns comprehensive startup information including:
        - Current stage and elapsed time
        - Model load progress (0.0 to 1.0)
        - Per-hook timing and status
        - Per-stage timing
        - Any errors encountered

        Returns:
            Envelope with detailed startup diagnostics.
        """
        from infrastructure.staged_loader import get_staged_loader

        loader = get_staged_loader()
        status = loader.get_status()
        return success_response(data={
            "stage": status["stage"],
            "stage_value": status["stage_value"],
            "elapsed_seconds": status["elapsed_seconds"],
            "model_progress": status["model_progress"],
            "model_progress_message": status["model_progress_message"],
            "hooks": status["hooks"],
            "stages": status["stages"],
            "errors": status["errors"],
        })

    @endpoint("health.startup_history")
    async def startup_history(self) -> dict:
        """Startup performance history.

        Returns recent startup records, performance statistics,
        per-stage timing breakdowns, and alerts for slow or failed
        startups.

        Returns:
            Envelope with startup history, stats, stage breakdown, and alerts.
        """
        from infrastructure.startup_history import get_startup_history

        history = get_startup_history()
        return success_response(data={
            "records": history.get_records(limit=10),
            "stats": history.get_stats(),
            "stage_stats": history.get_stage_stats(),
            "slow_startups": history.get_slow_startups(threshold_seconds=60.0),
            "alerts": history.get_alerts(),
        })

    @endpoint("health.startup_diagnostics")
    async def startup_diagnostics(self) -> dict:
        """Startup diagnostics for debugging.

        Returns comprehensive startup debugging information including:
        - Current stage and timing
        - Model load progress
        - Per-hook status and timing
        - Error details
        - System resource usage during startup
        - Environment configuration

        Returns:
            Envelope with detailed startup diagnostics.
        """
        import psutil
        from infrastructure.staged_loader import get_staged_loader
        from infrastructure.startup_history import get_startup_history

        loader = get_staged_loader()
        history = get_startup_history()

        # Get current process info
        process = psutil.Process()
        mem_info = process.memory_info()

        # Get environment config
        import os
        env_config = {
            "autoload_model": os.environ.get("SLO_AUTOLOAD_MODEL", ""),
            "autoload_device": os.environ.get("SLO_AUTOLOAD_DEVICE", ""),
            "wanDB_enabled": os.environ.get("SLO_WANDB", "0") == "1",
            "log_level": os.environ.get("SLO_LOG_LEVEL", "INFO"),
        }

        return success_response(data={
            "stage": loader.stage_name,
            "stage_value": int(loader.stage),
            "elapsed_seconds": round(loader.elapsed, 1),
            "model_progress": round(loader._model_progress, 2),
            "model_progress_message": loader._model_progress_message,
            "hooks": {
                name: info.to_dict()
                for name, info in loader._hook_infos.items()
            },
            "errors": dict(loader._errors),
            "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
            "pid": process.pid,
            "env_config": env_config,
            "history_stats": history.get_stats(),
        })

    @endpoint("health.startup_config")
    async def startup_config(self) -> dict:
        """Startup configuration.

        Returns the current startup configuration including disabled
        hooks, custom timeouts, and stage overrides.

        Returns:
            Envelope with startup configuration.
        """
        from infrastructure.startup_config import get_startup_config

        config = get_startup_config()
        return success_response(data=config.to_dict())

    @endpoint("health.startup_health")
    async def startup_health(self) -> dict:
        """Quick startup health check.

        Returns a simple status indicating whether startup is complete.
        Useful for load balancers and monitoring systems that need a
        quick check without the full diagnostics payload.

        Returns:
            Envelope with startup health status.
        """
        from infrastructure.staged_loader import Stage, get_staged_loader

        loader = get_staged_loader()
        stage = loader.stage

        # Determine health based on stage
        if stage >= Stage.BACKGROUND:
            status = "healthy"
            message = "Startup complete"
        elif stage >= Stage.READY:
            status = "starting"
            message = "Model loading"
        elif stage >= Stage.CRITICAL:
            status = "starting"
            message = "Core services initializing"
        else:
            status = "initializing"
            message = "Server starting"

        return success_response(data={
            "status": status,
            "message": message,
            "stage": loader.stage_name,
            "elapsed_seconds": round(loader.elapsed, 1),
        })

    @endpoint("health.startup_compare")
    async def startup_compare(self, run_a: int = 0, run_b: int = 1) -> dict:
        """Compare two startup runs.

        Args:
            run_a: Index of first run (0 = oldest, -1 = most recent)
            run_b: Index of second run

        Returns:
            Envelope with comparison of two startup runs.
        """
        from infrastructure.startup_history import get_startup_history

        history = get_startup_history()
        records = history.get_records(limit=50)

        # Handle negative indices
        if run_a < 0:
            run_a = len(records) + run_a
        if run_b < 0:
            run_b = len(records) + run_b

        comparison = history.compare_runs(run_a, run_b)
        if comparison is None:
            return success_response(data={
                "error": "Invalid run indices",
                "available_runs": len(records),
            })

        return success_response(data=comparison)

    @endpoint("health.startup_rollback")
    async def startup_rollback(self) -> dict:
        """Startup rollback status and controls.

        Returns the current rollback state, registered actions,
        and provides an endpoint to trigger rollback manually.

        Returns:
            Envelope with rollback status.
        """
        from infrastructure.startup_rollback import get_startup_rollback

        rollback = get_startup_rollback()
        return success_response(data=rollback.get_status())

    async def startup_stream(self, request: Request) -> StreamingResponse:
        """SSE stream for real-time startup progress updates.

        Pushes startup status every 1 second during startup.
        Stops pushing once the server reaches BACKGROUND stage.

        Returns:
            SSE stream with startup progress events.
        """
        from infrastructure.staged_loader import Stage, get_staged_loader

        async def generate():
            loader = get_staged_loader()
            event_count = 0

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Get current status
                status = loader.get_status()
                stage = status.get("stage", "unknown")

                # Send event
                data = json.dumps({
                    "stream": "startup",
                    "data": {
                        "stage": stage,
                        "stage_value": status.get("stage_value", 0),
                        "elapsed_seconds": status.get("elapsed_seconds", 0),
                        "model_progress": status.get("model_progress", 0),
                        "model_progress_message": status.get("model_progress_message", ""),
                        "hooks": status.get("hooks", {}),
                        "errors": status.get("errors", {}),
                    },
                    "event": f"startup_{stage}",
                    "id": event_count,
                })
                yield f"data: {data}\n\n"
                event_count += 1

                # Stop pushing once we reach BACKGROUND stage
                if stage in ("background", "ready"):
                    # Send final event
                    final_data = json.dumps({
                        "stream": "startup",
                        "data": {
                            "stage": "complete",
                            "stage_value": 3,
                            "elapsed_seconds": status.get("elapsed_seconds", 0),
                            "model_progress": 1.0,
                            "model_progress_message": "Startup complete",
                        },
                        "event": "startup_complete",
                        "id": event_count,
                    })
                    yield f"data: {final_data}\n\n"
                    break

                # Wait before next update
                await asyncio.sleep(1.0)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @endpoint("health.debug_info")
    async def debug_info(self) -> dict:
        """Debug information for troubleshooting.

        Subset of ``/health/detailed`` focused on model state, inference
        metrics, and error histories. Lighter than detailed health —
        does not read CPU/memory or GPU info.

        Returns:
            Envelope with ``model_loaded``, ``model_type``, ``soul``,
            ``health_score``, ``model_metrics``, ``recent_errors``.
        """
        ctrl = get_health_controller()
        detailed = await asyncio.to_thread(ctrl.get_detailed_health)
        return success_response(
            data={
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
        )

    @endpoint("health.model_health")
    async def model_health(self) -> dict:
        """Model health monitor stats.

        Returns perplexity trends, loss history, and quality scores from
        the model health monitor. Lazy-attaches the current model on
        first call if not already set.

        Side effects:
            Attaches ``server_state.model`` to the health monitor on
            first call when the model is loaded but monitor is uninitialised.

        Returns:
            Envelope with ``status: "ok"`` plus monitor stats, or raises
            a classified error if the monitor is unavailable.
        """
        from domains.feedback.model_health import get_health_monitor

        mon = get_health_monitor()
        import state as server_state

        if server_state.model is not None and mon._model is None:
            mon.set_model(server_state.model, server_state.tokenizer)
        stats = mon.get_stats()
        return success_response(data={"status": "ok", **stats})

    @endpoint("health.health_summary")
    async def health_summary(self) -> dict:
        """Condensed health score and key metrics.

        Returns the numeric health score (0-100), status label, human
        summary, and top-level system metrics. Designed for the frontend
        status bar — lighter than ``/health/detailed``.

        During cold start, ``get_detailed_health`` can block on first-time
        module imports. A 5-second timeout with a lightweight fallback
        prevents the frontend status bar from hanging.

        Returns:
            Envelope with ``score``, ``status``, ``summary``,
            ``diagnoses``, ``model_loaded``, ``cpu_percent``,
            ``memory_percent``.
        """
        ctrl = get_health_controller()
        try:
            detailed = await asyncio.wait_for(
                asyncio.to_thread(ctrl.get_detailed_health),
                timeout=5.0,
            )
            hs = detailed.get("health_score", {})
            data = {
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
            }
        except (asyncio.TimeoutError, Exception):
            # Fast fallback during cold start
            import state as server_state

            data = {
                "score": 0,
                "status": "starting",
                "summary": "Server is starting up.",
                "diagnoses": [],
                "model_loaded": server_state.model is not None
                or server_state.provider is not None,
                "model_loading": STARTUP_PHASE.get("phase") == "loading_model",
                "model_type": getattr(server_state, "model_type", None),
                "soul": None,
                "uptime_seconds": 0,
                "request_count": 0,
                "error_count": 0,
                "tokens_per_sec": 0,
                "cpu_percent": None,
                "memory_percent": None,
            }
        return success_response(data=data)

    def _build_health_snapshot(self, ctrl) -> dict:
        """Build a single SSE health snapshot.

        Calls the controller synchronously — runs in a thread pool from
        the async ``health_stream`` generator. Returns a complete standard
        envelope ``{stream, phase, status, data, meta, message}``.

        Args:
            ctrl: The ``HealthController`` instance.

        Returns:
            Standard SSE envelope dict ready for JSON serialisation.
        """
        detailed = ctrl.get_detailed_health()
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
                "is_inferencing": detailed.get("is_inferencing", False),
                "inference_count": detailed.get("inference_count", 0),
                "uptime_seconds": detailed.get("uptime_seconds", 0),
                "request_count": detailed.get("request_count", 0),
                "error_count": detailed.get("error_count", 0),
                "tokens_per_sec": detailed.get("tokens_per_sec", 0),
                "avg_latency_ms": detailed.get("avg_latency_ms", 0),
                "requests_per_minute": detailed.get("requests_per_minute", 0),
                "total_tokens": detailed.get("total_tokens", 0),
                "avg_tokens_per_request": detailed.get("avg_tokens_per_request", 0),
                "cpu_percent": detailed.get("system", {}).get("cpu_percent"),
                "memory_percent": detailed.get("system", {}).get("memory_percent"),
                "health_score": hs.get("score", 0),
                "health_status": hs.get("status", "unknown"),
                "health_summary": hs.get("summary", ""),
                "diagnoses": hs.get("diagnoses", []),
                "num_parameters": detailed.get("num_parameters"),
                "quantization": detailed.get("quantization"),
                "training_pool": detailed.get("training_pool"),
                "model_metrics": detailed.get("model_metrics", []),
                "model_events": detailed.get("model_events", []),
                "health_history": detailed.get("health_history", []),
                "memory_history": detailed.get("memory_history", []),
                "rate_violations": detailed.get("rate_violations", []),
                "path_latencies": detailed.get("path_latencies", []),
                "recent_errors": detailed.get("recent_errors", []),
            },
            "meta": {"ts": time.time()},
            "message": hs.get("summary", ""),
        }

    @endpoint("health.health_stream")
    async def health_stream(self, request: Request) -> StreamingResponse:
        """SSE endpoint pushing health snapshots every 3 seconds.

        Holds the connection open until the client disconnects. Each
        event is a ``data:`` line containing the full health snapshot
        envelope (same shape as ``/health/detailed`` but wrapped in the
        standard SSE envelope).

        Args:
            request: FastAPI Request — checked for disconnect each cycle.

        Returns:
            StreamingResponse with ``text/event-stream`` content type.
        """
        ctrl = get_health_controller()

        async def generate() -> AsyncGenerator[str, None]:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snapshot = await asyncio.to_thread(self._build_health_snapshot, ctrl)
                    yield "data: " + json.dumps(snapshot, default=str) + "\n\n"
                except Exception as e:
                    logger.warning("Health stream snapshot failed: %s", e)
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "stream": "health",
                                "phase": "ERROR",
                                "status": "error",
                                "data": {"error": str(e)},
                                "message": str(e),
                            }
                        )
                        + "\n\n"
                    )
                await asyncio.sleep(self.HEALTH_STREAM_INTERVAL)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @endpoint("health.services")
    async def services_health(self) -> dict:
        """Health check for all subsystems: training, settings, plugins, cloud."""
        services = {}
        # Training
        try:
            from domains.training.outcome_tracker import TrainingOutcomeTracker
            tracker = TrainingOutcomeTracker()
            stats = tracker.get_stats()
            services["training"] = {"status": "ok", "total_runs": stats.get("total_runs", 0)}
        except Exception as e:
            services["training"] = {"status": "error", "error": str(e)}
        # Settings
        try:
            from domains.settings.persistent import get_settings
            ps = get_settings()
            services["settings"] = {"status": "ok", "sections": list(vars(ps.settings).keys())}
        except Exception as e:
            services["settings"] = {"status": "error", "error": str(e)}
        # Plugins
        try:
            from domains.plugins import PluginManager
            pm = PluginManager()
            services["plugins"] = {"status": "ok", "loaded": len(pm.list_plugins())}
        except Exception as e:
            services["plugins"] = {"status": "error", "error": str(e)}
        # Adaptive engine
        try:
            from domains.training.adaptive_config import AdaptiveConfigEngine
            AdaptiveConfigEngine()
            services["adaptive"] = {"status": "ok"}
        except Exception as e:
            services["adaptive"] = {"status": "error", "error": str(e)}
        healthy = all(s["status"] == "ok" for s in services.values())
        return success_response(data={"status": "healthy" if healthy else "degraded", "services": services})


router = HealthRouter().router
