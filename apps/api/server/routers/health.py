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
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from controllers.health import get_health_controller
from startup_progress import STARTUP_PHASE
from schemas.common import success_response, classify_and_raise


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
        self.router.add_api_route("/debug", self.debug_info, methods=["GET"])
        self.router.add_api_route("/model", self.model_health, methods=["GET"])
        self.router.add_api_route("/summary", self.health_summary, methods=["GET"])
        self.router.add_api_route("/stream", self.health_stream, methods=["GET"])

    async def health(self) -> dict:
        """Basic health status.

        Returns model loaded state, device, inference count, lifecycle
        phase, and resource allocation. This is the primary health check
        for load balancers and monitoring dashboards.

        Returns:
            Envelope with ``model_loaded``, ``model_type``, ``device``,
            ``is_inferencing``, ``lifecycle``, ``resource_allocation``.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_basic_health))

    async def liveness(self) -> dict:
        """Kubernetes liveness probe.

        Returns alive=True when the process is running. Does NOT check
        model readiness — use ``/health/ready`` for that.

        Returns:
            Envelope with ``status: "alive"``.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_liveness))

    async def readiness(self) -> dict:
        """Kubernetes readiness probe.

        Returns ready=True when the service can accept traffic. Returns
        ready=False during startup before routers are registered.

        Returns:
            Envelope with ``status: "ready"``.
        """
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_readiness))

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

    async def startup_progress(self) -> dict:
        """Current startup phase.

        Returns the lifecycle phase string (e.g. "running", "starting",
        "draining") from the startup progress tracker.

        Returns:
            Envelope with the current phase string.
        """
        return success_response(data=STARTUP_PHASE)

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
        try:
            from domains.feedback.model_health import get_health_monitor
            mon = get_health_monitor()
            import state as server_state
            if server_state.model is not None and mon._model is None:
                mon.set_model(server_state.model, server_state.tokenizer)
            stats = mon.get_stats()
            return success_response(data={"status": "ok", **stats})
        except Exception as e:
            logger.warning("Health monitor failed: %s", e)
            classify_and_raise(e, source="health_model_health")

    async def health_summary(self) -> dict:
        """Condensed health score and key metrics.

        Returns the numeric health score (0-100), status label, human
        summary, and top-level system metrics. Designed for the frontend
        status bar — lighter than ``/health/detailed``.

        Returns:
            Envelope with ``score``, ``status``, ``summary``,
            ``diagnoses``, ``model_loaded``, ``cpu_percent``,
            ``memory_percent``.
        """
        ctrl = get_health_controller()
        detailed = await asyncio.to_thread(ctrl.get_detailed_health)
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
                    snapshot = await asyncio.to_thread(
                        self._build_health_snapshot, ctrl
                    )
                    yield "data: " + json.dumps(snapshot, default=str) + "\n\n"
                except Exception as e:
                    logger.warning("Health stream snapshot failed: %s", e)
                    classify_and_raise(e, source="health_stream")
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


router = HealthRouter().router
