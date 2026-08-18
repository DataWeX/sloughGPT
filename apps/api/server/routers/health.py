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
from schemas.common import success_response, raise_error, classify_and_raise


class HealthRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/health", tags=["health"])
        self.HEALTH_STREAM_INTERVAL = 3.0
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.health, methods=["GET"])
        self.router.add_api_route("/live", self.liveness, methods=["GET"])
        self.router.add_api_route("/ready", self.readiness, methods=["GET"])
        self.router.add_api_route("/detailed", self.detailed_health, methods=["GET"])
        self.router.add_api_route("/startup-progress", self.startup_progress, methods=["GET"])
        self.router.add_api_route("/debug", self.debug_info, methods=["GET"])
        self.router.add_api_route("/model", self.model_health, methods=["GET"])
        self.router.add_api_route("/summary", self.health_summary, methods=["GET"])
        self.router.add_api_route("/stream", self.health_stream, methods=["GET"])

    async def health(self):
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_basic_health))

    async def liveness(self):
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_liveness))

    async def readiness(self):
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_readiness))

    async def detailed_health(self):
        ctrl = get_health_controller()
        return success_response(data=await asyncio.to_thread(ctrl.get_detailed_health))

    async def startup_progress(self):
        return success_response(data=STARTUP_PHASE)

    async def debug_info(self):
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

    async def model_health(self):
        try:
            from domains.feedback.model_health import get_health_monitor
            mon = get_health_monitor()
            import state as server_state
            if server_state.model is not None and mon._model is None:
                mon.set_model(server_state.model, server_state.tokenizer)
            stats = mon.get_stats()
            return success_response(data={"status": "ok", **stats})
        except Exception as e:
            classify_and_raise(e, source="health_model_health")
            raise_error(str(e), "E_INFRA_REGISTRY")

    async def health_summary(self):
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

    async def health_stream(self, request: Request):
        ctrl = get_health_controller()

        async def generate():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # _build_health_snapshot returns a complete standard envelope
                    # {stream, phase, status, data, meta, message}; yield it directly.
                    snapshot = await asyncio.to_thread(self._build_health_snapshot, ctrl)
                    yield "data: " + json.dumps(snapshot, default=str) + "\n\n"
                except Exception as e:
                    classify_and_raise(e, source="health_stream")
                    pass
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
