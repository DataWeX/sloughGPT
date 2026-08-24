"""
Metrics Router — Prometheus and internal metrics endpoints.

Uses the ``MetricsCollector`` for full request/inference tracking and
Prometheus text exposition format.
"""

from fastapi import APIRouter, Response
from domains.infrastructure.metrics import get_metrics_collector
from schemas.common import success_response, classify_and_raise


class MetricsRouter:
    """MetricsRouter — Prometheus and internal metrics endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/metrics", tags=["metrics"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            path="",
            endpoint=self.get_metrics,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/prometheus",
            endpoint=self.prometheus_metrics,
            methods=["GET"],
        )

    async def get_metrics(self) -> dict:
        """Get internal request metrics (JSON)."""
        try:
            c = get_metrics_collector()
            import state as server_state
            model_loaded = c._model_loaded
            model_name = c._model_name
            if server_state.model is not None or server_state.provider is not None:
                model_loaded = True
                model_name = server_state.model_type or model_name
            return success_response(data={
                "uptime_seconds": f"{c._start_time}",
                "model_loaded": model_loaded,
                "model_name": model_name,
                "active_requests": c._active_requests,
                "inferences_total": c._inference_count,
                "tokens_generated_total": c._tokens_generated,
            })
        except Exception as e:
            classify_and_raise(e, source="metrics.get")

    async def prometheus_metrics(self) -> dict:
        """Render server metrics in Prometheus text exposition format."""
        try:
            c = get_metrics_collector()
            body = c.render()
            return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")
        except Exception as e:
            classify_and_raise(e, source="metrics.prometheus")


router = MetricsRouter().router
