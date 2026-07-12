"""
Metrics Router — Prometheus and internal metrics endpoints.

Uses the ``MetricsCollector`` for full request/inference tracking and
Prometheus text exposition format.
"""

from fastapi import APIRouter, Response
from domains.infrastructure.metrics import get_metrics_collector
from schemas.common import success_response

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_metrics():
    """Get internal request metrics (JSON)."""
    c = get_metrics_collector()
    return success_response(data={
        "uptime_seconds": f"{c._start_time}",
        "model_loaded": c._model_loaded,
        "model_name": c._model_name,
        "active_requests": c._active_requests,
        "inferences_total": c._inference_count,
        "tokens_generated_total": c._tokens_generated,
    })


@router.get("/prometheus")
async def prometheus_metrics():
    """Prometheus-compatible metrics output (text/plain)."""
    c = get_metrics_collector()
    body = c.render()
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")
