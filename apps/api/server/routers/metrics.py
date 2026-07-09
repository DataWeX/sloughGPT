"""
Metrics Router — Prometheus and internal metrics endpoints.

Self-contained metrics tracking — does not depend on main.py.
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import APIRouter

from schemas.common import success_response

router = APIRouter(prefix="/metrics", tags=["metrics"])

_SIMPLE_METRICS = {
    "request_counter": 0,
    "started_at": datetime.now().isoformat(),
    "latency_buckets": defaultdict(float),
}


def increment_request_counter():
    _SIMPLE_METRICS["request_counter"] += 1


def record_latency(duration_ms: float):
    bucket = int(duration_ms / 100) * 100
    _SIMPLE_METRICS["latency_buckets"][f"{bucket}-{bucket+99}ms"] += 1


@router.get("")
async def get_metrics():
    """Get internal request metrics."""
    total = _SIMPLE_METRICS["request_counter"]
    return success_response(data={
        "requests_total": total,
        "started_at": _SIMPLE_METRICS["started_at"],
        "latency_buckets": dict(_SIMPLE_METRICS["latency_buckets"]),
    })


@router.get("/prometheus")
async def prometheus_metrics():
    """Prometheus-compatible metrics output."""
    total = _SIMPLE_METRICS["request_counter"]
    lines = [
        "# HELP http_requests_total Total HTTP requests",
        "# TYPE http_requests_total counter",
        f"http_requests_total {total}",
        "",
        "# HELP http_request_duration_ms Request duration buckets",
        "# TYPE http_request_duration_ms gauge",
    ]
    for bucket, count in sorted(_SIMPLE_METRICS["latency_buckets"].items()):
        lines.append(f'http_request_duration_ms_bucket{{le="{bucket}"}} {int(count)}')
    return "\n".join(lines) + "\n"
