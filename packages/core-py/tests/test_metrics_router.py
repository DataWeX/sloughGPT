"""Tests for the metrics API router (routers/metrics.py).

Covers: get_metrics, prometheus_metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.metrics import MetricsRouter  # noqa: E402


def _mock_collector() -> MagicMock:
    c = MagicMock()
    c._start_time = 1000.0
    c._model_loaded = False
    c._model_name = "none"
    c._active_requests = 0
    c._inference_count = 0
    c._tokens_generated = 0
    c.render.return_value = "# HELP slo_up Uptime\nslo_up 1000\n"
    return c


def _app(mr: MetricsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(mr.router)
    return app


class TestGetMetrics:
    @patch("routers.metrics.get_metrics_collector")
    def test_metrics(self, mock_get):
        mock_get.return_value = _mock_collector()
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "uptime_seconds" in data
        assert "model_loaded" in data
        assert "inferences_total" in data
        assert "tokens_generated_total" in data


class TestPrometheusMetrics:
    @patch("routers.metrics.get_metrics_collector")
    def test_prometheus(self, mock_get):
        mock_get.return_value = _mock_collector()
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "slo_up" in resp.text
