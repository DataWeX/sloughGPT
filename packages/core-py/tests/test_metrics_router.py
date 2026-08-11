"""Tests for the metrics API router (routers/metrics.py).

Covers: get_metrics, prometheus_metrics, edge cases.
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


def _mock_collector(**overrides) -> MagicMock:
    c = MagicMock()
    c._start_time = overrides.get("start_time", 1000.0)
    c._model_loaded = overrides.get("model_loaded", False)
    c._model_name = overrides.get("model_name", "none")
    c._active_requests = overrides.get("active_requests", 0)
    c._inference_count = overrides.get("inference_count", 0)
    c._tokens_generated = overrides.get("tokens_generated", 0)
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

    @patch("routers.metrics.get_metrics_collector")
    def test_metrics_model_loaded(self, mock_get):
        mock_get.return_value = _mock_collector(model_loaded=True, model_name="gpt2")
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["model_name"] == "gpt2"

    @patch("routers.metrics.get_metrics_collector")
    def test_metrics_with_inferences(self, mock_get):
        mock_get.return_value = _mock_collector(inference_count=42, tokens_generated=1500)
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert data["inferences_total"] == 42
        assert data["tokens_generated_total"] == 1500

    @patch("routers.metrics.get_metrics_collector")
    def test_metrics_uptime_positive(self, mock_get):
        mock_get.return_value = _mock_collector(start_time=500.0)
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics")
        data = resp.json()["data"]
        assert float(data["uptime_seconds"]) >= 0


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

    @patch("routers.metrics.get_metrics_collector")
    def test_prometheus_content_type(self, mock_get):
        mock_get.return_value = _mock_collector()
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics/prometheus")
        assert "charset" in resp.headers["content-type"]

    @patch("routers.metrics.get_metrics_collector")
    def test_prometheus_with_model(self, mock_get):
        mock_get.return_value = _mock_collector(model_loaded=True, model_name="qwen")
        mr = MetricsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
