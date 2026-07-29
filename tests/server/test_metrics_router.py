"""
Tests for the metrics router — GET /metrics and GET /metrics/prometheus.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.metrics import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetMetrics:
    """GET /metrics"""

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_returns_metrics_json(self, mock_get_coll, client):
        coll = MagicMock()
        coll._start_time = 100.0
        coll._model_loaded = True
        coll._model_name = "gpt2"
        coll._active_requests = 2
        coll._inference_count = 42
        coll._tokens_generated = 5000
        mock_get_coll.return_value = coll
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["inferences_total"] == 42
        assert data["tokens_generated_total"] == 5000


class TestPrometheusMetrics:
    """GET /metrics/prometheus"""

    @patch("apps.api.server.routers.metrics.get_metrics_collector")
    def test_returns_prometheus_text(self, mock_get_coll, client):
        coll = MagicMock()
        coll.render.return_value = "# HELP ..."
        mock_get_coll.return_value = coll
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert resp.text == "# HELP ..."
        assert resp.headers["content-type"].startswith("text/plain")
