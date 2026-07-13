"""
Tests for metrics router — /metrics and /metrics/prometheus.
"""
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.metrics import router as metrics_router

app = FastAPI()
app.include_router(metrics_router)
client = TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestMetrics:

    def test_get_metrics_structure(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = _data(resp)
        assert isinstance(data, dict)
        assert "inferences_total" in data
        assert "uptime_seconds" in data

    def test_get_metrics_types(self):
        resp = client.get("/metrics")
        data = _data(resp)
        assert isinstance(data["inferences_total"], int)
        assert isinstance(data["active_requests"], int)
        assert isinstance(data["model_loaded"], bool)


class TestPrometheus:

    def test_prometheus_initial(self):
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        text = resp.text
        assert len(text) > 0

    def test_prometheus_contains_uptime(self):
        resp = client.get("/metrics/prometheus")
        assert "sloughgpt_uptime_seconds" in resp.text

    def test_prometheus_contains_model_loaded(self):
        resp = client.get("/metrics/prometheus")
        assert "sloughgpt_model_loaded" in resp.text
