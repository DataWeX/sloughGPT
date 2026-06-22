"""
Tests for metrics router — /metrics and /metrics/prometheus.
"""
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.metrics import router as metrics_router, increment_request_counter, record_latency

app = FastAPI()
app.include_router(metrics_router)
client = TestClient(app)


class TestMetrics:

    def test_get_metrics_initial(self):
        # Reset by re-importing
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_total" in data
        assert "started_at" in data
        assert "latency_buckets" in data

    def test_get_metrics_after_increment(self):
        increment_request_counter()
        record_latency(150.0)
        resp = client.get("/metrics")
        data = resp.json()
        assert data["requests_total"] > 0
        # latency_buckets should have at least one entry (previous run may have set this)
        # Asserting existence is enough — ordering may vary
        assert len(data["latency_buckets"]) >= 1

    def test_get_metrics_structure(self):
        resp = client.get("/metrics")
        data = resp.json()
        assert isinstance(data["requests_total"], int)
        assert isinstance(data["latency_buckets"], dict)


class TestPrometheus:

    def test_prometheus_initial(self):
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        # FastAPI serializes string return as JSON, so we get a quoted string
        text = resp.json() if isinstance(resp.json(), str) else ""
        assert text.startswith("# HELP")

    def test_prometheus_contains_http_requests_total(self):
        resp = client.get("/metrics/prometheus")
        text = resp.json() if isinstance(resp.json(), str) else ""
        assert "http_requests_total" in text

    def test_prometheus_contains_request_duration(self):
        resp = client.get("/metrics/prometheus")
        text = resp.json() if isinstance(resp.json(), str) else ""
        assert "http_request_duration_ms" in text
