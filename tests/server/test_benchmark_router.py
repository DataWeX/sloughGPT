"""
Tests for the benchmark router — run, metrics, quality, responses, stats, clear.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.benchmark import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestRunBenchmark:
    def test_returns_metrics(self, client):
        resp = client.post("/benchmark/run")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "model_loaded" in data


class TestGetMetrics:
    def test_returns_metrics(self, client):
        resp = client.get("/benchmark/metrics")
        assert resp.status_code == 200


class TestQuality:
    @patch("domains.get_benchmark_domain")
    def test_returns_quality(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.evaluate_latest.return_value = {"coherence": 0.8}
        resp = client.get("/benchmark/quality")
        assert resp.status_code == 200


class TestLoggedResponses:
    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_returns_responses(self, mock_get_tracker, client):
        tracker = mock_get_tracker.return_value
        tracker.get_responses.return_value = []
        resp = client.get("/benchmark/responses")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0


class TestTrackerStats:
    @patch("domains.get_benchmark_domain")
    def test_returns_stats(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.get_stats.return_value = {"total_responses": 5}
        resp = client.get("/benchmark/stats")
        assert resp.status_code == 200


class TestClearHistory:
    @patch("domains.get_benchmark_domain")
    def test_clears_history(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        resp = client.post("/benchmark/history/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
