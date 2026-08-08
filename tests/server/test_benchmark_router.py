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

    def test_run_returns_model_info(self, client):
        resp = client.post("/benchmark/run")
        data = resp.json()["data"]
        assert "model" in data
        assert "model_loaded" in data

    def test_forwards_model_param(self, client):
        resp = client.post("/benchmark/run?model=my-model")
        assert resp.json()["data"]["model"] == "my-model"

    @patch("apps.api.server.routers.benchmark.BenchmarkRouter._get_model_metrics")
    def test_uses_injected_metrics(self, mock_metrics, client):
        mock_metrics.return_value = {"model": "gpt2", "model_loaded": True, "tokens_per_second": 3.5}
        resp = client.post("/benchmark/run")
        assert resp.json()["data"]["model_loaded"] is True
        assert resp.json()["data"]["tokens_per_second"] == 3.5


class TestGetMetrics:
    def test_returns_metrics(self, client):
        resp = client.get("/benchmark/metrics")
        assert resp.status_code == 200

    def test_metrics_has_structure(self, client):
        resp = client.get("/benchmark/metrics")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, (dict, list))

    def test_metrics_reports_unloaded(self, client):
        resp = client.get("/benchmark/metrics")
        assert resp.json()["data"]["model_loaded"] in (True, False)

    @patch("apps.api.server.routers.benchmark.BenchmarkRouter._get_model_metrics")
    def test_metrics_forwards_model(self, mock_metrics, client):
        mock_metrics.return_value = {"model": "gpt2", "model_loaded": False}
        resp = client.get("/benchmark/metrics?model=custom")
        mock_metrics.assert_called_once_with("custom")


class TestPerplexity:
    def test_requires_loaded_model(self, client):
        resp = client.post("/benchmark/perplexity")
        assert resp.status_code in (400, 500)

    def test_accepts_text_param(self, client):
        resp = client.post("/benchmark/perplexity?text=hello world")
        assert resp.status_code in (400, 500)


class TestQuality:
    @patch("domains.get_benchmark_domain")
    def test_returns_quality(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.evaluate_latest.return_value = {"coherence": 0.8}
        resp = client.get("/benchmark/quality")
        assert resp.status_code == 200

    @patch("domains.get_benchmark_domain")
    def test_quality_empty(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.evaluate_latest.return_value = {}
        resp = client.get("/benchmark/quality")
        assert resp.status_code == 200

    @patch("domains.get_benchmark_domain")
    def test_quality_forwards_limit(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.evaluate_latest.return_value = {}
        resp = client.get("/benchmark/quality?limit=10&model=gpt2")
        assert resp.status_code == 200
        bench.evaluate_latest.assert_called_once_with(limit=10)


class TestLoggedResponses:
    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_returns_responses(self, mock_get_tracker, client):
        tracker = mock_get_tracker.return_value
        tracker.get_responses.return_value = []
        resp = client.get("/benchmark/responses")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 0

    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_returns_empty(self, mock_get_tracker, client):
        tracker = mock_get_tracker.return_value
        tracker.get_responses.return_value = []
        resp = client.get("/benchmark/responses?limit=50")
        assert resp.status_code == 200

    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_serializes_tracker_entries(self, mock_get_tracker, client):
        class FakeResp:
            timestamp = "2026-01-01T00:00:00"
            user_message = "hi"
            assistant_response = "hello"
            model = "gpt2"
            tokens_generated = 4
            duration_ms = 10

        tracker = mock_get_tracker.return_value
        tracker.get_responses.return_value = [FakeResp()]
        resp = client.get("/benchmark/responses")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1
        assert data["responses"][0]["tokens_generated"] == 4
        assert data["responses"][0]["model"] == "gpt2"

    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_forwards_model_filter(self, mock_get_tracker, client):
        tracker = mock_get_tracker.return_value
        tracker.get_responses.return_value = []
        resp = client.get("/benchmark/responses?model=my-model")
        assert resp.status_code == 200
        tracker.get_responses.assert_called_once_with(limit=20, model="my-model")


class TestTrackerStats:
    @patch("domains.get_benchmark_domain")
    def test_returns_stats(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.get_stats.return_value = {"total_responses": 5}
        resp = client.get("/benchmark/stats")
        assert resp.status_code == 200

    @patch("domains.get_benchmark_domain")
    def test_stats_empty(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        bench.get_stats.return_value = {}
        resp = client.get("/benchmark/stats")
        assert resp.status_code == 200


class TestClearHistory:
    @patch("domains.get_benchmark_domain")
    def test_clears_history(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        resp = client.post("/benchmark/history/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True

    @patch("domains.get_benchmark_domain")
    def test_clear_returns_success(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        resp = client.post("/benchmark/history/clear")
        assert resp.json()["status"] == "success"

    @patch("domains.get_benchmark_domain")
    def test_clear_calls_history(self, mock_get_bench, client):
        bench = mock_get_bench.return_value
        client.post("/benchmark/history/clear")
        bench.clear_history.assert_called_once()
