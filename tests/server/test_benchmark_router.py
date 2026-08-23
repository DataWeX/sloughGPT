"""
Tests for the benchmark router — run, metrics, quality, responses, stats, clear.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.benchmark import router


def _fake_provider(ids, logits=None):
    """Build a SloNetChatProvider stand-in with a numpy forward pass."""
    import numpy as np

    class _Logits:
        def __init__(self, arr):
            self.data = arr

    class _Model:
        def __init__(self, arr):
            self._arr = arr

        def forward(self, input_ids, targets=None):
            if self._arr is None:
                raise RuntimeError("no logits")
            return _Logits(self._arr), None

    class _Provider:
        def __init__(self, token_ids, arr):
            self._token_ids = token_ids
            self._arr = arr

        def tokenize(self, text):
            return list(self._token_ids)

        def _get_model(self):
            return _Model(self._arr)

    if logits is None:
        logits = np.zeros((1, len(ids), 3), dtype=np.float64)
    return _Provider(list(ids), logits)


def _patch_server(provider):
    """Replace the ServerState singleton with a fake holding ``provider``."""
    fake_core = MagicMock()
    fake_core.model.get.return_value = provider
    return patch("domains.infrastructure.server_state.get_server_state",
                 return_value=fake_core)


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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
        with _patch_server(None):
            resp = client.post("/benchmark/perplexity")
        assert resp.status_code == 400

    def test_accepts_text_param(self, client):
        with _patch_server(None):
            resp = client.post("/benchmark/perplexity?text=hello world")
        assert resp.status_code == 400


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


class TestBenchmarkMethodMismatch:
    """Wrong HTTP methods on benchmark routes."""

    def test_run_get_matches_model_lookup(self, client):
        resp = client.get("/benchmark/run")
        assert resp.status_code in (200, 404)

    def test_metrics_post_405(self, client):
        resp = client.post("/benchmark/metrics")
        assert resp.status_code == 405

    def test_quality_post_405(self, client):
        resp = client.post("/benchmark/quality")
        assert resp.status_code == 405

    def test_responses_post_405(self, client):
        resp = client.post("/benchmark/responses")
        assert resp.status_code == 405

    def test_stats_post_405(self, client):
        resp = client.post("/benchmark/stats")
        assert resp.status_code == 405

    def test_clear_get_405(self, client):
        resp = client.get("/benchmark/history/clear")
        assert resp.status_code == 405

    def test_perplexity_get_matches_model_lookup(self, client):
        resp = client.get("/benchmark/perplexity")
        assert resp.status_code in (200, 404)


class TestPerplexityPath:
    """Perplexity with a working SloNet provider (pure NumPy)."""

    def test_perplexity_no_provider_returns_400(self, client):
        with _patch_server(None):
            resp = client.post("/benchmark/perplexity?text=hello")
        assert resp.status_code == 400

    def test_perplexity_model_not_loaded_returns_400(self, client):
        class _NoModel:
            def tokenize(self, text):
                return [1, 2, 3]

            def _get_model(self):
                return None

        with _patch_server(_NoModel()):
            resp = client.post("/benchmark/perplexity?text=hello")
        assert resp.status_code == 400

    def test_perplexity_too_few_tokens_returns_400(self, client):
        provider = _fake_provider([5])
        with _patch_server(provider):
            resp = client.post("/benchmark/perplexity?text=hi")
        assert resp.status_code == 400

    def test_perplexity_error_returns_500(self, client):
        with patch("domains.infrastructure.server_state.get_server_state",
                   side_effect=RuntimeError("controller crash")), \
             patch("domains.infrastructure.errors.emit_error_event"):
            resp = client.post("/benchmark/perplexity?text=hello")
        assert resp.status_code == 500

    def test_perplexity_computes_value(self, client):
        import numpy as np
        # ids [0, 1, 2]; successors [1, 2] scored with certainty → ppl 1.0
        logits = np.zeros((1, 3, 3), dtype=np.float64)
        logits[0, 0, 1] = 100.0
        logits[0, 1, 2] = 100.0
        provider = _fake_provider([0, 1, 2], logits)
        with _patch_server(provider):
            resp = client.post("/benchmark/perplexity?text=hello")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["perplexity"] == 1.0
        assert data["loss"] == 0.0
        assert data["tokens"] == 3

    def test_perplexity_uses_text_preview(self, client):
        import numpy as np
        logits = np.zeros((1, 3, 3), dtype=np.float64)
        logits[0, 0, 1] = 100.0
        logits[0, 1, 2] = 100.0
        provider = _fake_provider([0, 1, 2], logits)
        with _patch_server(provider):
            resp = client.post("/benchmark/perplexity?text=a%20very%20long%20sentence%20that%20exceeds%20thirty%20chars")
        assert resp.json()["data"]["text"] == "a very long sentence that exce"


class TestErrorPaths:
    """Exception propagation in benchmark endpoints."""

    @patch("domains.get_benchmark_domain")
    def test_quality_error_raises_500(self, mock_get_bench, client):
        mock_get_bench.side_effect = RuntimeError("bench down")
        resp = client.get("/benchmark/quality")
        assert resp.status_code == 500

    @patch("domains.feedback.response_tracker.get_response_tracker")
    def test_responses_error_raises_500(self, mock_get_tracker, client):
        mock_get_tracker.side_effect = RuntimeError("tracker down")
        resp = client.get("/benchmark/responses")
        assert resp.status_code == 500

    @patch("domains.get_benchmark_domain")
    def test_stats_error_raises_500(self, mock_get_bench, client):
        mock_get_bench.side_effect = RuntimeError("stats down")
        resp = client.get("/benchmark/stats")
        assert resp.status_code == 500

    @patch("domains.get_benchmark_domain")
    def test_clear_error_raises_500(self, mock_get_bench, client):
        mock_get_bench.side_effect = RuntimeError("clear down")
        resp = client.post("/benchmark/history/clear")
        assert resp.status_code == 500
