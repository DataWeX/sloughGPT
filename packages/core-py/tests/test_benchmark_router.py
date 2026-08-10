"""Tests for the benchmark API router (routers/benchmark.py).

Covers: run_benchmark, get_model_metrics, get_quality_metrics, get_logged_responses,
get_tracker_stats, clear_history. Domain deps are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.benchmark import BenchmarkRouter  # noqa: E402


def _app(br: BenchmarkRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(br.router)
    return app


class TestRunBenchmark:
    def test_run_benchmark_default(self):
        br = BenchmarkRouter()
        mock_ctrl = MagicMock()
        mock_ctrl._hf_model = None
        with patch("controllers.models.get_models_controller", return_value=mock_ctrl):
            client = TestClient(_app(br))
            resp = client.post("/benchmark/run")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["model_loaded"] is False

    def test_run_benchmark_model_loaded(self):
        br = BenchmarkRouter()
        mock_ctrl = MagicMock()
        mock_ctrl._hf_model = MagicMock()
        mock_ctrl._last_inference_time = 100.0
        mock_ctrl._total_tokens_generated = 500
        mock_ctrl._inference_count = 10
        with patch("controllers.models.get_models_controller", return_value=mock_ctrl):
            client = TestClient(_app(br))
            resp = client.post("/benchmark/run?model=gpt2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert data["inference_count"] == 10


class TestGetQualityMetrics:
    def test_quality_metrics(self):
        br = BenchmarkRouter()
        mock_bench = MagicMock()
        mock_bench.evaluate_latest.return_value = {"coherence_score": 0.8, "repetition_rate": 0.1}
        with patch("domains.get_benchmark_domain", return_value=mock_bench):
            client = TestClient(_app(br))
            resp = client.get("/benchmark/quality")
        assert resp.status_code == 200
        assert resp.json()["data"]["coherence_score"] == 0.8


class TestGetLoggedResponses:
    def test_logged_responses(self):
        br = BenchmarkRouter()
        mock_tracker = MagicMock()
        r1 = SimpleNamespace(timestamp=1.0, user_message="hi", assistant_response="hello", model="gpt2", tokens_generated=5, duration_ms=100)
        mock_tracker.get_responses.return_value = [r1]
        with patch("domains.feedback.response_tracker.get_response_tracker", return_value=mock_tracker):
            client = TestClient(_app(br))
            resp = client.get("/benchmark/responses")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 1


class TestTrackerStats:
    def test_stats(self):
        br = BenchmarkRouter()
        mock_bench = MagicMock()
        mock_bench.get_stats.return_value = {"total_responses": 42}
        with patch("domains.get_benchmark_domain", return_value=mock_bench):
            client = TestClient(_app(br))
            resp = client.get("/benchmark/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_responses"] == 42


class TestClearHistory:
    def test_clear(self):
        br = BenchmarkRouter()
        mock_bench = MagicMock()
        with patch("domains.get_benchmark_domain", return_value=mock_bench):
            client = TestClient(_app(br))
            resp = client.post("/benchmark/history/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
        mock_bench.clear_history.assert_called_once()
