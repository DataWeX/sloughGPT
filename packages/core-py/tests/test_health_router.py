"""Tests for the health API router (routers/health.py).

Covers: health, liveness, readiness, detailed, debug, summary, model_health, startup_progress.
Health controller is mocked; only HTTP-level behavior is tested.
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

from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.health import HealthRouter  # noqa: E402
from tests.conftest import build_test_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_health_router() -> HealthRouter:
    return HealthRouter()


def _app(hr: HealthRouter):
    return build_test_app(hr.router)


def _mock_ctrl(**overrides) -> MagicMock:
    """Return a mock HealthController with sensible defaults."""
    ctrl = MagicMock()
    ctrl.get_basic_health.return_value = {
        "model_loaded": True,
        "model_type": "qwen",
        "soul": "assistant",
        "is_inferencing": False,
        "inference_count": 10,
        **overrides,
    }
    ctrl.get_liveness.return_value = {"alive": True}
    ctrl.get_readiness.return_value = {"ready": True}
    ctrl.get_detailed_health.return_value = {
        "model_loaded": True,
        "model_type": "qwen",
        "soul": "assistant",
        "uptime_seconds": 120,
        "request_count": 50,
        "error_count": 2,
        "inference_count": 10,
        "total_tokens": 500,
        "tokens_per_sec": 1.5,
        "avg_tokens_per_request": 5.0,
        "avg_latency_ms": 100.0,
        "requests_per_minute": 25.0,
        "health_score": {"score": 95, "status": "healthy", "summary": "All good"},
        "system": {"cpu_percent": 30.0, "memory_percent": 60.0},
        "model_metrics": [],
        "model_events": [],
        "health_history": [],
        "memory_history": [],
        "rate_violations": [],
        "path_latencies": [],
        "recent_errors": [],
        **overrides,
    }
    return ctrl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    @patch("routers.health.get_health_controller")
    def test_health_returns_basic(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["model_loaded"] is True

    @patch("routers.health.get_health_controller")
    def test_liveness(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["data"]["alive"] is True

    @patch("routers.health.get_health_controller")
    def test_readiness(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["data"]["ready"] is True

    @patch("routers.health.get_health_controller")
    def test_detailed_health(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["model_loaded"] is True
        assert "system" in data  # raw dict from controller includes system block

    @patch("routers.health.get_health_controller")
    def test_debug_info(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/debug")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "cpu_percent" in data
        assert "health_score" in data

    @patch("routers.health.get_health_controller")
    def test_health_summary(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/summary")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "score" in data
        assert "status" in data
        assert "diagnoses" in data

    def test_startup_progress(self):
        hr = _make_health_router()
        client = TestClient(_app(hr))
        resp = client.get("/health/startup-progress")
        assert resp.status_code == 200
        assert "phase" in resp.json()["data"]

    @patch("routers.health.get_health_controller")
    def test_model_health_ok(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        # model_health imports get_health_monitor lazily — mock it
        mock_mon = MagicMock()
        mock_mon.get_stats.return_value = {"accuracy": 0.9}
        with patch("routers.health.get_health_controller", mock_get_ctrl), \
             patch("domains.feedback.model_health.get_health_monitor", return_value=mock_mon):
            resp = client.get("/health/model")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ok"

    @patch("routers.health.get_health_controller")
    def test_model_health_error(self, mock_get_ctrl):
        mock_get_ctrl.return_value = _mock_ctrl()
        hr = _make_health_router()
        client = TestClient(_app(hr))
        with patch("domains.feedback.model_health.get_health_monitor", side_effect=RuntimeError("boom")):
            resp = client.get("/health/model")
        assert resp.status_code == 500
        assert resp.json()["error"] == "An unexpected error occurred."
