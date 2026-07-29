"""
Tests for the health router — health, liveness, readiness, startup-progress, debug, model, summary.
"""

import sys
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.health import router

SERVER_DIR = str(Path(__file__).resolve().parents[2] / "apps/api/server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import state as _server_state


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _make_detailed(**overrides):
    return {
        "model_loaded": True,
        "model_loading": False,
        "model_type": "gpt2",
        "soul": "sage",
        "uptime_seconds": 3600,
        "request_count": 100,
        "error_count": 2,
        "inference_count": 50,
        "total_tokens": 5000,
        "tokens_per_sec": 2.5,
        "avg_tokens_per_request": 50,
        "avg_latency_ms": 250,
        "requests_per_minute": 10,
        "num_parameters": 124000000,
        "quantization": None,
        "health_score": {"score": 85, "status": "healthy", "summary": "All good", "diagnoses": []},
        "model_metrics": [],
        "model_events": [],
        "health_history": [],
        "memory_history": [],
        "rate_violations": [],
        "path_latencies": [],
        "recent_errors": [],
        "training_pool": {"active": 1, "max": 2, "tracked": 3},
        "system": {"cpu_percent": 45.0, "memory_percent": 60.0},
        "gpu": {"backend": None},
        **overrides,
    }


class TestHealth:
    """GET /health"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_health_returns_basic(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_basic_health.return_value = {
            "status": "healthy", "model_loaded": True, "model_type": "gpt2",
        }
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["model_loaded"] is True


class TestLiveness:
    """GET /health/live"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_liveness_returns_ok(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_liveness.return_value = {"status": "alive"}
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "alive"


class TestReadiness:
    """GET /health/ready"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_readiness_returns_ok(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_readiness.return_value = {"ready": True}
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["data"]["ready"] is True


class TestDetailedHealth:
    """GET /health/detailed"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_detailed_returns_full(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["model_loaded"] is True
        assert body["model_type"] == "gpt2"
        assert body["uptime_seconds"] == 3600
        assert body["request_count"] == 100


class TestStartupProgress:
    """GET /health/startup-progress"""

    @patch("apps.api.server.routers.health.STARTUP_PHASE", {"phase": "ready", "progress": 100})
    def test_startup_progress_returns_phase(self, client):
        resp = client.get("/health/startup-progress")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["phase"] == "ready"
        assert body["progress"] == 100

    @patch("apps.api.server.routers.health.STARTUP_PHASE", {"phase": "loading", "progress": 50})
    def test_startup_progress_loading(self, client):
        resp = client.get("/health/startup-progress")
        body = resp.json()["data"]
        assert body["phase"] == "loading"
        assert body["progress"] == 50


class TestDebugInfo:
    """GET /health/debug"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_debug_returns_detailed_snapshot(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/debug")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["model_loaded"] is True
        assert body["cpu_percent"] == 45.0
        assert body["memory_percent"] == 60.0
        assert "health_score" in body


class TestModelHealth:
    """GET /health/model"""

    @patch("domains.feedback.model_health.get_health_monitor")
    def test_model_health_no_model(self, mock_get_mon, client):
        mon = MagicMock()
        mon._model = None
        mon.get_stats.return_value = {"inference_count": 0}
        mock_get_mon.return_value = mon
        resp = client.get("/health/model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.feedback.model_health.get_health_monitor")
    def test_model_health_with_model(self, mock_get_mon, client):
        import state as _state
        _state.model = "gpt2"
        _state.tokenizer = "tok"
        mon = MagicMock()
        mon._model = None
        mon.get_stats.return_value = {"inference_count": 42}
        mock_get_mon.return_value = mon
        try:
            resp = client.get("/health/model")
            assert resp.status_code == 200
        finally:
            _state.model = None
            _state.tokenizer = None

    @patch("domains.feedback.model_health.get_health_monitor")
    def test_model_health_error(self, mock_get_mon, client):
        mock_get_mon.side_effect = RuntimeError("monitor down")
        resp = client.get("/health/model")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "error"


class TestHealthSummary:
    """GET /health/summary"""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_summary_returns_condensed(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/summary")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["score"] == 85
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["cpu_percent"] == 45.0
