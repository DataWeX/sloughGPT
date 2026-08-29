"""
Tests for the health router — health, liveness, readiness, startup-progress, debug, model, summary.
"""

import sys
import asyncio
import json
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock, PropertyMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.health import router

SERVER_DIR = str(Path(__file__).resolve().parents[2] / "apps/api/server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import state as _server_state


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
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

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_health_model_not_loaded(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_basic_health.return_value = {
            "status": "healthy", "model_loaded": False, "model_type": None,
        }
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_loaded"] is False

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_health_controller_error_returns_500(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_basic_health.side_effect = RuntimeError("health controller down")
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health")
        assert resp.status_code == 500


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

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_detailed_has_system_info(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/detailed")
        body = resp.json()["data"]
        assert "system" in body
        assert body["system"]["cpu_percent"] == 45.0

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_detailed_has_training_pool(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/detailed")
        body = resp.json()["data"]
        assert body["training_pool"]["active"] == 1


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

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_debug_has_latency_info(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/debug")
        body = resp.json()["data"]
        assert "avg_latency_ms" in body
        assert body["avg_latency_ms"] == 250

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_debug_has_token_info(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/debug")
        body = resp.json()["data"]
        assert body["total_tokens"] == 5000
        assert body["tokens_per_sec"] == 2.5


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
        assert resp.status_code == 500

    @patch("domains.feedback.model_health.get_health_monitor")
    def test_model_health_ok_with_stats(self, mock_get_mon, client):
        mon = MagicMock()
        mon._model = None
        mon.get_stats.return_value = {"inference_count": 42, "latency_ms": 5}
        mock_get_mon.return_value = mon
        import state as _state
        _state.model = None
        _state.tokenizer = None
        resp = client.get("/health/model")
        body = resp.json()["data"]
        assert body["status"] == "ok"
        assert body["inference_count"] == 42
        assert body["latency_ms"] == 5

    @patch("domains.feedback.model_health.get_health_monitor")
    def test_model_health_registers_state_model(self, mock_get_mon, client):
        mon = MagicMock()
        mon._model = None
        mon.get_stats.return_value = {"inference_count": 0}
        mock_get_mon.return_value = mon
        import state as _state
        _state.model = "gpt2"
        _state.tokenizer = "tok"
        try:
            client.get("/health/model")
        finally:
            _state.model = None
            _state.tokenizer = None
        mon.set_model.assert_called_once_with("gpt2", "tok")


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

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_summary_has_model_type(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/summary")
        body = resp.json()["data"]
        assert body["model_type"] == "gpt2"
        assert body["soul"] == "sage"

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_summary_has_uptime(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/summary")
        body = resp.json()["data"]
        assert body["uptime_seconds"] == 3600

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_summary_has_diagnoses(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed(
            health_score={"score": 60, "status": "degraded", "summary": "High CPU", "diagnoses": ["CPU at 95%"]}
        )
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/summary")
        body = resp.json()["data"]
        assert body["diagnoses"] == ["CPU at 95%"]

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_summary_no_model(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed(
            model_loaded=False, model_type=None, soul=None
        )
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health/summary")
        body = resp.json()["data"]
        assert body["model_loaded"] is False
        assert body["model_type"] is None


class TestHealthStream:
    """GET /health/stream — SSE snapshots."""

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_stream_yields_sse_snapshot(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.return_value = _make_detailed()
        ctrl.get_detailed_health.return_value["is_inferencing"] = True
        ctrl.get_detailed_health.return_value["inference_count"] = 7
        mock_get_ctrl.return_value = ctrl
        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])), \
             patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with client.stream("GET", "/health/stream") as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                body = resp.read()
                lines = [ln for ln in body.decode().split("\r\n") if ln]
                first = next((ln for ln in lines if ln.startswith("data: ")), None)
                assert first is not None
                snapshot = json.loads(first[6:])
                assert snapshot["stream"] == "health"
                assert snapshot["phase"] == "HEALTH"
                assert snapshot["status"] == "working"
                assert snapshot["data"]["is_inferencing"] is True
                assert snapshot["data"]["inference_count"] == 7
                assert snapshot["data"]["model_loaded"] is True

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_stream_survives_build_exception(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_detailed_health.side_effect = RuntimeError("boom")
        mock_get_ctrl.return_value = ctrl
        with patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])), \
             patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            with client.stream("GET", "/health/stream") as resp:
                assert resp.status_code == 200
                body = resp.read()
                assert b"data:" not in body


class TestHealthMethodCoverage:
    """Method-mismatch coverage — all health routes are GET-only."""

    def test_root_wrong_method_405(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405

    def test_live_wrong_method_405(self, client):
        resp = client.post("/health/live")
        assert resp.status_code == 405

    def test_ready_wrong_method_405(self, client):
        resp = client.post("/health/ready")
        assert resp.status_code == 405

    def test_detailed_wrong_method_405(self, client):
        resp = client.put("/health/detailed")
        assert resp.status_code == 405

    def test_startup_progress_wrong_method_405(self, client):
        resp = client.post("/health/startup-progress")
        assert resp.status_code == 405

    def test_debug_wrong_method_405(self, client):
        resp = client.post("/health/debug")
        assert resp.status_code == 405

    def test_summary_wrong_method_405(self, client):
        resp = client.post("/health/summary")
        assert resp.status_code == 405

    def test_model_wrong_method_405(self, client):
        resp = client.post("/health/model")
        assert resp.status_code == 405

    def test_stream_wrong_method_405(self, client):
        resp = client.delete("/health/stream")
        assert resp.status_code == 405



