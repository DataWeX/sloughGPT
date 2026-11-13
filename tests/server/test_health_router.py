"""
Tests for the health router — health, liveness, readiness, startup-progress, debug, model, summary.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.health import router

SERVER_DIR = str(Path(__file__).resolve().parents[2] / "apps/api/server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)


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
            "status": "healthy",
            "model_loaded": True,
            "model_type": "gpt2",
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
            "status": "healthy",
            "model_loaded": False,
            "model_type": None,
        }
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_loaded"] is False

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_health_controller_error_returns_fallback_200(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_basic_health.side_effect = RuntimeError("health controller down")
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/health")
        assert resp.status_code == 200


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

    @patch("domain.feedback.get_health_monitor")
    def test_model_health_no_model(self, mock_get_mon, client):
        mon = MagicMock()
        mon._model = None
        mon.get_stats.return_value = {"inference_count": 0}
        mock_get_mon.return_value = mon
        resp = client.get("/health/model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domain.feedback.get_health_monitor")
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

    @patch("domain.feedback.get_health_monitor")
    def test_model_health_error(self, mock_get_mon, client):
        mock_get_mon.side_effect = RuntimeError("monitor down")
        resp = client.get("/health/model")
        assert resp.status_code == 500

    @patch("domain.feedback.get_health_monitor")
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

    @patch("domain.feedback.get_health_monitor")
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
            health_score={
                "score": 60,
                "status": "degraded",
                "summary": "High CPU",
                "diagnoses": ["CPU at 95%"],
            }
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
        with (
            patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])),
            patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        ):
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
        with (
            patch("fastapi.Request.is_disconnected", new=AsyncMock(side_effect=[False, True])),
            patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        ):
            with client.stream("GET", "/health/stream") as resp:
                assert resp.status_code == 200
                body = resp.read()
                assert b'"phase": "ERROR"' in body
                assert b'"error": "boom"' in body


class TestHealthStreamStallVisibility:
    """`/health/stream` must say so out loud when a subsystem check wedges.

    `_build_health_snapshot` -> `get_detailed_health()` is synchronous and
    unbounded. Awaited inline (the pre-`fix/health-stream-stall-visibility`
    shape) a blocked check stalled the stream with no signal at all — and this
    is the one stream an observer most needs to tell apart from a dead process.

    These tests use real timing rather than the `patch("asyncio.sleep")` trick
    above, because the timing *is* the behaviour under test.
    """

    @pytest.fixture
    def fast_stream(self):
        """Shrink the cadence and stall bound so this runs in milliseconds.

        The instance is anonymous — `router = HealthRouter().router` throws it
        away — but the bound method still sits on the route, so `__self__`
        recovers it.
        """
        route = next(r for r in router.routes if getattr(r, "name", "") == "health_stream")
        inst = route.endpoint.__self__
        saved = (inst.HEALTH_STREAM_INTERVAL, inst.HEALTH_SNAPSHOT_STALL_S)
        inst.HEALTH_STREAM_INTERVAL = 0.02
        inst.HEALTH_SNAPSHOT_STALL_S = 0.1
        yield inst
        inst.HEALTH_STREAM_INTERVAL, inst.HEALTH_SNAPSHOT_STALL_S = saved

    @staticmethod
    def _frames(client, cycles):
        """Collect `cycles` SSE frames from /health/stream."""
        with patch(
            "fastapi.Request.is_disconnected",
            new=AsyncMock(side_effect=[False] * cycles + [True]),
        ):
            with client.stream("GET", "/health/stream") as resp:
                assert resp.status_code == 200
                body = resp.read().decode()
        return [
            json.loads(ln[6:])
            for ln in body.replace("\r\n", "\n").split("\n")
            if ln.startswith("data: ")
        ]

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_snapshot_merely_late_is_not_reported_as_stall(
        self, mock_get_ctrl, client, fast_stream
    ):
        """A build slower than the cadence but inside the bound is just late."""
        ctrl = MagicMock()

        def slow_but_fine():
            time.sleep(0.05)  # > interval (0.02), < stall bound (0.1)
            return {"stream": "health", "phase": "HEALTH", "status": "working", "data": {}}

        ctrl.get_detailed_health.side_effect = slow_but_fine
        mock_get_ctrl.return_value = ctrl

        frames = self._frames(client, 4)
        assert frames, "stream produced nothing"
        assert all(f["phase"] == "HEALTH" for f in frames), (
            f"a late-but-bounded snapshot was misreported: {[f['phase'] for f in frames]}"
        )

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_wedged_check_reports_stall_and_spawns_exactly_one_build(
        self, mock_get_ctrl, client, fast_stream
    ):
        """The anti-leak pin: many stall frames, one build.

        `wait_for(asyncio.to_thread(...))` abandons the future, not the thread,
        so retrying it every cycle spawns a fresh pool thread per cycle
        (measured 2->3->4->5 over four cycles). Shielding one task keeps it at
        exactly one — asserted here by call count, which is what actually
        matters, rather than by counting threads.
        """
        ctrl = MagicMock()
        calls = {"n": 0}

        def wedged():
            calls["n"] += 1
            time.sleep(0.6)  # far past the 0.1 stall bound
            return {"stream": "health", "phase": "HEALTH", "status": "working", "data": {}}

        ctrl.get_detailed_health.side_effect = wedged
        mock_get_ctrl.return_value = ctrl

        frames = self._frames(client, 5)
        assert frames, "stream produced nothing"
        assert all(f["phase"] == "STALL" for f in frames), (
            f"expected stalls while wedged: {[f['phase'] for f in frames]}"
        )
        assert calls["n"] == 1, (
            f"a wedged check spawned {calls['n']} builds across 5 cycles; "
            "exactly one must be in flight"
        )
        # The stall frame carries enough for an observer to act on.
        first = frames[0]
        assert first["status"] == "working", "a stall is not an error yet"
        assert first["data"]["stalled"] is True
        assert first["data"]["waited_seconds"] == pytest.approx(0.1)
        assert "blocked" in first["message"]

    @patch("apps.api.server.routers.health.get_health_controller")
    def test_stream_recovers_once_the_wedged_check_returns(
        self, mock_get_ctrl, client, fast_stream
    ):
        """Stall, then a snapshot — the stream must not die on a wedge.

        The `task.done()` step in the generator is what makes this pass: a
        build can finish during a cycle whose wait timed out, and without
        consuming that result the next cycle would discard it and start again,
        stalling forever.
        """
        ctrl = MagicMock()
        state = {"first": True}

        def wedges_once():
            if state["first"]:
                state["first"] = False
                time.sleep(0.15)  # one cycle's worth past the bound
            return {"stream": "health", "phase": "HEALTH", "status": "working", "data": {}}

        ctrl.get_detailed_health.side_effect = wedges_once
        mock_get_ctrl.return_value = ctrl

        frames = self._frames(client, 6)
        phases = [f["phase"] for f in frames]
        assert "STALL" in phases, f"expected a stall first: {phases}"
        assert "HEALTH" in phases, f"stream never recovered after the wedge: {phases}"
        assert phases.index("STALL") < phases.index("HEALTH")
        assert "ERROR" not in phases, "a stall must not surface as an error"


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
