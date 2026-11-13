"""Tests for the Dashboard router — SSE stream + REST endpoints."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app_with_router(auth_user_dict=_AUTH_USER):
    """Build the app AND keep hold of its `DashboardRouter`.

    The instance is otherwise unreachable: `include_router` nests the routes
    under a `_IncludedRouter` rather than flattening them into `app.routes`, so
    `app.routes` has no `/dashboard/stream` entry to recover a bound method
    from. Tests that need to tune the instance want it directly.
    """
    from routers.dashboard import DashboardRouter

    router_obj = DashboardRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app, router_obj


def _build_app(auth_user_dict=_AUTH_USER):
    return _build_app_with_router(auth_user_dict)[0]


def _data(resp):
    body = resp.json()
    return body.get("data", body)


# ── Events endpoint ──────────────────────────────────────────────────────────


class TestDashboardEvents:
    def test_events_returns_list(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events")
        assert resp.status_code == 200
        data = _data(resp)
        assert "events" in data
        assert "count" in data
        assert isinstance(data["events"], list)

    def test_events_with_limit(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events?n=5")
        assert resp.status_code == 200
        data = _data(resp)
        assert len(data["events"]) <= 5

    def test_events_default_limit(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events")
        data = _data(resp)
        assert len(data["events"]) <= 20

    def test_events_empty_when_no_activity(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events?n=0")
        assert resp.status_code == 200


# ── Summary endpoint ─────────────────────────────────────────────────────────


class TestDashboardSummary:
    def test_summary_structure(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        assert resp.status_code == 200
        data = _data(resp)
        assert "health" in data
        assert "active_processes" in data
        assert "processes" in data
        assert "services" in data

    def test_summary_health_fields(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        data = _data(resp)
        health = data["health"]
        assert "model_loaded" in health
        assert isinstance(health["model_loaded"], bool)

    def test_summary_services_counts(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        data = _data(resp)
        services = data["services"]
        assert "total" in services
        assert "healthy" in services
        assert services["total"] >= 0
        assert services["healthy"] <= services["total"]

    def test_summary_processes_is_dict(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        data = _data(resp)
        assert isinstance(data["processes"], dict)

    def test_summary_active_processes_is_int(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        data = _data(resp)
        assert isinstance(data["active_processes"], int)


# ── SSE stream endpoint ──────────────────────────────────────────────────────


class TestDashboardStream:
    def test_stream_method_get(self):
        _app = _build_app()
        client = TestClient(_app)
        resp = client.options("/dashboard/stream")
        allow = resp.headers.get("allow", "")
        assert "GET" in allow or resp.status_code in (200, 405)


# ── Auth enforcement ─────────────────────────────────────────────────────────


class TestDashboardAuth:
    def test_summary_requires_auth(self):
        _app = _build_app(auth_user_dict=None)
        _app.dependency_overrides[require_auth_if_enabled] = lambda: None
        resp = TestClient(_app).get("/dashboard/summary")
        assert resp.status_code in (200, 401, 403)

    def test_events_requires_auth(self):
        _app = _build_app(auth_user_dict=None)
        _app.dependency_overrides[require_auth_if_enabled] = lambda: None
        resp = TestClient(_app).get("/dashboard/events")
        assert resp.status_code in (200, 401, 403)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestDashboardEdgeCases:
    def test_summary_with_mocked_health(self):
        _app = _build_app()
        with patch(
            "routers.dashboard._get_health_summary",
            return_value={
                "model_loaded": True,
                "model_type": "test-model",
                "uptime_seconds": 100,
                "cpu_percent": 50.0,
                "memory_percent": 75.0,
            },
        ):
            resp = TestClient(_app).get("/dashboard/summary")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["health"]["model_loaded"] is True
            assert data["health"]["model_type"] == "test-model"

    def test_events_with_populated_buffer(self):
        _app = _build_app()
        mock_buffer = MagicMock()
        mock_buffer.recent.return_value = [
            {"event": "test_event", "ts": 1.0},
            {"event": "another_event", "ts": 2.0},
        ]
        with patch(
            "domain.infrastructure.get_event_buffer",
            return_value=mock_buffer,
        ):
            resp = TestClient(_app).get("/dashboard/events?n=10")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["count"] == 2
            assert len(data["events"]) == 2


# ── Stall visibility on the stream ───────────────────────────────────────────


class TestDashboardStreamStallVisibility:
    """`/dashboard/stream` must say so out loud when the snapshot build wedges.

    `_build_snapshot` is synchronous and unbounded. Awaited inline (the
    pre-`fix/health-stream-stall-visibility` shape) a blocked build silenced
    this stream with no signal whatsoever — and on a dashboard silence reads as
    "idle", never as "broken".

    Ported from `tests/server/test_health_router.py::TestHealthStreamStallVisibility`,
    which carries the full derivation of the polled-task pattern. These use real
    timing rather than `patch("asyncio.sleep")` because the timing *is* the
    behaviour under test.
    """

    @pytest.fixture
    def fast_stream(self):
        """Shrink the cadence and stall bound so this runs in milliseconds."""
        _app, inst = _build_app_with_router()
        saved = (inst.SNAPSHOT_INTERVAL, inst.DASHBOARD_SNAPSHOT_STALL_S)
        inst.SNAPSHOT_INTERVAL = 0.02
        inst.DASHBOARD_SNAPSHOT_STALL_S = 0.1
        yield _app, inst
        inst.SNAPSHOT_INTERVAL, inst.DASHBOARD_SNAPSHOT_STALL_S = saved

    @staticmethod
    def _frames(_app, cycles):
        """Collect `cycles` SSE frames from /dashboard/stream."""
        with patch(
            "fastapi.Request.is_disconnected",
            new=AsyncMock(side_effect=[False] * cycles + [True]),
        ):
            with TestClient(_app, raise_server_exceptions=False).stream(
                "GET", "/dashboard/stream"
            ) as resp:
                assert resp.status_code == 200
                body = resp.read().decode()
        return [
            json.loads(ln[6:])
            for ln in body.replace("\r\n", "\n").split("\n")
            if ln.startswith("data: ")
        ]

    def test_snapshot_merely_late_is_not_reported_as_stall(self, fast_stream):
        """A build slower than the cadence but inside the bound is just late."""
        _app, _inst = fast_stream

        def slow_but_fine():
            time.sleep(0.05)  # > interval (0.02), < stall bound (0.1)
            return {"stream": "dashboard", "phase": "DASHBOARD", "status": "working", "data": {}}

        with patch("routers.dashboard._build_snapshot", side_effect=slow_but_fine):
            frames = self._frames(_app, 4)

        assert frames, "stream produced nothing"
        assert all(f["phase"] == "DASHBOARD" for f in frames), (
            f"a late-but-bounded snapshot was misreported: {[f['phase'] for f in frames]}"
        )

    def test_wedged_build_reports_stall_and_spawns_exactly_one_build(self, fast_stream):
        """The anti-leak pin: many stall frames, one build.

        `wait_for(asyncio.to_thread(...))` abandons the future, not the thread,
        so retrying it every cycle spawns a fresh pool thread per cycle
        (measured 2->3->4->5 over four cycles on health). Shielding one task
        keeps it at exactly one — asserted by call count, which is what actually
        matters, rather than by counting threads.
        """
        _app, _inst = fast_stream
        calls = {"n": 0}

        def wedged():
            calls["n"] += 1
            time.sleep(0.6)  # far past the 0.1 stall bound
            return {"stream": "dashboard", "phase": "DASHBOARD", "status": "working", "data": {}}

        with patch("routers.dashboard._build_snapshot", side_effect=wedged):
            frames = self._frames(_app, 5)

        assert frames, "stream produced nothing"
        assert all(f["phase"] == "STALL" for f in frames), (
            f"expected stalls while wedged: {[f['phase'] for f in frames]}"
        )
        assert calls["n"] == 1, (
            f"a wedged build spawned {calls['n']} builds across 5 cycles; "
            "exactly one must be in flight"
        )
        first = frames[0]
        assert first["status"] == "working", "a stall is not an error yet"
        assert first["data"]["stalled"] is True
        assert first["data"]["waited_seconds"] == pytest.approx(0.1)
        assert "blocked" in first["message"]

    def test_stream_recovers_once_the_wedged_build_returns(self, fast_stream):
        """Stall, then a snapshot — the stream must not die on a wedge.

        The `task.done()` step in the generator is what makes this pass: a
        build can finish during a cycle whose wait timed out, and without
        consuming that result the next cycle would discard it and start again,
        stalling forever while re-spawning a thread.
        """
        _app, _inst = fast_stream
        state = {"first": True}

        def wedges_once():
            if state["first"]:
                state["first"] = False
                time.sleep(0.15)  # one cycle's worth past the bound
            return {"stream": "dashboard", "phase": "DASHBOARD", "status": "working", "data": {}}

        with patch("routers.dashboard._build_snapshot", side_effect=wedges_once):
            frames = self._frames(_app, 6)

        phases = [f["phase"] for f in frames]
        assert "STALL" in phases, f"expected a stall first: {phases}"
        assert "DASHBOARD" in phases, f"stream never recovered after the wedge: {phases}"
        assert phases.index("STALL") < phases.index("DASHBOARD")
        assert "ERROR" not in phases, "a stall must not surface as an error"
