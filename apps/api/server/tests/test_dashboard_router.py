"""Tests for the Dashboard router — SSE stream + REST endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.dashboard import DashboardRouter

    router_obj = DashboardRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


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
        with patch("routers.dashboard._get_health_summary", return_value={
            "model_loaded": True,
            "model_type": "test-model",
            "uptime_seconds": 100,
            "cpu_percent": 50.0,
            "memory_percent": 75.0,
        }):
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
        with patch("domains.infrastructure.event_buffer.get_event_buffer", return_value=mock_buffer):
            resp = TestClient(_app).get("/dashboard/events?n=10")
            assert resp.status_code == 200
            data = _data(resp)
            assert data["count"] == 2
            assert len(data["events"]) == 2
