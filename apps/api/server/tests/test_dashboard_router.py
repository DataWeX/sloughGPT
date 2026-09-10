"""Tests for the Dashboard router — SSE stream + REST events."""

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


# ── Events endpoint ──────────────────────────────────────────────────────────


class TestDashboardEvents:
    def test_events_returns_list(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    def test_events_with_limit(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/events?limit=5")
        assert resp.status_code == 200


# ── Summary endpoint ─────────────────────────────────────────────────────────


class TestDashboardSummary:
    def test_summary(self):
        _app = _build_app()
        resp = TestClient(_app).get("/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
