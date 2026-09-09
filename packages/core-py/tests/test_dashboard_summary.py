"""Tests for the dashboard summary endpoint."""

import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.dashboard import DashboardRouter


def _app(dr: DashboardRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(dr.router)
    return app


class TestDashboardSummary:
    def test_summary_returns_ok(self):
        dr = DashboardRouter()
        client = TestClient(_app(dr))
        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "health" in data
        assert "active_processes" in data
        assert "services" in data
        assert "total" in data["services"]
        assert "healthy" in data["services"]

    def test_summary_health_structure(self):
        dr = DashboardRouter()
        client = TestClient(_app(dr))
        resp = client.get("/dashboard/summary")
        data = resp.json()["data"]
        health = data["health"]
        assert "model_loaded" in health
        assert "uptime_seconds" in health
        assert "cpu_percent" in health
        assert "memory_percent" in health
