"""Tests for the health services endpoint."""

import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.health import HealthRouter


def _app(hr: HealthRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(hr.router)
    return app


class TestServicesHealth:
    def test_services_health_returns_ok(self):
        hr = HealthRouter()
        client = TestClient(_app(hr))
        resp = client.get("/health/services")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] in ("healthy", "degraded")
        assert "services" in data
        assert "training" in data["services"]
        assert "settings" in data["services"]
        assert "plugins" in data["services"]
        assert "adaptive" in data["services"]

    def test_services_all_ok(self):
        hr = HealthRouter()
        client = TestClient(_app(hr))
        resp = client.get("/health/services")
        data = resp.json()["data"]
        for svc in data["services"].values():
            assert svc["status"] == "ok"
