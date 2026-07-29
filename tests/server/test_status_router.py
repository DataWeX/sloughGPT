"""
Tests for the status router — /status, /ready, /live.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.status import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetStatus:
    """GET /status"""

    def test_returns_healthy_with_uptime(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["status"] == "healthy"
        assert "uptime_seconds" in body["data"]
        assert isinstance(body["data"]["uptime_seconds"], (int, float))
        assert "timestamp" in body["data"]

    def test_uptime_increases(self, client):
        import time
        a = client.get("/status").json()["data"]["uptime_seconds"]
        time.sleep(0.01)
        b = client.get("/status").json()["data"]["uptime_seconds"]
        assert b > a


class TestReadiness:
    """GET /ready"""

    def test_returns_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["ready"] is True


class TestLiveness:
    """GET /live"""

    def test_returns_alive(self, client):
        resp = client.get("/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["alive"] is True
