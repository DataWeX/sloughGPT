"""
Tests for status router — /status, /ready, /live endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.status import router as status_router

app = FastAPI()
app.include_router(status_router)
client = TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestStatus:

    def test_status_healthy(self):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["status"] == "healthy"

    def test_status_has_uptime(self):
        resp = client.get("/status")
        data = _data(resp)
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_status_has_timestamp(self):
        resp = client.get("/status")
        data = _data(resp)
        assert "timestamp" in data


class TestReady:

    def test_ready_returns_true(self):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = _data(resp)
        assert data.get("ready") is True


class TestLive:

    def test_live_returns_true(self):
        resp = client.get("/live")
        assert resp.status_code == 200
        data = _data(resp)
        assert data.get("alive") is True
