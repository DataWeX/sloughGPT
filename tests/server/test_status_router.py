"""
Tests for the status router — /status, /ready, /live.
"""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.status import StatusRouter


@pytest.fixture
def status_router():
    return StatusRouter()


@pytest.fixture
def app(status_router):
    _app = FastAPI()
    _app.include_router(status_router.router)
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
        a = client.get("/status").json()["data"]["uptime_seconds"]
        time.sleep(0.01)
        b = client.get("/status").json()["data"]["uptime_seconds"]
        assert b > a

    def test_uptime_always_positive(self, client):
        resp = client.get("/status")
        assert resp.json()["data"]["uptime_seconds"] >= 0

    def test_timestamp_is_iso_format(self, client):
        resp = client.get("/status")
        ts = resp.json()["data"]["timestamp"]
        assert "T" in ts

    def test_status_success_field(self, client):
        resp = client.get("/status")
        assert resp.json()["status"] == "success"

    def test_uptime_is_numeric(self, client):
        resp = client.get("/status")
        uptime = resp.json()["data"]["uptime_seconds"]
        assert isinstance(uptime, (int, float))

    def test_concurrent_status_checks(self, client):
        for _ in range(5):
            resp = client.get("/status")
            assert resp.json()["data"]["status"] == "healthy"


class TestReadiness:
    """GET /ready"""

    def test_returns_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["ready"] is True

    def test_ready_is_always_true(self, client):
        resp = client.get("/ready")
        assert resp.json()["data"]["ready"] is True

    def test_ready_multiple_times(self, client):
        for _ in range(3):
            assert client.get("/ready").json()["data"]["ready"] is True


class TestLiveness:
    """GET /live"""

    def test_returns_alive(self, client):
        resp = client.get("/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["alive"] is True

    def test_alive_is_always_true(self, client):
        resp = client.get("/live")
        assert resp.json()["data"]["alive"] is True

    def test_concurrent_liveness_checks(self, client):
        for _ in range(5):
            resp = client.get("/live")
            assert resp.json()["data"]["alive"] is True

    def test_liveness_success_field(self, client):
        resp = client.get("/live")
        assert resp.json()["status"] == "success"

    def test_ready_wrong_method_is_405(self, client):
        resp = client.post("/ready")
        assert resp.status_code == 405

    def test_live_wrong_method_is_405(self, client):
        resp = client.post("/live")
        assert resp.status_code == 405

    def test_status_wrong_method_is_405(self, client):
        resp = client.post("/status")
        assert resp.status_code == 405

    def test_ready_data_shape(self, client):
        body = client.get("/ready").json()
        assert set(body["data"].keys()) >= {"ready"}
        assert body["data"]["ready"] is True

    def test_live_data_shape(self, client):
        body = client.get("/live").json()
        assert set(body["data"]) == {"alive"}
        assert body["data"]["alive"] is True

    def test_status_has_status_and_uptime_keys(self, client):
        data = client.get("/status").json()["data"]
        assert set(data.keys()) == {"status", "uptime_seconds", "timestamp"}
        assert data["status"] == "healthy"


class TestStatusMethods:
    """405s for disallowed methods on status endpoints"""

    def test_status_put_rejected(self, client):
        assert client.put("/status").status_code == 405

    def test_status_delete_rejected(self, client):
        assert client.delete("/status").status_code == 405

    def test_ready_put_rejected(self, client):
        assert client.put("/ready").status_code == 405

    def test_ready_delete_rejected(self, client):
        assert client.delete("/ready").status_code == 405

    def test_live_put_rejected(self, client):
        assert client.put("/live").status_code == 405

    def test_live_delete_rejected(self, client):
        assert client.delete("/live").status_code == 405

    def test_status_patch_rejected(self, client):
        assert client.patch("/status").status_code == 405


class TestTimestampFormat:
    """GET /status timestamp is ISO-8601 parseable"""

    def test_timestamp_parseable(self, client):
        from datetime import datetime
        ts = client.get("/status").json()["data"]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2026

    def test_uptime_changes_over_time(self, client):
        first = client.get("/status").json()["data"]["uptime_seconds"]
        time.sleep(0.02)
        second = client.get("/status").json()["data"]["uptime_seconds"]
        assert second >= first
