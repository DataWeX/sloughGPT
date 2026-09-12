"""Tests for status router — /status, /ready, /live endpoints."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_app_error_handler
from routers.status import router as status_router

app = FastAPI()
register_app_error_handler(app)
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

    def test_status_uptime_increases(self):
        resp1 = client.get("/status")
        resp2 = client.get("/status")
        t1 = _data(resp1)["uptime_seconds"]
        t2 = _data(resp2)["uptime_seconds"]
        assert t2 >= t1

    def test_status_timestamp_is_iso(self):
        resp = client.get("/status")
        ts = _data(resp)["timestamp"]
        assert "T" in ts

    def test_status_structure(self):
        resp = client.get("/status")
        body = resp.json()
        assert "data" in body
        assert "status" in body


class TestReady:
    def test_ready_returns_true(self):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = _data(resp)
        assert data.get("ready") is True

    def test_ready_has_checks(self):
        resp = client.get("/ready")
        data = _data(resp)
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_ready_checks_database(self):
        resp = client.get("/ready")
        data = _data(resp)
        checks = data.get("checks", {})
        assert "database" in checks
        assert isinstance(checks["database"], bool)

    def test_ready_checks_inference(self):
        resp = client.get("/ready")
        data = _data(resp)
        checks = data.get("checks", {})
        assert "inference" in checks
        assert isinstance(checks["inference"], bool)

    def test_ready_with_db_down(self):
        with patch("domains.feedback.database.get_feedback_db", side_effect=Exception("db down")):
            resp = client.get("/ready")
            data = _data(resp)
            assert data["checks"]["database"] is False

    def test_ready_with_inference_down(self):
        with patch("domains.inference.native.engine.get_engine", side_effect=Exception("engine missing")):
            resp = client.get("/ready")
            data = _data(resp)
            assert data["checks"]["inference"] is False


class TestLive:
    def test_live_returns_true(self):
        resp = client.get("/live")
        assert resp.status_code == 200
        data = _data(resp)
        assert data.get("alive") is True

    def test_live_structure(self):
        resp = client.get("/live")
        body = resp.json()
        assert "data" in body
        data = body["data"]
        assert "alive" in data


class TestStatusMethodMismatch:
    def test_status_post_405(self):
        resp = client.post("/status")
        assert resp.status_code == 405

    def test_ready_post_405(self):
        resp = client.post("/ready")
        assert resp.status_code == 405

    def test_live_post_405(self):
        resp = client.post("/live")
        assert resp.status_code == 405
