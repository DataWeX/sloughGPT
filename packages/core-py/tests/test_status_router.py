"""Tests for the status API router (routers/status.py).

Covers: get_status, ready, live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.status import StatusRouter  # noqa: E402


def _app(sr: StatusRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    return app


class TestStatus:
    def test_get_status(self):
        sr = StatusRouter()
        client = TestClient(_app(sr))
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_ready(self):
        sr = StatusRouter()
        client = TestClient(_app(sr))
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["data"]["ready"] is True

    def test_live(self):
        sr = StatusRouter()
        client = TestClient(_app(sr))
        resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["data"]["alive"] is True
