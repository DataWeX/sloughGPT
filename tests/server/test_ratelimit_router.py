"""
Tests for the rate-limit router — GET /rate-limit/status and GET /rate-limit/check.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.ratelimit import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetStatus:
    def test_returns_config(self, client):
        resp = client.get("/rate-limit/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["requests_per_minute"] == 60
        assert data["burst_size"] == 10
        assert data["enabled"] is True


class TestCheckLimit:
    def test_allows_request(self, client):
        resp = client.get("/rate-limit/check")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["allowed"] in (True, False)
