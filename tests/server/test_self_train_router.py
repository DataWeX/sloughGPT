"""
Tests for the self-train router — POST start/stop and GET status.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.self_train import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestSelfTrainStart:
    @patch("state._self_train_proc", None)
    def test_starts_when_not_running(self, client):
        resp = client.post("/self-train/start", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] in ("started", "error")

    def test_rejects_invalid_model_name(self, client):
        resp = client.post("/self-train/start", json={"model": "bad model!"})
        assert resp.status_code == 422


class TestSelfTrainStop:
    def test_returns_not_running(self, client):
        resp = client.post("/self-train/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "not_running"


class TestSelfTrainStatus:
    def test_returns_not_started(self, client):
        resp = client.get("/self-train/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "not_started"
