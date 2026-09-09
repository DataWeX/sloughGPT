"""Tests for OpenWebUI integration router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add API server to path
api_server = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "api" / "server"
if str(api_server) not in sys.path:
    sys.path.insert(0, str(api_server))

from routers.openwebui import OpenWebUIRouter


@pytest.fixture
def client():
    app = FastAPI()
    router = OpenWebUIRouter()
    app.include_router(router.router)
    return TestClient(app)


class TestOpenWebUIRouter:
    def test_list_datasets(self, client):
        response = client.get("/openwebui/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "datasets" in data["data"]

    def test_list_checkpoints(self, client):
        response = client.get("/openwebui/checkpoints")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "checkpoints" in data["data"]

    def test_training_status(self, client):
        response = client.get("/openwebui/training/status")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["status"] == "idle"
