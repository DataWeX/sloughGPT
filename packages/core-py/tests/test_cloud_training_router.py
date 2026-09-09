"""Tests for cloud training router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add API server to path
api_server = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "api" / "server"
if str(api_server) not in sys.path:
    sys.path.insert(0, str(api_server))

from routers.cloud_training import CloudTrainingRouter


@pytest.fixture
def client():
    app = FastAPI()
    router = CloudTrainingRouter()
    app.include_router(router.router)
    return TestClient(app)


class TestCloudTrainingRouter:
    def test_list_jobs(self, client):
        response = client.get("/cloud-training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "jobs" in data["data"]

    def test_training_status(self, client):
        response = client.get("/cloud-training/local_001/status")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["status"] == "completed"
