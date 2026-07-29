"""
Tests for the experiments router — CRUD and metric/param logging for ML experiments.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.experiments import router


@pytest.fixture
def app(tmp_path):
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestCreateExperiment:
    def test_creates_experiment(self, client):
        resp = client.post("/experiments", json={"name": "test_run"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "id" in data
        assert data["created"] is True


class TestListExperiments:
    def test_lists_experiments(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"]["experiments"], list)
