"""Tests for experiments router — filesystem CRUD, metric/param logging, validation."""

import json
import sys
import pytest
import shutil
from pathlib import Path

pytest.importorskip("fastapi")

# Ensure apps/api/server is on the path for schemas.common import
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.experiments import ExperimentsRouter, ExperimentCreate


@pytest.fixture
def exp_dir(tmp_path):
    """Provide a temporary experiments directory."""
    d = tmp_path / "experiments"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def app(exp_dir):
    """Create FastAPI app with experiments router pointing at tmp dir."""
    router_instance = ExperimentsRouter()
    router_instance.EXPERIMENTS_DIR = exp_dir
    app = FastAPI()
    app.include_router(router_instance.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestCreateExperiment:
    def test_create(self, client):
        resp = client.post("/experiments", json={"name": "test_exp"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["created"] is True
        assert "test_exp" in data["id"]

    def test_create_with_config(self, client):
        resp = client.post("/experiments", json={"name": "exp", "config": {"lr": 0.01}})
        assert resp.status_code == 200
        assert resp.json()["data"]["created"] is True


class TestListExperiments:
    def test_empty(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert resp.json()["data"]["experiments"] == []
        assert resp.json()["data"]["count"] == 0

    def test_with_experiments(self, client, exp_dir):
        (exp_dir / "exp1").mkdir()
        (exp_dir / "exp2").mkdir()
        resp = client.get("/experiments")
        assert resp.json()["data"]["count"] == 2
        assert set(resp.json()["data"]["experiments"]) == {"exp1", "exp2"}


class TestGetExperiment:
    def test_found(self, client, exp_dir):
        (exp_dir / "my_exp").mkdir()
        resp = client.get("/experiments/my_exp")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == "my_exp"

    def test_not_found(self, client):
        resp = client.get("/experiments/nonexistent")
        assert resp.status_code == 404

    def test_invalid_id(self, client):
        resp = client.get("/experiments/invalid id with spaces")
        assert resp.status_code in (400, 404)


class TestDeleteExperiment:
    def test_delete(self, client, exp_dir):
        (exp_dir / "to_delete").mkdir()
        resp = client.delete("/experiments/to_delete")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True
        assert not (exp_dir / "to_delete").exists()

    def test_delete_not_found(self, client):
        resp = client.delete("/experiments/nonexistent")
        assert resp.status_code == 404

    def test_delete_invalid_id(self, client):
        resp = client.delete("/experiments/invalid id with spaces")
        assert resp.status_code in (400, 404)


class TestLogMetric:
    def test_log_metric(self, client, exp_dir):
        resp = client.post("/experiments/exp1/log_metric?metric_name=loss&value=0.5&step=1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "logged"
        assert data["metric"] == "loss"

    def test_log_metric_invalid_id(self, client):
        resp = client.post("/experiments/invalid id with spaces/log_metric?metric_name=x&value=1.0")
        assert resp.status_code in (400, 404)


class TestLogParam:
    def test_log_param(self, client, exp_dir):
        resp = client.post("/experiments/exp1/log_param?param_name=lr&value=0.001")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "logged"


class TestCompleteExperiment:
    def test_complete(self, client, exp_dir):
        resp = client.post("/experiments/exp1/complete")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "completed"


class TestGetExperimentData:
    def test_empty_data(self, client, exp_dir):
        (exp_dir / "exp1").mkdir()
        resp = client.get("/experiments/exp1/data")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "metrics" in data
        assert "params" in data


class TestGetExperimentRuns:
    def test_runs_count(self, client, exp_dir):
        (exp_dir / "exp1").mkdir()
        resp = client.get("/experiments/exp1/runs")
        assert resp.status_code == 200
        assert resp.json()["data"]["runs"] == 0

    def test_runs_not_found(self, client):
        resp = client.get("/experiments/nonexistent/runs")
        assert resp.status_code == 404
