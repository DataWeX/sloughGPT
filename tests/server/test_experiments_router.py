"""
Tests for the experiments router — CRUD, metric/param logging, path traversal.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch

from apps.api.server.routers.experiments import ExperimentsRouter


@pytest.fixture
def experiments_router(tmp_path):
    r = ExperimentsRouter()
    r.EXPERIMENTS_DIR = tmp_path / "experiments"
    return r


@pytest.fixture
def app(experiments_router):
    _app = FastAPI()
    _app.include_router(experiments_router.router)
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
        assert data["name"] == "test_run"

    def test_experiment_id_contains_name(self, client):
        resp = client.post("/experiments", json={"name": "my_experiment"})
        assert resp.json()["data"]["id"].startswith("my_experiment_")

    def test_experiment_dir_created(self, client, experiments_router):
        resp = client.post("/experiments", json={"name": "dir_test"})
        exp_id = resp.json()["data"]["id"]
        exp_dir = experiments_router.EXPERIMENTS_DIR / exp_id
        assert exp_dir.exists()

    def test_experiment_id_has_timestamp(self, client):
        resp = client.post("/experiments", json={"name": "ts_test"})
        exp_id = resp.json()["data"]["id"]
        import re
        assert re.search(r'\d{8}_\d{6}$', exp_id)


class TestListExperiments:
    def test_empty_list(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["experiments"] == []
        assert data["count"] == 0

    def test_lists_experiments(self, client):
        client.post("/experiments", json={"name": "exp_a"})
        client.post("/experiments", json={"name": "exp_b"})
        resp = client.get("/experiments")
        data = resp.json()["data"]
        assert data["count"] >= 2

    def test_returns_list_type(self, client):
        resp = client.get("/experiments")
        assert isinstance(resp.json()["data"]["experiments"], list)


class TestGetExperiment:
    def test_get_existing_experiment(self, client):
        create = client.post("/experiments", json={"name": "get_test"})
        exp_id = create.json()["data"]["id"]
        resp = client.get(f"/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == exp_id

    def test_get_nonexistent_experiment(self, client):
        resp = client.get("/experiments/nonexistent_12345")
        assert resp.status_code == 404

    def test_get_invalid_id_with_dots(self, client):
        resp = client.get("/experiments/name.with.dots")
        assert resp.status_code in (400, 404)

    def test_get_id_with_special_chars(self, client):
        resp = client.get("/experiments/name%20with%20spaces")
        assert resp.status_code == 400


class TestDeleteExperiment:
    def test_delete_existing(self, client):
        create = client.post("/experiments", json={"name": "del_test"})
        exp_id = create.json()["data"]["id"]
        resp = client.delete(f"/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    def test_delete_nonexistent(self, client):
        resp = client.delete("/experiments/nonexistent_xyz")
        assert resp.status_code == 404

    def test_delete_invalid_id(self, client):
        resp = client.delete("/experiments/invalid..id")
        assert resp.status_code == 400

    def test_delete_removes_from_list(self, client):
        create = client.post("/experiments", json={"name": "vanish"})
        exp_id = create.json()["data"]["id"]
        client.delete(f"/experiments/{exp_id}")
        resp = client.get("/experiments")
        assert exp_id not in resp.json()["data"]["experiments"]


class TestGetExperimentRuns:
    def test_empty_runs(self, client):
        create = client.post("/experiments", json={"name": "runs_test"})
        exp_id = create.json()["data"]["id"]
        resp = client.get(f"/experiments/{exp_id}/runs")
        assert resp.status_code == 200
        assert resp.json()["data"]["runs"] == 0

    def test_runs_with_json_files(self, client, tmp_path):
        from apps.api.server.routers.experiments import ExperimentsRouter
        r = ExperimentsRouter()
        r.EXPERIMENTS_DIR = tmp_path / "experiments2"
        _app = FastAPI()
        _app.include_router(r.router)
        c = TestClient(_app, raise_server_exceptions=False)

        create = c.post("/experiments", json={"name": "json_test"})
        exp_id = create.json()["data"]["id"]
        exp_dir = r.EXPERIMENTS_DIR / exp_id
        (exp_dir / "run_1.json").write_text('{"loss": 0.5}')
        (exp_dir / "run_2.json").write_text('{"loss": 0.3}')
        resp = c.get(f"/experiments/{exp_id}/runs")
        assert resp.json()["data"]["runs"] == 2

    def test_runs_nonexistent_experiment(self, client):
        resp = client.get("/experiments/no_such_exp/runs")
        assert resp.status_code == 404


class TestCompleteExperiment:
    def test_complete_returns_completed(self, client):
        resp = client.post("/experiments/some_exp/complete")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "completed"
        assert resp.json()["data"]["id"] == "some_exp"


class TestLogMetric:
    def test_logs_metric(self, client):
        resp = client.post(
            "/experiments/test_exp/log_metric",
            params={"metric_name": "loss", "value": 0.5, "step": 1}
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "logged"

    def test_metric_invalid_id(self, client):
        resp = client.post(
            "/experiments/invalid..id/log_metric",
            params={"metric_name": "x", "value": 1.0, "step": 0}
        )
        assert resp.status_code == 400

    def test_metric_default_step_zero(self, client):
        resp = client.post(
            "/experiments/exp1/log_metric",
            params={"metric_name": "f1", "value": 0.7}
        )
        assert resp.status_code == 200


class TestLogParam:
    def test_logs_param(self, client):
        resp = client.post(
            "/experiments/test_exp/log_param",
            params={"param_name": "learning_rate", "value": 0.001}
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "logged"

    def test_param_invalid_id(self, client):
        resp = client.post(
            "/experiments/invalid..id/log_param",
            params={"param_name": "x", "value": 1}
        )
        assert resp.status_code == 400

    def test_param_string_value(self, client):
        resp = client.post(
            "/experiments/exp2/log_param",
            params={"param_name": "model", "value": "gpt2"}
        )
        assert resp.status_code == 200
