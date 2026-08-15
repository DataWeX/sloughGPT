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


class TestExperimentData:
    """GET /experiments/{id}/data"""

    def test_empty_experiment_data(self, client):
        resp = client.get("/experiments/zzz_12345/data")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["metrics"] == []
        assert data["params"] == []
        assert data["status"] is None

    def test_data_invalid_id_is_400(self, client):
        resp = client.get("/experiments/invalid..id/data")
        assert resp.status_code == 400

    def test_data_reads_logged_metrics_and_params(self, client, experiments_router):
        import os, json
        e_id = "readback_123"
        log_dir = experiments_router.EXPERIMENTS_DIR
        metrics_file = os.path.join(log_dir, f"{e_id}_metrics.jsonl")
        params_file = os.path.join(log_dir, f"{e_id}_params.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(metrics_file, "w") as f:
                f.write(json.dumps({"metric": "loss", "value": 0.5, "step": 1}) + "\n")
            with open(params_file, "w") as f:
                f.write(json.dumps({"param": "lr", "value": 0.001}) + "\n")
            resp = client.get(f"/experiments/{e_id}/data")
            data = resp.json()["data"]
            assert data["metrics"][0]["value"] == 0.5
            assert data["params"][0]["value"] == 0.001
        finally:
            for p in (metrics_file, params_file):
                if os.path.exists(p):
                    os.remove(p)

    def test_data_skips_corrupt_json_lines(self, client):
        import os
        from apps.api.server.routers import experiments as exp_mod
        e_id = "corrupt_123"
        log_dir = os.path.join(os.path.dirname(exp_mod.__file__), "..", "data", "experiments")
        metrics_file = os.path.join(log_dir, f"{e_id}_metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        try:
            with open(metrics_file, "w") as f:
                f.write("not valid json\n")
            resp = client.get(f"/experiments/{e_id}/data")
            assert resp.json()["data"]["metrics"] == []
        finally:
            if os.path.exists(metrics_file):
                os.remove(metrics_file)


class TestCompleteExperimentEdges:
    """POST /experiments/{id}/complete — persistence and validation"""

    def test_complete_invalid_id_is_400(self, client):
        resp = client.post("/experiments/invalid..id/complete")
        assert resp.status_code == 400

    def test_complete_persists_status_readable_by_data(self, client):
        import os
        from apps.api.server.routers import experiments as exp_mod
        e_id = "persist_123"
        log_dir = os.path.join(os.path.dirname(exp_mod.__file__), "..", "data", "experiments")
        status_file = os.path.join(log_dir, f"{e_id}_status.json")
        os.makedirs(log_dir, exist_ok=True)
        try:
            resp = client.post(f"/experiments/{e_id}/complete")
            assert resp.status_code == 200
            data = client.get(f"/experiments/{e_id}/data").json()["data"]
            assert data["status"]["status"] == "completed"
            assert data["status"]["experiment_id"] == e_id
        finally:
            if os.path.exists(status_file):
                os.remove(status_file)

    def test_complete_get_is_405(self, client):
        resp = client.get("/experiments/some_exp/complete")
        assert resp.status_code == 405


class TestExperimentsMethods:
    """405s for disallowed methods"""

    def test_runs_post_is_405(self, client):
        resp = client.post("/experiments/some_exp/runs")
        assert resp.status_code == 405

    def test_data_post_is_405(self, client):
        resp = client.post("/experiments/some_exp/data")
        assert resp.status_code == 405

    def test_log_metric_get_is_405(self, client):
        resp = client.get("/experiments/some_exp/log_metric")
        assert resp.status_code == 405

    def test_log_param_delete_is_405(self, client):
        resp = client.delete("/experiments/some_exp/log_param")
        assert resp.status_code == 405

    def test_put_experiment_is_405(self, client):
        resp = client.put("/experiments/some_exp")
        assert resp.status_code == 405


class TestExperimentsValidation:
    """422s and traversal protection"""

    def test_create_missing_name_is_422(self, client):
        resp = client.post("/experiments", json={})
        assert resp.status_code == 422

    def test_create_name_wrong_type_is_422(self, client):
        resp = client.post("/experiments", json={"name": 123})
        assert resp.status_code == 422

    def test_get_dotdot_is_400(self, client):
        resp = client.get("/experiments/foo..bar")
        assert resp.status_code == 400

    def test_get_experiment_with_config_ignored(self, client):
        resp = client.post("/experiments", json={"name": "cfg_run", "config": {"lr": 0.1}})
        assert resp.status_code == 200
        assert resp.json()["data"]["created"] is True

    def test_list_ignores_files_in_dir(self, client, experiments_router, tmp_path):
        client.post("/experiments", json={"name": "only_dir"})
        (experiments_router.EXPERIMENTS_DIR / "loose_file.txt").write_text("x")
        resp = client.get("/experiments")
        names = resp.json()["data"]["experiments"]
        assert "loose_file.txt" not in names

    def test_get_experiment_accepts_file_path(self, client, experiments_router):
        """get_experiment only checks existence — a loose file is accepted."""
        experiments_router.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        (experiments_router.EXPERIMENTS_DIR / "afile").write_text("x")
        resp = client.get("/experiments/afile")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == "afile"
