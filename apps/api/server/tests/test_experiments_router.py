"""Tests for the /experiments router — CRUD, metric/param logging, compare."""

import pytest
from test_support import _data, get_test_client


class TestCreateExperiment:
    def setup_method(self):
        self.client = get_test_client()

    def test_create_experiment(self):
        resp = self.client.post("/experiments", json={"name": "test_exp"})
        assert resp.status_code == 200
        data = _data(resp)
        assert data["name"] == "test_exp"
        assert data["created"] is True
        assert "id" in data

    def test_create_experiment_with_config(self):
        resp = self.client.post("/experiments", json={"name": "config_exp", "config": {"lr": 0.001}})
        assert resp.status_code == 200
        data = _data(resp)
        assert data["name"] == "config_exp"

    def test_create_experiment_missing_name(self):
        resp = self.client.post("/experiments", json={})
        assert resp.status_code == 422

    def test_create_experiment_invalid_name(self):
        resp = self.client.post("/experiments", json={"name": "bad name!"})
        assert resp.status_code == 422


class TestListExperiments:
    def setup_method(self):
        self.client = get_test_client()

    def test_list_experiments(self):
        resp = self.client.get("/experiments")
        assert resp.status_code == 200
        data = _data(resp)
        assert "experiments" in data
        assert "count" in data
        assert isinstance(data["experiments"], list)

    def test_list_after_create(self):
        self.client.post("/experiments", json={"name": "list_test"})
        resp = self.client.get("/experiments")
        assert resp.status_code == 200
        assert _data(resp)["count"] >= 1


class TestGetExperiment:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_experiment_invalid_id(self):
        resp = self.client.get("/experiments/invalid..id")
        assert resp.status_code == 400

    def test_get_experiment_not_found(self):
        resp = self.client.get("/experiments/nonexistent_12345")
        assert resp.status_code == 404


class TestDeleteExperiment:
    def setup_method(self):
        self.client = get_test_client()

    def test_delete_experiment_invalid_id(self):
        resp = self.client.delete("/experiments/invalid..id")
        assert resp.status_code == 400

    def test_delete_experiment_not_found(self):
        resp = self.client.delete("/experiments/nonexistent_12345")
        assert resp.status_code == 404


class TestExperimentRuns:
    def setup_method(self):
        self.client = get_test_client()

    def test_runs_invalid_id(self):
        resp = self.client.get("/experiments/invalid..id/runs")
        assert resp.status_code == 400

    def test_runs_not_found(self):
        resp = self.client.get("/experiments/nonexistent_12345/runs")
        assert resp.status_code == 404


class TestExperimentData:
    def setup_method(self):
        self.client = get_test_client()

    def test_data_invalid_id(self):
        resp = self.client.get("/experiments/invalid..id/data")
        assert resp.status_code == 400

    def test_data_not_found(self):
        resp = self.client.get("/experiments/nonexistent_12345/data")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["metrics"] == []
        assert data["params"] == []


class TestLogMetric:
    def setup_method(self):
        self.client = get_test_client()

    def test_log_metric_invalid_id(self):
        resp = self.client.post("/experiments/invalid..id/log_metric?metric_name=loss&value=0.5")
        assert resp.status_code == 400

    def test_log_metric_missing_params(self):
        resp = self.client.post("/experiments/test/log_metric")
        assert resp.status_code == 422


class TestLogParam:
    def setup_method(self):
        self.client = get_test_client()

    def test_log_param_invalid_id(self):
        resp = self.client.post("/experiments/invalid..id/log_param?param_name=lr&value=0.001")
        assert resp.status_code == 400


class TestCompleteExperiment:
    def setup_method(self):
        self.client = get_test_client()

    def test_complete_invalid_id(self):
        resp = self.client.post("/experiments/invalid..id/complete")
        assert resp.status_code == 400


class TestCompareExperiments:
    def setup_method(self):
        self.client = get_test_client()

    def test_compare_requires_ids(self):
        resp = self.client.get("/experiments/compare")
        assert resp.status_code == 422

    def test_compare_single_id_fails(self):
        resp = self.client.get("/experiments/compare?ids=exp1")
        assert resp.status_code == 400

    def test_compare_invalid_id(self):
        resp = self.client.get("/experiments/compare?ids=bad..id,ok")
        assert resp.status_code == 400

    def test_compare_too_many_ids(self):
        ids = ",".join([f"exp{i}" for i in range(15)])
        resp = self.client.get(f"/experiments/compare?ids={ids}")
        assert resp.status_code == 400


class TestExperimentLifecycle:
    """End-to-end: create → log metrics/params → complete → data."""

    def setup_method(self):
        self.client = get_test_client()

    def test_full_lifecycle(self):
        # Create
        resp = self.client.post(
            "/experiments", json={"name": "lifecycle_test", "config": {"lr": 0.01}}
        )
        assert resp.status_code == 200
        exp_id = _data(resp)["id"]

        # Log metrics
        resp = self.client.post(
            f"/experiments/{exp_id}/log_metric", params={"metric_name": "loss", "value": 0.5}
        )
        assert resp.status_code == 200

        resp = self.client.post(
            f"/experiments/{exp_id}/log_metric", params={"metric_name": "accuracy", "value": 0.85}
        )
        assert resp.status_code == 200

        # Log params
        resp = self.client.post(
            f"/experiments/{exp_id}/log_param", params={"param_name": "lr", "value": 0.01}
        )
        assert resp.status_code == 200

        # Get data
        resp = self.client.get(f"/experiments/{exp_id}/data")
        assert resp.status_code == 200
        data = _data(resp)
        assert len(data["metrics"]) == 2
        assert len(data["params"]) == 1

        # Get runs
        resp = self.client.get(f"/experiments/{exp_id}/runs")
        assert resp.status_code == 200

        # Complete
        resp = self.client.post(f"/experiments/{exp_id}/complete")
        assert resp.status_code == 200
        assert _data(resp)["status"] == "completed"

        # Delete
        resp = self.client.delete(f"/experiments/{exp_id}")
        assert resp.status_code == 200
        assert _data(resp)["deleted"] is True
