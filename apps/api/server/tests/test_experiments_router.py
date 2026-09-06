"""Tests for the /experiments router (CRUD + metric/param logging)."""

from test_support import get_test_client


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


class TestCreateExperiment:
    def test_create_experiment(self):
        client = get_test_client()
        resp = client.post("/experiments", json={"name": "test_exp"})
        assert resp.status_code == 200
        data = _data(resp)
        assert data["name"] == "test_exp"
        assert data["created"] is True
        assert "id" in data

    def test_create_experiment_with_config(self):
        client = get_test_client()
        resp = client.post("/experiments", json={"name": "config_exp", "config": {"lr": 0.001}})
        assert resp.status_code == 200
        data = _data(resp)
        assert data["name"] == "config_exp"

    def test_create_experiment_missing_name(self):
        client = get_test_client()
        resp = client.post("/experiments", json={})
        assert resp.status_code == 422


class TestListExperiments:
    def test_list_experiments(self):
        client = get_test_client()
        resp = client.get("/experiments")
        assert resp.status_code == 200
        data = _data(resp)
        assert "experiments" in data
        assert "count" in data
        assert isinstance(data["experiments"], list)


class TestGetExperiment:
    def test_get_experiment_invalid_id(self):
        client = get_test_client()
        resp = client.get("/experiments/invalid..id")
        assert resp.status_code == 400

    def test_get_experiment_not_found(self):
        client = get_test_client()
        resp = client.get("/experiments/nonexistent_12345")
        assert resp.status_code == 404


class TestDeleteExperiment:
    def test_delete_experiment_invalid_id(self):
        client = get_test_client()
        resp = client.delete("/experiments/invalid..id")
        assert resp.status_code == 400

    def test_delete_experiment_not_found(self):
        client = get_test_client()
        resp = client.delete("/experiments/nonexistent_12345")
        assert resp.status_code == 404


class TestExperimentRuns:
    def test_runs_invalid_id(self):
        client = get_test_client()
        resp = client.get("/experiments/invalid..id/runs")
        assert resp.status_code == 400

    def test_runs_not_found(self):
        client = get_test_client()
        resp = client.get("/experiments/nonexistent_12345/runs")
        assert resp.status_code == 404


class TestLogMetric:
    def test_log_metric_invalid_id(self):
        client = get_test_client()
        resp = client.post("/experiments/invalid..id/log_metric?metric_name=loss&value=0.5")
        assert resp.status_code == 400

    def test_log_metric_missing_params(self):
        client = get_test_client()
        resp = client.post("/experiments/test/log_metric")
        assert resp.status_code == 422


class TestLogParam:
    def test_log_param_invalid_id(self):
        client = get_test_client()
        resp = client.post("/experiments/invalid..id/log_param?param_name=lr&value=0.001")
        assert resp.status_code == 400


class TestCompleteExperiment:
    def test_complete_invalid_id(self):
        client = get_test_client()
        resp = client.post("/experiments/invalid..id/complete")
        assert resp.status_code == 400
