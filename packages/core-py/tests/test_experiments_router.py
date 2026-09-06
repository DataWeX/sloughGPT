"""Tests for experiments router — DB CRUD, metric/param logging, validation."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

pytest.importorskip("fastapi")

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.experiments import ExperimentsRouter


@pytest.fixture(autouse=True)
def mock_db():
    with patch("apps.api.server.routers.experiments._get_db") as mock_get:
        stores = {}

        class FakeDeleteResult:
            def __init__(self, n):
                self.deleted_count = n
            def __bool__(self):
                return self.deleted_count > 0

        def make_col(name):
            store = stores.setdefault(name, [])
            col = MagicMock()
            col.find.return_value = store
            col.count.side_effect = lambda q=None: len(
                [d for d in store if not q or all(d.get(k) == v for k, v in q.items())]
            )
            col.find_one.side_effect = lambda q: next(
                (d for d in store if d.get("experiment_id") == q.get("experiment_id")), None
            )
            col.insert_one.side_effect = lambda doc: store.append(doc)
            col.delete_many.side_effect = lambda q: (
                FakeDeleteResult(
                    len([store.remove(d) for d in list(store) if d.get("experiment_id") == q.get("experiment_id")])
                ) if any(d.get("experiment_id") == q.get("experiment_id") for d in store)
                else FakeDeleteResult(0)
            )
            return col

        mock_db = MagicMock()
        mock_db.collection.side_effect = make_col
        mock_get.return_value = mock_db
        yield stores
        stores.clear()


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(ExperimentsRouter().router)
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

    def test_with_experiments(self, client, mock_db):
        mock_db.setdefault("experiments", []).append({"experiment_id": "exp1", "name": "exp1"})
        mock_db.setdefault("experiments", []).append({"experiment_id": "exp2", "name": "exp2"})
        resp = client.get("/experiments")
        assert resp.json()["data"]["count"] == 2
        assert set(resp.json()["data"]["experiments"]) == {"exp1", "exp2"}


class TestGetExperiment:
    def test_found(self, client, mock_db):
        mock_db.setdefault("experiments", []).append({"experiment_id": "my_exp", "name": "my_exp"})
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
    def test_delete(self, client, mock_db):
        mock_db.setdefault("experiments", []).append({"experiment_id": "to_delete", "name": "to_delete"})
        resp = client.delete("/experiments/to_delete")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    def test_delete_not_found(self, client):
        resp = client.delete("/experiments/nonexistent")
        assert resp.status_code == 404

    def test_delete_invalid_id(self, client):
        resp = client.delete("/experiments/invalid id with spaces")
        assert resp.status_code in (400, 404)


class TestLogMetric:
    def test_log_metric(self, client):
        resp = client.post("/experiments/exp1/log_metric?metric_name=loss&value=0.5&step=1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "logged"
        assert data["metric"] == "loss"

    def test_log_metric_invalid_id(self, client):
        resp = client.post("/experiments/invalid id with spaces/log_metric?metric_name=x&value=1.0")
        assert resp.status_code in (400, 404)


class TestLogParam:
    def test_log_param(self, client):
        resp = client.post("/experiments/exp1/log_param?param_name=lr&value=0.001")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "logged"


class TestCompleteExperiment:
    def test_complete(self, client):
        resp = client.post("/experiments/exp1/complete")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "completed"


class TestGetExperimentData:
    def test_empty_data(self, client):
        resp = client.get("/experiments/exp1/data")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "metrics" in data
        assert "params" in data


class TestGetExperimentRuns:
    def test_runs_count(self, client, mock_db):
        mock_db.setdefault("experiments", []).append({"experiment_id": "exp1", "name": "exp1"})
        resp = client.get("/experiments/exp1/runs")
        assert resp.status_code == 200
        assert resp.json()["data"]["runs"] == 0

    def test_runs_not_found(self, client):
        resp = client.get("/experiments/nonexistent/runs")
        assert resp.status_code == 404
