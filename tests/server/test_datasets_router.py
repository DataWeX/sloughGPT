"""
Tests for the datasets router — list, create, get, update, delete, import, export, versions.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.datasets import DatasetsRouter


@pytest.fixture
def router(tmp_path):
    r = DatasetsRouter()
    r._DATASETS_DIR = tmp_path / "datasets"
    r._DATASETS_DIR.mkdir(exist_ok=True)
    return r


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListDatasets:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_returns_list(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_datasets.return_value = []
        resp = client.get("/datasets")
        assert resp.status_code == 200


_DATASET_FIXTURE = {"id": "ds1", "name": "test", "path": "/datasets/ds1"}


class TestCreateDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_creates_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.post("/datasets", json={"name": "test"})
        assert resp.status_code == 200


class TestGetDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_gets_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.get_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.get("/datasets/valid_id")
        assert resp.status_code == 200

    def test_returns_404_for_bad_id(self, client):
        resp = client.get("/datasets/bad_id")
        assert resp.status_code == 404


class TestSearchDatasets:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_searches(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.search_datasets.return_value = []
        resp = client.get("/datasets/search?q=test")
        assert resp.status_code == 200


class TestDatasetVersions:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_create_version(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_version_snapshot.return_value = "20260801120000"
        resp = client.post("/datasets/ds1/versions")
        assert resp.status_code == 200
        assert resp.json()["timestamp"] == "20260801120000"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_create_version_404_when_dataset_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_version_snapshot.return_value = None
        resp = client.post("/datasets/ds1/versions")
        assert resp.status_code == 404

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_list_versions(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_versions.return_value = ["20260801120000", "20260801110000"]
        resp = client.get("/datasets/ds1/versions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["versions"][0] == "20260801120000"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_restore_version(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.restore_version.return_value = True
        resp = client.post("/datasets/ds1/versions/20260801120000")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_restore_version_404_when_version_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.restore_version.return_value = False
        resp = client.post("/datasets/ds1/versions/20260801120000")
        assert resp.status_code == 404
