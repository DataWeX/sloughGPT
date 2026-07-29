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
