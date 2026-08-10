"""Tests for the datasets API router (routers/datasets.py).

Covers: list, get, create, delete, validate_id, stats.
Controller is mocked; only HTTP-level behavior is tested.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.datasets import DatasetsRouter  # noqa: E402


def _mock_ctrl(**overrides) -> MagicMock:
    ctrl = MagicMock()
    ctrl.list_datasets.return_value = [
        {"id": "ds1", "name": "Test Dataset", "path": "/datasets/ds1", "type": "text", "size_bytes": 100, "size_formatted": "100 B", "num_samples": 10},
    ]
    ctrl.get_dataset.return_value = {"id": "ds1", "name": "Test Dataset", "path": "/datasets/ds1", "type": "text", "size_bytes": 100, "size_formatted": "100 B", "num_samples": 10}
    ctrl.create_dataset.return_value = {"id": "new-ds", "name": "New", "path": "/datasets/new-ds", "type": "text", "size_bytes": 0, "size_formatted": "Empty", "num_samples": 0}
    ctrl.delete_dataset.return_value = True
    ctrl.get_dataset_stats.return_value = {"samples": 10, "chars": 500, "avg_length": 50.0, "lines": 10, "format": "text"}
    return ctrl


def _app(dr: DatasetsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(dr.router)
    return app


class TestListDatasets:
    @patch("routers.datasets.get_datasets_controller")
    def test_list(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.get("/datasets")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    @patch("routers.datasets.get_datasets_controller")
    def test_list_with_query(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.get("/datasets?q=test")
        assert resp.status_code == 200


class TestGetDataset:
    @patch("routers.datasets.get_datasets_controller")
    def test_get_found(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.get("/datasets/ds1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "ds1"

    @patch("routers.datasets.get_datasets_controller")
    def test_get_not_found(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.get_dataset.return_value = None
        mock_get.return_value = ctrl
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.get("/datasets/nonexistent")
        assert resp.status_code == 404


class TestCreateDataset:
    @patch("routers.datasets.get_datasets_controller")
    def test_create(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.post("/datasets", json={"name": "New", "description": ""})
        assert resp.status_code == 200
        assert resp.json()["id"] == "new-ds"


class TestDeleteDataset:
    @patch("routers.datasets.get_datasets_controller")
    def test_delete(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.delete("/datasets/ds1")
        assert resp.status_code == 200

    @patch("routers.datasets.get_datasets_controller")
    def test_delete_not_found(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.delete_dataset.return_value = False
        mock_get.return_value = ctrl
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.delete("/datasets/nonexistent")
        assert resp.status_code == 404


class TestGetStats:
    @patch("routers.datasets.get_datasets_controller")
    def test_stats(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        dr = DatasetsRouter()
        client = TestClient(_app(dr))
        resp = client.get("/datasets/ds1/stats")
        assert resp.status_code == 200
        assert resp.json()["samples"] == 10


class TestValidateId:
    def test_invalid_id(self):
        dr = DatasetsRouter()
        with pytest.raises(Exception):
            dr._validate_dataset_id("../../../etc")

    def test_valid_id(self):
        dr = DatasetsRouter()
        assert dr._validate_dataset_id("my-dataset_1") == "my-dataset_1"
