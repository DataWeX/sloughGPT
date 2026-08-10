"""Tests for the models API router (routers/models.py).

Covers: get_export_formats, current_model, get_model_logs, get_catalog, get_catalog_stats.
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
from routers.models import ModelsRouter  # noqa: E402


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_current_model.return_value = None
    ctrl.list_hf_models.return_value = []
    ctrl.get_model_logs.return_value = []
    return ctrl


def _app(mr: ModelsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(mr.router)
    return app


class TestGetExportFormats:
    @patch("domains.training.export.list_export_formats")
    def test_formats(self, mock_list):
        mock_list.return_value = [{"format": "safetensors", "recommended": True}]
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/export/formats")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1


class TestCurrentModel:
    @patch("routers.models.get_models_controller")
    def test_no_model(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/current")
        assert resp.status_code == 404

    @patch("routers.models.get_models_controller")
    def test_with_model(self, mock_get):
        ctrl = _mock_ctrl()
        ctrl.get_current_model.return_value = {"model_id": "gpt2", "device": "cpu", "parameters": 124000000}
        mock_get.return_value = ctrl
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/current")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_id"] == "gpt2"


class TestGetModelLogs:
    @patch("routers.models.get_models_controller")
    def test_empty_logs(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/logs")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestGetCatalog:
    @patch("routers.models.get_models_controller")
    def test_catalog(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/catalog")
        assert resp.status_code == 200


class TestGetCatalogStats:
    @patch("routers.models.get_models_controller")
    def test_stats(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/catalog/stats")
        assert resp.status_code == 200


class TestGetConversionStatus:
    @patch("routers.models.get_models_controller")
    def test_status(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        mr = ModelsRouter()
        client = TestClient(_app(mr))
        resp = client.get("/models/conversion-status")
        assert resp.status_code == 200
