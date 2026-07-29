"""
Tests for the models router — list, load, unload, HF models.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.models import ModelsRouter


@pytest.fixture
def router():
    r = ModelsRouter()
    r._hf_cache_dir = MagicMock()
    r._hf_cache_dir.exists.return_value = False
    return r


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListModels:
    """GET /models"""

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    def test_list_models_with_loaded(self, mock_cached, mock_size, mock_get_ctrl, client):
        mock_cached.return_value = False
        mock_size.return_value = 0.5
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = {
            "model_id": "gpt2", "device": "cpu",
            "parameters": 124000000, "vocab_size": 50257,
            "loaded_at": "2026-01-01T00:00:00",
        }
        ctrl.list_hf_models.return_value = []
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        models = body["data"]
        assert any(m["model_id"] == "gpt2" and m["status"] == "loaded" for m in models)

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_list_models_empty(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = None
        ctrl.list_hf_models.return_value = []
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestLoadModel:
    """POST /models/load"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_success(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {
            "status": "loaded", "model_id": "gpt2",
            "device": "cpu", "parameters": 124000000,
        }
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        ctrl.load_model.assert_called_once()

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_failure(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "error", "error": "model not found"}
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/load", json={"model_id": "nonexistent"})
        assert resp.status_code == 200
        ctrl.load_model.assert_called_once()


class TestUnloadModel:
    """POST /models/unload"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_unload_success(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.unload_model.return_value = {"status": "unloaded"}
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"


class TestListHFModels:
    """GET /models/hf"""

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    @patch("domains.infrastructure.resource_manager.get_resource_manager")
    def test_list_hf_models(self, mock_rm, mock_cached, mock_size, mock_get_ctrl, client):
        mock_rm.return_value = MagicMock(inference_pool_size=2)
        mock_cached.return_value = False
        mock_size.return_value = 0.5

        ctrl = MagicMock()
        ctrl.list_hf_models.return_value = [
            {"model_id": "gpt2", "parameters": 124000000},
            {"model_id": "gpt2-medium", "parameters": 355000000},
        ]
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models/hf")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2
        ids = [m["id"] for m in body["data"]]
        assert "gpt2" in ids
        assert "gpt2-medium" in ids

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    @patch("domains.infrastructure.resource_manager.get_resource_manager")
    def test_list_hf_models_search(self, mock_rm, mock_cached, mock_size, mock_get_ctrl, client):
        mock_rm.return_value = MagicMock(inference_pool_size=2)
        mock_cached.return_value = False
        mock_size.return_value = 0.5

        ctrl = MagicMock()
        ctrl.list_hf_models.return_value = [
            {"model_id": "gpt2-xl", "parameters": 1500000000},
        ]
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models/hf?q=gpt2")
        assert resp.status_code == 200
        ctrl.list_hf_models.assert_called_with("gpt2")
