"""
Tests for the registry router — GET /registry/models, /registry/models/{id}, /registry/best, /registry/stats.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.registry import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListModels:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_model_list(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = [{"model_id": "gpt2"}, {"model_id": "qwen"}]
        resp = client.get("/registry/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 2
        assert data["models"][0]["model_id"] == "gpt2"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_empty_registry(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 0
        assert data["models"] == []

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_single_model(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = [{"model_id": "only-one"}]
        resp = client.get("/registry/models")
        assert resp.json()["data"]["count"] == 1

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_success_status(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models")
        assert resp.json()["status"] == "success"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_many_models(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        models = [{"model_id": f"model-{i}"} for i in range(50)]
        reg.list_models.return_value = models
        resp = client.get("/registry/models")
        assert resp.json()["data"]["count"] == 50

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/registry/models")
        assert resp.status_code == 405

    def test_registry_error_returns_500(self, client):
        with patch("domains.infrastructure.model_registry.get_model_registry", side_effect=RuntimeError("broken")):
            resp = client.get("/registry/models")
        assert resp.status_code == 500

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_models_data_keys(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models")
        assert set(resp.json()["data"].keys()) == {"models", "count"}


class TestGetModel:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_finds_model(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = [{"model_id": "gpt2"}]
        resp = client.get("/registry/models/gpt2")
        assert resp.status_code == 200

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_404_for_missing(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models/nonexistent")
        assert resp.status_code == 404

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_404_detail_message(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models/missing-model")
        assert "not found" in resp.json()["detail"].lower()

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_correct_model(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = [
            {"model_id": "gpt2"},
            {"model_id": "qwen"},
        ]
        resp = client.get("/registry/models/qwen")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_id"] == "qwen"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_id_with_special_chars(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = []
        resp = client.get("/registry/models/some%20model")
        assert resp.status_code == 404

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_preserves_extra_fields(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.list_models.return_value = [
            {"model_id": "gpt2", "params": "124M", "source": "hf", "loaded": True}
        ]
        resp = client.get("/registry/models/gpt2")
        data = resp.json()["data"]
        assert data["model_id"] == "gpt2"
        assert data["params"] == "124M"
        assert data["loaded"] is True

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/registry/models/gpt2")
        assert resp.status_code == 405

    def test_model_lookup_error_returns_500(self, client):
        with patch("domains.infrastructure.model_registry.get_model_registry", side_effect=RuntimeError("broken")):
            resp = client.get("/registry/models/gpt2")
        assert resp.status_code == 500


class TestBestModel:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_health(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"status": "healthy"}
        resp = client.get("/registry/best")
        assert resp.json()["data"]["status"] == "healthy"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_success_status(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"status": "ok"}
        resp = client.get("/registry/best")
        assert resp.json()["status"] == "success"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_empty_health(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {}
        resp = client.get("/registry/best")
        assert resp.status_code == 200

    def test_wrong_method_returns_405(self, client):
        resp = client.post("/registry/best")
        assert resp.status_code == 405

    def test_best_error_returns_500(self, client):
        with patch("domains.infrastructure.model_registry.get_model_registry", side_effect=RuntimeError("broken")):
            resp = client.get("/registry/best")
        assert resp.status_code == 500


class TestRegistryStats:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_stats(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"models": [], "status": "ok"}
        resp = client.get("/registry/stats")
        assert resp.json()["data"]["status"] == "ok"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_success_status(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"status": "ok"}
        resp = client.get("/registry/stats")
        assert resp.json()["status"] == "success"

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_stats_with_models(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {
            "total_models": 3,
            "healthy": 2,
            "degraded": 1,
        }
        resp = client.get("/registry/stats")
        data = resp.json()["data"]
        assert data["total_models"] == 3

    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_best_and_stats_use_same_endpoint(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"status": "ok"}
        resp_best = client.get("/registry/best")
        resp_stats = client.get("/registry/stats")
        assert resp_best.json()["data"] == resp_stats.json()["data"]

    def test_wrong_method_returns_405(self, client):
        resp = client.put("/registry/stats")
        assert resp.status_code == 405

    def test_stats_error_returns_500(self, client):
        with patch("domains.infrastructure.model_registry.get_model_registry", side_effect=RuntimeError("broken")):
            resp = client.get("/registry/stats")
        assert resp.status_code == 500
