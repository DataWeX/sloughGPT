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


class TestBestModel:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_health(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"status": "healthy"}
        resp = client.get("/registry/best")
        assert resp.json()["data"]["status"] == "healthy"


class TestRegistryStats:
    @patch("domains.infrastructure.model_registry.get_model_registry")
    def test_returns_stats(self, mock_get_reg, client):
        reg = mock_get_reg.return_value
        reg.health_summary.return_value = {"models": [], "status": "ok"}
        resp = client.get("/registry/stats")
        assert resp.json()["data"]["status"] == "ok"
