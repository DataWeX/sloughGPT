"""
Tests for the config router — GET/PUT /config/generation.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.config import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


DEFAULT_CFG = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "max_new_tokens": 200,
    "max_context_length": 1024,
}


class TestGetGenerationConfig:
    """GET /config/generation"""

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_returns_generation_config(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/config/generation")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        for k, v in DEFAULT_CFG.items():
            assert body["data"][k] == v

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_default_values(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/config/generation")
        body = resp.json()["data"]
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.9
        assert body["max_new_tokens"] == 200


class TestUpdateGenerationConfig:
    """PUT /config/generation"""

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_updates_single_field(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "temperature": 0.5}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"temperature": 0.5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["temperature"] == 0.5

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_updates_multiple_fields(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        expected = {**DEFAULT_CFG, "temperature": 0.3, "top_k": 100}
        ctrl.update_generation_config.return_value = expected
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"temperature": 0.3, "top_k": 100})
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["temperature"] == 0.3
        assert body["top_k"] == 100

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_empty_update_returns_current(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={})
        assert resp.status_code == 200
        assert resp.json()["data"]["temperature"] == 0.8
