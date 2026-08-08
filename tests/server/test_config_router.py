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

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_exact_data_keys(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/config/generation")
        assert set(resp.json()["data"].keys()) == set(DEFAULT_CFG.keys())

    def test_get_error_returns_500(self, client):
        with patch("apps.api.server.routers.config.get_config_controller", side_effect=RuntimeError("broken")):
            resp = client.get("/config/generation")
        assert resp.status_code == 500


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

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_partial_update_only_sends_changed_fields(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        client.put("/config/generation", json={"top_p": 0.5})
        _, kwargs = ctrl.update_generation_config.call_args
        assert kwargs.get("top_p") == 0.5
        assert kwargs.get("temperature") is None

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_update_max_new_tokens(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "max_new_tokens": 512}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"max_new_tokens": 512})
        assert resp.json()["data"]["max_new_tokens"] == 512

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_update_repetition_penalty(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "repetition_penalty": 1.5}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"repetition_penalty": 1.5})
        assert resp.json()["data"]["repetition_penalty"] == 1.5

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_update_max_context_length(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "max_context_length": 4096}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"max_context_length": 4096})
        assert resp.json()["data"]["max_context_length"] == 4096

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_patch_method_also_works(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "temperature": 0.1}
        mock_get_ctrl.return_value = ctrl
        resp = client.patch("/config/generation", json={"temperature": 0.1})
        assert resp.status_code == 200
        assert resp.json()["data"]["temperature"] == 0.1

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_null_fields_ignored(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        client.put("/config/generation", json={"temperature": None, "top_p": None})
        _, kwargs = ctrl.update_generation_config.call_args
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_negative_temperature_accepted_by_update_schema(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "temperature": -1.0}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"temperature": -1.0})
        assert resp.status_code == 200

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_negative_top_k_accepted_by_update_schema(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = {**DEFAULT_CFG, "top_k": -5}
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"top_k": -5})
        assert resp.status_code == 200

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_updates_all_six_fields(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        updates = {
            "temperature": 0.5,
            "top_p": 0.7,
            "top_k": 100,
            "repetition_penalty": 1.3,
            "max_new_tokens": 512,
            "max_context_length": 2048,
        }
        expected = {**DEFAULT_CFG, **updates}
        ctrl.update_generation_config.return_value = expected
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json=updates)
        assert resp.status_code == 200
        body = resp.json()["data"]
        for k, v in updates.items():
            assert body[k] == v
        _, kwargs = ctrl.update_generation_config.call_args
        for k in updates:
            assert kwargs.get(k) == updates[k]

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_update_error_returns_500(self, mock_get_ctrl, client):
        mock_get_ctrl.side_effect = RuntimeError("broken")
        resp = client.put("/config/generation", json={"temperature": 0.5})
        assert resp.status_code == 500


class TestConfigValidation:
    """Protocol and validation gaps for /config/generation"""

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_get_rejects_post(self, mock_get_ctrl, client):
        resp = client.post("/config/generation")
        assert resp.status_code == 405

    def test_invalid_field_type_returns_422(self, client):
        resp = client.put("/config/generation", json={"temperature": "hot"})
        assert resp.status_code == 422

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_unknown_field_ignored(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.update_generation_config.return_value = dict(DEFAULT_CFG)
        mock_get_ctrl.return_value = ctrl
        resp = client.put("/config/generation", json={"nonexistent_field": 1, "top_k": 77})
        assert resp.status_code == 200
        _, kwargs = ctrl.update_generation_config.call_args
        assert kwargs.get("top_k") == 77
        assert "nonexistent_field" not in kwargs

    @patch("apps.api.server.routers.config.get_config_controller")
    def test_delete_rejected(self, mock_get_ctrl, client):
        resp = client.delete("/config/generation")
        assert resp.status_code == 405
