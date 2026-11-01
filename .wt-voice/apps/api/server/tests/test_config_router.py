from infrastructure.exception_handlers import register_app_error_handler

"""
Tests for config router — GET/PUT/PATCH generation config.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.config import router as config_router

app = FastAPI()
register_app_error_handler(app)
app.include_router(config_router)
client = TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


SAMPLE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
}


@pytest.fixture(autouse=True)
def mock_ctrl():
    ctrl = MagicMock()
    ctrl.get_generation_config.return_value = dict(SAMPLE_CONFIG)
    ctrl.update_generation_config.return_value = dict(SAMPLE_CONFIG)
    with patch("routers.config.get_config_controller", return_value=ctrl):
        yield ctrl


class TestGetConfig:
    def test_get_generation_config(self, mock_ctrl):
        resp = client.get("/config/generation")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 256

    def test_get_all_fields(self, mock_ctrl):
        resp = client.get("/config/generation")
        data = _data(resp)
        assert "temperature" in data
        assert "max_tokens" in data
        assert "top_p" in data


class TestUpdateConfig:
    def test_put_updates_config(self, mock_ctrl):
        resp = client.put("/config/generation", json={"temperature": 0.9})
        assert resp.status_code == 200

    def test_patch_updates_config(self, mock_ctrl):
        resp = client.patch("/config/generation", json={"max_tokens": 512})
        assert resp.status_code == 200

    def test_put_passes_updates_to_controller(self, mock_ctrl):
        client.put("/config/generation", json={"temperature": 0.9, "max_new_tokens": 512})
        mock_ctrl.update_generation_config.assert_called_once_with(
            temperature=0.9,
            max_new_tokens=512,
        )

    def test_patch_passes_updates_to_controller(self, mock_ctrl):
        client.patch("/config/generation", json={"top_p": 0.95})
        mock_ctrl.update_generation_config.assert_called_once_with(top_p=0.95)

    def test_update_all_fields(self, mock_ctrl):
        payload = {
            "temperature": 0.5,
            "max_new_tokens": 128,
            "top_p": 0.8,
            "top_k": 50,
            "repetition_penalty": 1.2,
        }
        client.put("/config/generation", json=payload)
        mock_ctrl.update_generation_config.assert_called_once_with(**payload)
