"""Tests for the config API router (routers/config.py).

Covers: get_generation_config, update_generation_config.
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
from routers.config import ConfigRouter  # noqa: E402


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_generation_config.return_value = {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 40,
        "max_tokens": 128,
        "repetition_penalty": 1.15,
    }
    ctrl.update_generation_config.return_value = {"temperature": 0.9}
    return ctrl


def _app(cr: ConfigRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(cr.router)
    return app


class TestGetGenerationConfig:
    @patch("routers.config.get_config_controller")
    def test_get(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        cr = ConfigRouter()
        client = TestClient(_app(cr))
        resp = client.get("/config/generation")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["temperature"] == 0.7
        assert data["top_p"] == 0.85


class TestUpdateGenerationConfig:
    @patch("routers.config.get_config_controller")
    def test_update(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        cr = ConfigRouter()
        client = TestClient(_app(cr))
        resp = client.put("/config/generation", json={"temperature": 0.9})
        assert resp.status_code == 200

    @patch("routers.config.get_config_controller")
    def test_patch(self, mock_get):
        mock_get.return_value = _mock_ctrl()
        cr = ConfigRouter()
        client = TestClient(_app(cr))
        resp = client.patch("/config/generation", json={"top_p": 0.95})
        assert resp.status_code == 200
