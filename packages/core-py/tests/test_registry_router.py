"""Tests for the registry API router (routers/registry.py).

Covers: list_models, get_model (found/not-found), get_best_model, get_registry_stats.
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
from routers.registry import RegistryRouter  # noqa: E402


def _mock_registry(**overrides) -> MagicMock:
    reg = MagicMock()
    reg.list_models.return_value = [
        {"model_id": "gpt2", "status": "loaded"},
        {"model_id": "qwen", "status": "unloaded"},
    ]
    reg.health_summary.return_value = {
        "total_models": 2,
        "loaded_models": 1,
        "failed_models": 0,
        **overrides,
    }
    return reg


def _app(rr: RegistryRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(rr.router)
    return app


class TestListModels:
    @patch("routers.registry.RegistryRouter._get_registry")
    def test_list_models(self, mock_get):
        mock_get.return_value = _mock_registry()
        rr = RegistryRouter()
        client = TestClient(_app(rr))
        resp = client.get("/registry/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["count"] == 2
        assert len(data["models"]) == 2


class TestGetModel:
    @patch("routers.registry.RegistryRouter._get_registry")
    def test_model_found(self, mock_get):
        mock_get.return_value = _mock_registry()
        rr = RegistryRouter()
        client = TestClient(_app(rr))
        resp = client.get("/registry/models/gpt2")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_id"] == "gpt2"

    @patch("routers.registry.RegistryRouter._get_registry")
    def test_model_not_found(self, mock_get):
        mock_get.return_value = _mock_registry()
        rr = RegistryRouter()
        client = TestClient(_app(rr))
        resp = client.get("/registry/models/nonexistent")
        assert resp.status_code == 404


class TestBestAndStats:
    @patch("routers.registry.RegistryRouter._get_registry")
    def test_best_model(self, mock_get):
        mock_get.return_value = _mock_registry()
        rr = RegistryRouter()
        client = TestClient(_app(rr))
        resp = client.get("/registry/best")
        assert resp.status_code == 200
        assert "total_models" in resp.json()["data"]

    @patch("routers.registry.RegistryRouter._get_registry")
    def test_stats(self, mock_get):
        mock_get.return_value = _mock_registry()
        rr = RegistryRouter()
        client = TestClient(_app(rr))
        resp = client.get("/registry/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_models"] == 2
        assert data["loaded_models"] == 1
