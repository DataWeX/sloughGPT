"""
Tests for the meta-weights router — POST /meta-weights/get and GET /meta-weights/stats.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.meta_weights import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetMetaWeights:
    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_adjustment(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_adjustment.return_value = type("W", (), {
            "temperature": 0.9, "repetition_penalty": 1.1,
            "top_p": 0.95, "top_k": 40,
        })()
        mgr._weight_history = [1, 2, 3]
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["temperature"] == 0.9
        assert resp.json()["based_on_samples"] == 3

    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_503_when_unavailable(self, mock_get_mgr, client):
        mock_get_mgr.return_value = None
        resp = client.post("/meta-weights/get", json={"user_message": "hello"})
        assert resp.status_code == 503


class TestGetStats:
    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_stats(self, mock_get_mgr, client):
        mgr = mock_get_mgr.return_value
        mgr.get_stats.return_value = {"total_adjustments": 10}
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 200
        assert resp.json()["data"]["total_adjustments"] == 10

    @patch("domains.feedback.get_meta_weight_manager")
    def test_returns_503_when_unavailable(self, mock_get_mgr, client):
        mock_get_mgr.return_value = None
        resp = client.get("/meta-weights/stats")
        assert resp.status_code == 503
