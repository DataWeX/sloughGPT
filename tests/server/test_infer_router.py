"""
Tests for the unified inference router — /infer prefix endpoints.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.infer import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestInfer:
    @patch("state.model")
    @patch("domains.models.provider.get_provider")
    def test_generates_text(self, mock_get_prov, mock_model, client):
        provider = AsyncMock()
        provider.chat.return_value = "Hello world"
        mock_get_prov.return_value = provider
        mock_model.model_id = "test-model"
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello world"

    @patch("state.model", None)
    def test_returns_503_when_no_model(self, client):
        resp = client.post("/infer", json={"prompt": "Hi"})
        assert resp.status_code == 503


class TestInferTokenize:
    def test_tokenizes_text(self, client):
        resp = client.post("/infer/tokenize", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["tokens"]) == data["count"]


class TestInferDetokenize:
    def test_detokenizes_ids(self, client):
        resp = client.post("/infer/detokenize", json={"ids": [104, 101, 108, 108, 111]})
        assert resp.status_code == 200
        assert resp.json()["text"] == "hello"


class TestInferHealth:
    def test_returns_no_model_status(self, client):
        resp = client.get("/infer/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


class TestInferInfo:
    def test_returns_503_when_no_model(self, client):
        resp = client.get("/infer/info")
        assert resp.status_code == 503
