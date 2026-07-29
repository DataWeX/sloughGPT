"""
Tests for the voice router — POST /voice/tts and GET /voice/status.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.voice import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestTTS:
    def test_returns_fallback_when_model_unavailable(self, client):
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio"] == ""
        assert data["backend"] == "browser-fallback"

    def test_rejects_empty_text(self, client):
        resp = client.post("/voice/tts", json={"text": ""})
        assert resp.status_code == 400


class TestStatus:
    def test_returns_status(self, client):
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "server_tts" in data
