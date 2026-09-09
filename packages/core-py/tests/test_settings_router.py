"""Tests for the settings API router."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.settings import SettingsRouter


def _app(sr: SettingsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(sr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


class TestGetSettings:
    def test_get_all(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "generation" in data
        assert "training" in data
        assert "voice" in data
        assert "ui" in data


class TestGenerationSettings:
    def test_get_generation(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/generation")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "temperature" in data
        assert "top_p" in data

    def test_update_generation(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/generation", json={"temperature": 0.5})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["temperature"] == 0.5

    def test_update_generation_no_changes(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/generation", json={})
        assert resp.status_code == 200
        assert "No changes" in resp.json()["data"]["message"]


class TestTrainingSettings:
    def test_get_training(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "preferred_model" in data

    def test_update_training(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/training", json={"preferred_model": "gpt2"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["preferred_model"] == "gpt2"


class TestVoiceSettings:
    def test_get_voice(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/voice")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "noise_gate_db" in data

    def test_update_voice(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/voice", json={"noise_gate_db": -35.0})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["noise_gate_db"] == -35.0


class TestAdaptiveSettings:
    def test_get_adaptive(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/adaptive")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "enabled" in data

    def test_update_adaptive(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/adaptive", json={"exploration_rate": 0.3})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["exploration_rate"] == 0.3


class TestResetSettings:
    def test_reset(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        # First change something
        client.patch("/settings/generation", json={"temperature": 0.1})
        # Then reset
        resp = client.post("/settings/reset")
        assert resp.status_code == 200
        assert "reset" in resp.json()["data"]["status"]
        # Verify defaults restored
        resp = client.get("/settings/generation")
        assert resp.json()["data"]["temperature"] == 0.8


class TestValidation:
    def test_invalid_temperature(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/generation", json={"temperature": 5.0})
        assert resp.status_code == 422

    def test_invalid_noise_gate(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.patch("/settings/voice", json={"noise_gate_db": 10.0})
        assert resp.status_code == 422
