"""
Tests for the companion router — personality, presets, chat, prompt.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.companion import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

COMPANION_TARGET = "apps.api.server.routers.companion._companion_router._get_companion"


def _mock_companion():
    comp = MagicMock()
    comp.to_dict.return_value = {
        "name": "Friend",
        "traits": {
            "warmth": 0.7,
            "curiosity": 0.6,
            "creativity": 0.5,
            "confidence": 0.5,
            "humor": 0.4,
        },
    }
    comp.get_system_prompt.return_value = "You are a warm friend."
    return comp


class TestCompanionInfo:
    """GET /companion/"""

    @patch(COMPANION_TARGET)
    def test_get_info(self, mock_get):
        mock_get.return_value = _mock_companion()
        resp = client.get("/companion/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert "name" in data
        assert "traits" in data
        assert data["name"] == "Friend"


class TestSetPersonality:
    """POST /companion/personality"""

    @patch(COMPANION_TARGET)
    def test_set_personality(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/personality", json={
            "name": "Alice",
            "warmth": 0.9,
            "curiosity": 0.8,
            "creativity": 0.7,
            "confidence": 0.6,
            "humor": 0.5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["status"] == "ok"
        assert "traits" in data
        comp.set_personality.assert_called_once_with(
            name="Alice", warmth=0.9, curiosity=0.8,
            creativity=0.7, confidence=0.6, humor=0.5,
        )

    @patch(COMPANION_TARGET)
    def test_set_personality_defaults(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/personality", json={"name": "Bob"})
        assert resp.status_code == 200
        comp.set_personality.assert_called_once_with(
            name="Bob", warmth=0.7, curiosity=0.6,
            creativity=0.5, confidence=0.5, humor=0.4,
        )


class TestPreset:
    """POST /companion/preset"""

    @patch("domains.companion.create_companion")
    @patch(COMPANION_TARGET)
    def test_use_preset(self, mock_get, mock_create):
        new_comp = _mock_companion()
        mock_create.return_value = new_comp
        mock_get.return_value = _mock_companion()

        resp = client.post("/companion/preset", json={"name": "Buddy", "preset": "playful"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["status"] == "ok"
        assert data["preset"] == "playful"
        assert "traits" in data
        mock_create.assert_called_once_with(name="Buddy", personality="playful")


class TestPrompt:
    """GET /companion/prompt"""

    @patch(COMPANION_TARGET)
    def test_get_prompt(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.get("/companion/prompt")
        assert resp.status_code == 200
        body = resp.json()
        data = body["data"]
        assert "system_prompt" in data
        assert data["system_prompt"] == "You are a warm friend."


class TestChat:
    """POST /companion/chat"""

    @patch(COMPANION_TARGET)
    def test_chat(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={"message": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "system_prompt" in data

    @patch(COMPANION_TARGET)
    def test_chat_with_mood(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={
            "message": "I'm feeling sad",
            "user_mood": "sad",
        })
        assert resp.status_code == 200
        comp.adjust_for_mood.assert_called_once_with("sad")

    @patch(COMPANION_TARGET)
    def test_chat_no_system_prompt(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={
            "message": "Hi",
            "include_system_prompt": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_prompt"] == ""


class TestListPresets:
    """GET /companion/presets"""

    def test_list_presets(self):
        resp = client.get("/companion/presets")
        assert resp.status_code == 200
        body = resp.json()
        data = body["data"]
        assert "presets" in data
        assert len(data["presets"]) == 4
        ids = [p["id"] for p in data["presets"]]
        assert "warm" in ids
        assert "curious" in ids
        assert "playful" in ids
        assert "balanced" in ids
