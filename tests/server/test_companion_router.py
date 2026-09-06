"""
Tests for the companion router — personality, presets, chat, prompt.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.companion import router
from apps.api.server.infrastructure.exception_handlers import register_all_handlers

app = FastAPI()
register_all_handlers(app)
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
        assert "traits" in data
        assert comp.name == "Alice"
        assert comp.warmth == 0.9
        assert comp.curiosity == 0.8
        assert comp.creativity == 0.7
        assert comp.confidence == 0.6
        assert comp.humor == 0.5

    @patch(COMPANION_TARGET)
    def test_set_personality_defaults(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.post("/companion/personality", json={"name": "Bob"})
        assert resp.status_code == 200
        assert comp.name == "Bob"

    def test_set_personality_warmth_out_of_range(self):
        resp = client.post("/companion/personality", json={
            "name": "Bad",
            "warmth": 1.5,
        })
        assert resp.status_code == 422

    def test_set_personality_negative_curiosity(self):
        resp = client.post("/companion/personality", json={
            "name": "Bad",
            "curiosity": -0.1,
        })
        assert resp.status_code == 422


class TestPatchPersonality:
    """PATCH /companion/personality"""

    @patch(COMPANION_TARGET)
    def test_patch_single_field(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.patch("/companion/personality", json={"warmth": 0.95})
        assert resp.status_code == 200
        body = resp.json()
        assert "traits" in body["data"]
        assert comp.warmth == 0.95

    @patch(COMPANION_TARGET)
    def test_patch_multiple_fields(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.patch("/companion/personality", json={
            "name": "Patched",
            "humor": 0.9,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "traits" in body["data"]

    def test_patch_out_of_range_value(self):
        resp = client.patch("/companion/personality", json={"confidence": 2.0})
        assert resp.status_code == 422

    @patch(COMPANION_TARGET)
    def test_patch_empty_body(self, mock_get):
        comp = _mock_companion()
        mock_get.return_value = comp

        resp = client.patch("/companion/personality", json={})
        assert resp.status_code == 200


class TestResetCompanion:
    """DELETE /companion/"""

    @patch("domains.companion.create_companion")
    @patch(COMPANION_TARGET)
    def test_reset_companion(self, mock_get, mock_create):
        new_comp = _mock_companion()
        mock_create.return_value = new_comp
        mock_get.return_value = _mock_companion()

        resp = client.delete("/companion/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["reset"] is True


class TestPreset:
    """POST /companion/preset"""

    @patch("domains.companion.create_companion")
    @patch(COMPANION_TARGET)
    def test_use_preset(self, mock_get, mock_create):
        new_comp = _mock_companion()
        mock_create.return_value = new_comp
        mock_get.return_value = _mock_companion()

        resp = client.post("/companion/preset", json="playful")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            body = resp.json()
            assert "traits" in body["data"]

    @patch("domains.companion.create_companion")
    @patch(COMPANION_TARGET)
    def test_use_preset_warm(self, mock_get, mock_create):
        new_comp = _mock_companion()
        mock_create.return_value = new_comp
        mock_get.return_value = _mock_companion()

        resp = client.post("/companion/preset", json="warm")
        assert resp.status_code in (200, 404)


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


class TestChat:
    """POST /companion/chat"""

    @patch(COMPANION_TARGET)
    def test_chat(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(return_value="Hello! I'm here for you.")
        comp.build_system_prompt = MagicMock(return_value="You are a warm friend.")
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={"message": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    @patch(COMPANION_TARGET)
    def test_chat_with_mood(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(return_value="I understand.")
        comp.build_system_prompt = MagicMock(return_value="You are a warm friend.")
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={
            "message": "I'm feeling sad",
            "user_mood": "sad",
        })
        assert resp.status_code == 200

    @patch(COMPANION_TARGET)
    def test_chat_no_system_prompt(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(return_value="Hi!")
        comp.build_system_prompt = MagicMock(return_value="You are a warm friend.")
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={
            "message": "Hi",
            "include_system_prompt": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_prompt"] == ""

    @patch(COMPANION_TARGET)
    def test_chat_with_user_name(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(return_value="Hello Alice!")
        comp.build_system_prompt = MagicMock(return_value="You are a warm friend.")
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={
            "message": "Hi there",
            "user_name": "Alice",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    @patch(COMPANION_TARGET)
    def test_chat_provider_error(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(side_effect=Exception("model crash"))
        comp.build_system_prompt = MagicMock(return_value="You are a warm friend.")
        mock_get.return_value = comp

        resp = client.post("/companion/chat", json={"message": "Hello"})
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data


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

    def test_presets_have_required_fields(self):
        resp = client.get("/companion/presets")
        presets = resp.json()["data"]["presets"]
        for p in presets:
            assert "id" in p
            assert "name" in p
            assert "description" in p


class TestSetPersonalityEdgeCases:
    """POST /companion/personality — validation + propagation."""

    def test_warmth_exactly_one_ok(self):
        resp = client.post("/companion/personality", json={"name": "Edge", "warmth": 1.0})
        assert resp.status_code == 200

    def test_warmth_exactly_zero_ok(self):
        resp = client.post("/companion/personality", json={"name": "Edge", "humor": 0.0})
        assert resp.status_code == 200

    def test_missing_name_uses_default(self):
        resp = client.post("/companion/personality", json={"warmth": 0.5})
        assert resp.status_code == 200

    def test_negative_humor_rejected(self):
        resp = client.post("/companion/personality", json={"humor": -0.01})
        assert resp.status_code == 422

    def test_name_too_long_rejected(self):
        resp = client.post("/companion/personality", json={"name": "x" * 101})
        assert resp.status_code == 422


class TestPatchPersonalityEdgeCases:
    """PATCH /companion/personality — validation edges."""

    def test_patch_name_only(self):
        resp = client.patch("/companion/personality", json={"name": "OnlyName"})
        assert resp.status_code == 200

    def test_patch_nulls_ignored(self):
        resp = client.patch("/companion/personality", json={
            "warmth": None, "curiosity": None,
        })
        assert resp.status_code == 200

    def test_patch_all_fields(self):
        resp = client.patch("/companion/personality", json={
            "name": "All", "warmth": 0.1, "curiosity": 0.2,
            "creativity": 0.3, "confidence": 0.4, "humor": 0.5,
        })
        assert resp.status_code == 200

    def test_patch_negative_creativity_rejected(self):
        resp = client.patch("/companion/personality", json={"creativity": -0.5})
        assert resp.status_code == 422

    def test_patch_above_one_rejected(self):
        resp = client.patch("/companion/personality", json={"curiosity": 1.01})
        assert resp.status_code == 422


class TestChatValidation:
    """POST /companion/chat — request validation."""

    def test_missing_message_422(self):
        resp = client.post("/companion/chat", json={})
        assert resp.status_code == 422

    def test_empty_message_rejected(self):
        resp = client.post("/companion/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_message_too_long_rejected(self):
        resp = client.post("/companion/chat", json={"message": "x" * 10001})
        assert resp.status_code == 422

    @patch(COMPANION_TARGET)
    def test_no_provider_returns_error_response(self, mock_get):
        comp = _mock_companion()
        comp.generate = AsyncMock(side_effect=Exception("No model loaded"))
        mock_get.return_value = comp
        resp = client.post("/companion/chat", json={"message": "Hello"})
        assert resp.status_code == 500


class TestPresetValidation:
    """POST /companion/preset — validation edges."""

    def test_preset_name_too_long_rejected(self):
        resp = client.post("/companion/preset", json={"name": "x" * 101, "preset": "warm"})
        assert resp.status_code == 422

    def test_unknown_preset_name_404(self):
        resp = client.post("/companion/preset", json="nonexistent")
        assert resp.status_code == 404


class TestMethodCoverage:
    """Method mismatch coverage."""

    def test_info_wrong_method_405(self):
        resp = client.post("/companion/")
        assert resp.status_code == 405

    def test_prompt_wrong_method_405(self):
        resp = client.post("/companion/prompt")
        assert resp.status_code == 405

    def test_presets_post_requires_body(self):
        resp = client.post("/companion/presets")
        assert resp.status_code == 422

    def test_chat_wrong_method_405(self):
        resp = client.get("/companion/chat")
        assert resp.status_code == 405
