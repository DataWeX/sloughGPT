"""Tests for the /companion router.

Covers all 10 endpoints:
  GET    /companion/                    — get_companion_info
  DELETE /companion/                    — reset_companion
  POST   /companion/personality         — set_personality
  PATCH  /companion/personality         — patch_personality
  POST   /companion/preset              — use_preset
  GET    /companion/prompt              — get_prompt
  POST   /companion/chat                — chat
  GET    /companion/presets             — list_presets
  POST   /companion/presets             — create_preset
  DELETE /companion/presets/{preset_id} — delete_preset
"""

import time
import pytest
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


def _traits(resp):
    """Extract traits dict from companion response."""
    return _d(resp).get("traits", _d(resp))


@pytest.fixture(autouse=True)
def _fresh_companion():
    """Reset the companion singleton before each test."""
    import routers.companion as comp_mod
    import domains.companion as dom_mod

    comp_mod._companion_router._companion = None
    dom_mod._companion = None
    yield
    comp_mod._companion_router._companion = None
    dom_mod._companion = None


class TestGetCompanionInfo:
    def setup_method(self):
        self.client = get_test_client()

    def test_returns_companion_state(self):
        resp = self.client.get("/companion/")
        assert resp.status_code == 200
        data = _d(resp)
        assert "traits" in data
        assert "name" in data["traits"]
        assert "warmth" in data["traits"]


class TestResetCompanion:
    def setup_method(self):
        self.client = get_test_client()

    def test_reset_returns_ok(self):
        resp = self.client.delete("/companion/")
        assert resp.status_code == 200
        assert _d(resp)["reset"] is True

    def test_reset_restores_defaults(self):
        self.client.post(
            "/companion/personality",
            json={"name": "Changed", "warmth": 0.1, "curiosity": 0.1, "creativity": 0.1, "confidence": 0.1, "humor": 0.1},
        )
        self.client.delete("/companion/")
        resp = self.client.get("/companion/")
        t = _traits(resp)
        assert t["name"] == "Friend"
        assert t["warmth"] == 0.9


class TestSetPersonality:
    def setup_method(self):
        self.client = get_test_client()

    def test_set_full_personality(self):
        resp = self.client.post(
            "/companion/personality",
            json={"name": "Buddy", "warmth": 0.9, "curiosity": 0.8, "creativity": 0.7, "confidence": 0.6, "humor": 0.5},
        )
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["name"] == "Buddy"
        assert t["warmth"] == 0.9
        assert t["curiosity"] == 0.8

    def test_set_with_defaults(self):
        resp = self.client.post("/companion/personality", json={"name": "Solo"})
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["name"] == "Solo"
        assert t["warmth"] == 0.7

    def test_set_missing_body(self):
        resp = self.client.post("/companion/personality", json={})
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["name"] == "Friend"


class TestPatchPersonality:
    def setup_method(self):
        self.client = get_test_client()

    def test_patch_name_only(self):
        resp = self.client.patch("/companion/personality", json={"name": "Patched"})
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["name"] == "Patched"
        assert t["warmth"] == 0.7

    def test_patch_warmth_only(self):
        resp = self.client.patch("/companion/personality", json={"warmth": 0.3})
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["warmth"] == 0.3
        assert t["name"] == "Friend"

    def test_patch_multiple_fields(self):
        resp = self.client.patch(
            "/companion/personality", json={"name": "Multi", "humor": 0.99}
        )
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["name"] == "Multi"
        assert t["humor"] == 0.99

    def test_patch_empty_body(self):
        resp = self.client.patch("/companion/personality", json={})
        assert resp.status_code == 200


class TestUsePreset:
    def setup_method(self):
        self.client = get_test_client()

    def test_use_existing_preset(self):
        resp = self.client.post("/companion/preset", json="warm")
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["warmth"] == 0.9

    def test_use_nonexistent_preset(self):
        resp = self.client.post("/companion/preset", json="nonexistent")
        assert resp.status_code == 404

    def test_use_preset_missing_body(self):
        resp = self.client.post("/companion/preset")
        assert resp.status_code == 422


class TestGetPrompt:
    def setup_method(self):
        self.client = get_test_client()

    def test_returns_system_prompt(self):
        resp = self.client.get("/companion/prompt")
        assert resp.status_code == 200
        data = _d(resp)
        assert "system_prompt" in data
        assert isinstance(data["system_prompt"], str)
        assert len(data["system_prompt"]) > 0


class TestChat:
    def setup_method(self):
        self.client = get_test_client()

    def test_chat_basic(self):
        resp = self.client.post("/companion/chat", json={"message": "hello"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "response" in data
        assert "system_prompt" in data
        assert "elapsed_ms" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_with_options(self):
        resp = self.client.post(
            "/companion/chat",
            json={"message": "hi", "user_name": "Test", "user_mood": "happy", "max_tokens": 128, "temperature": 0.5},
        )
        assert resp.status_code == 200
        assert "response" in _d(resp)

    def test_chat_without_system_prompt(self):
        resp = self.client.post(
            "/companion/chat", json={"message": "hello", "include_system_prompt": False}
        )
        assert resp.status_code == 200
        assert _d(resp)["system_prompt"] == ""

    def test_chat_missing_message(self):
        resp = self.client.post("/companion/chat", json={})
        assert resp.status_code == 422

    def test_chat_empty_message(self):
        resp = self.client.post("/companion/chat", json={"message": ""})
        assert resp.status_code == 422


class TestListPresets:
    def setup_method(self):
        self.client = get_test_client()

    def test_returns_presets_list(self):
        resp = self.client.get("/companion/presets")
        assert resp.status_code == 200
        data = _d(resp)
        assert "presets" in data
        assert isinstance(data["presets"], list)
        assert len(data["presets"]) >= 4

    def test_presets_have_required_fields(self):
        resp = self.client.get("/companion/presets")
        data = _d(resp)
        for preset in data["presets"]:
            assert "id" in preset
            assert "name" in preset
            assert "traits" in preset


class TestCreatePreset:
    def setup_method(self):
        self.client = get_test_client()

    def test_create_new_preset(self):
        pid = f"test-{int(time.time() * 1000)}"
        resp = self.client.post(
            "/companion/presets",
            json={"id": pid, "name": "Test Preset", "description": "A test", "traits": {"warmth": 0.8}},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["preset"]["id"] == pid
        assert data["preset"]["name"] == "Test Preset"

    def test_create_duplicate_preset(self):
        pid = f"dup-{int(time.time() * 1000)}"
        self.client.post(
            "/companion/presets",
            json={"id": pid, "name": "First", "traits": {}},
        )
        resp = self.client.post(
            "/companion/presets",
            json={"id": pid, "name": "Second", "traits": {}},
        )
        assert resp.status_code == 409

    def test_create_preset_invalid_id_format(self):
        resp = self.client.post(
            "/companion/presets",
            json={"id": "Invalid ID!", "name": "Bad", "traits": {}},
        )
        assert resp.status_code == 422

    def test_create_preset_missing_fields(self):
        resp = self.client.post("/companion/presets", json={"name": "No ID"})
        assert resp.status_code == 422


class TestDeletePreset:
    def setup_method(self):
        self.client = get_test_client()

    def test_delete_existing_preset(self):
        pid = f"del-{int(time.time() * 1000)}"
        self.client.post(
            "/companion/presets",
            json={"id": pid, "name": "To Delete", "traits": {}},
        )
        resp = self.client.delete(f"/companion/presets/{pid}")
        assert resp.status_code == 200
        assert _d(resp)["deleted"] == pid

    def test_delete_nonexistent_preset(self):
        resp = self.client.delete("/companion/presets/no-such-preset")
        assert resp.status_code == 404


class TestCompanionLifecycle:
    """End-to-end: get → set personality → patch → chat → reset."""

    def setup_method(self):
        self.client = get_test_client()

    def test_full_lifecycle(self):
        resp = self.client.get("/companion/")
        assert resp.status_code == 200
        original_name = _traits(resp)["name"]

        resp = self.client.post(
            "/companion/personality",
            json={"name": "Lifecycle", "warmth": 0.95},
        )
        assert resp.status_code == 200
        assert _traits(resp)["name"] == "Lifecycle"

        resp = self.client.patch("/companion/personality", json={"humor": 0.88})
        assert resp.status_code == 200
        t = _traits(resp)
        assert t["humor"] == 0.88
        assert t["name"] == "Lifecycle"

        resp = self.client.post("/companion/chat", json={"message": "test"})
        assert resp.status_code == 200
        assert "response" in _d(resp)

        resp = self.client.delete("/companion/")
        assert resp.status_code == 200
        resp = self.client.get("/companion/")
        assert _traits(resp)["name"] == original_name
