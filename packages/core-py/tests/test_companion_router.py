"""Tests for the companion API router (routers/companion.py).

Covers: CompanionRouter get/set/patch/reset preset/prompt/chat/presets.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: the companion router imports get_companion / create_companion inside
the handler function body, so we must patch at the domain module level.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.companion import router  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_companion(**overrides):
    defaults = dict(
        _traits={"name": "Friend", "warmth": 0.7, "curiosity": 0.6, "creativity": 0.5, "confidence": 0.5, "humor": 0.4},
        set_personality=lambda **kw: None,
        get_system_prompt=lambda: "You are a friendly companion.",
        adjust_for_mood=lambda mood: None,
    )
    defaults.update(overrides)
    ns = SimpleNamespace(**defaults)
    # Allow set_personality to update _traits
    def _set_personality(**kw):
        ns._traits.update(kw)
    ns.set_personality = _set_personality
    # Make to_dict return current traits
    ns.to_dict = lambda: {"name": ns._traits.get("name", "Friend"), "traits": dict(ns._traits)}
    return ns


def _app():
    app = FastAPI()
    app.include_router(router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Tests — patch at domains.companion.* (lazy imports inside handler body)
# ---------------------------------------------------------------------------

PATCH_GET = "domains.companion.get_companion"
PATCH_CREATE = "domains.companion.create_companion"


class TestGetCompanionInfo:
    @patch(PATCH_GET)
    def test_returns_traits(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        client = TestClient(_app())
        resp = client.get("/companion/")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "traits" in data
        assert data["traits"]["warmth"] == 0.7


class TestSetPersonality:
    @patch(PATCH_GET)
    def test_full_replacement(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        client = TestClient(_app())
        resp = client.post("/companion/personality", json={
            "name": "Alice",
            "warmth": 0.9,
            "curiosity": 0.8,
            "creativity": 0.7,
            "confidence": 0.6,
            "humor": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "traits" in data


class TestPatchPersonality:
    @patch(PATCH_GET)
    def test_partial_update(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        client = TestClient(_app())
        resp = client.patch("/companion/personality", json={"warmth": 0.95})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "traits" in data


class TestResetCompanion:
    @patch(PATCH_CREATE)
    @patch(PATCH_GET)
    def test_reset_to_default(self, mock_get, mock_create):
        mock_get.return_value = _make_companion()
        mock_create.return_value = _make_companion()
        client = TestClient(_app())
        resp = client.delete("/companion/")
        assert resp.status_code == 200
        assert resp.json()["data"]["reset"] is True
        mock_create.assert_called_once()


class TestUsePreset:
    @patch(PATCH_CREATE)
    @patch(PATCH_GET)
    def test_apply_preset(self, mock_get, mock_create):
        mock_get.return_value = _make_companion()
        comp = _make_companion()
        mock_create.return_value = comp
        client = TestClient(_app())
        resp = client.post("/companion/preset", json={"name": "Alice", "preset": "warm"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["preset"] == "warm"
        assert "traits" in data


class TestGetPrompt:
    @patch(PATCH_GET)
    def test_returns_system_prompt(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        client = TestClient(_app())
        resp = client.get("/companion/prompt")
        assert resp.status_code == 200
        assert "You are a friendly" in resp.json()["data"]["system_prompt"]


class TestListPresets:
    @patch(PATCH_GET)
    def test_returns_preset_list(self, mock_get):
        mock_get.return_value = _make_companion()
        client = TestClient(_app())
        resp = client.get("/companion/presets")
        assert resp.status_code == 200
        presets = resp.json()["data"]["presets"]
        assert len(presets) == 4
        ids = [p["id"] for p in presets]
        assert "warm" in ids
        assert "curious" in ids


class TestChat:
    @patch("domains.models.provider.get_provider")
    @patch(PATCH_GET)
    def test_chat_with_model(self, mock_get, mock_provider_fn):
        comp = _make_companion()
        mock_get.return_value = comp

        async def _chat(messages, max_tokens=256, temperature=0.7):
            return "Hello there!"

        mock_provider = MagicMock()
        mock_provider.chat = _chat
        mock_provider_fn.return_value = mock_provider

        client = TestClient(_app())
        resp = client.post("/companion/chat", json={"message": "Hi", "include_system_prompt": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello there!"
        assert "system_prompt" in data

    @patch(PATCH_GET)
    def test_chat_no_model_returns_error_message(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        with patch("domains.models.provider.get_provider", return_value=None):
            client = TestClient(_app())
            resp = client.post("/companion/chat", json={"message": "Hi"})
        assert resp.status_code == 503
        assert "No model loaded" in resp.json()["error"]

    @patch(PATCH_GET)
    def test_chat_with_mood_adjustment(self, mock_get):
        comp = _make_companion()
        mock_get.return_value = comp
        with patch("domains.models.provider.get_provider", return_value=None):
            client = TestClient(_app())
            resp = client.post("/companion/chat", json={"message": "Hi", "user_mood": "happy"})
        assert resp.status_code == 503
