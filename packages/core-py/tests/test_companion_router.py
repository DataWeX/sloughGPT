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
    defaults = {
        # Companion.traits is a PUBLIC attribute object, not a private dict:
        # routers/companion.py reads `current.warmth` / `current.name` straight
        # off it when merging a partial update or a preset.
        "traits": SimpleNamespace(
            name="Friend",
            warmth=0.7,
            curiosity=0.6,
            creativity=0.5,
            confidence=0.5,
            humor=0.4,
        ),
        "set_personality": lambda **kw: None,
        "get_system_prompt": lambda: "You are a friendly companion.",
        "build_system_prompt": lambda: "You are a friendly companion.",
        "adjust_for_mood": lambda mood: None,
    }
    defaults.update(overrides)
    ns = SimpleNamespace(**defaults)

    # Mirrors Companion.set_personality: traits mutate in place.
    def _set_personality(**kw):
        for key, val in kw.items():
            setattr(ns.traits, key, val)

    ns.set_personality = _set_personality
    # Mirrors Companion.to_dict: {"traits": {...}, "system_prompt": ...}
    # (domain/companion/_internal/companion.py:218 -- no top-level "name").
    ns.to_dict = lambda: {
        "traits": {
            k: getattr(ns.traits, k)
            for k in ("name", "warmth", "curiosity", "creativity", "confidence", "humor")
        },
        "system_prompt": "You are a friendly companion.",
    }

    # Companion.respond(user_message, context=None) -> str; the chat route
    # calls it positionally by keyword and needs a plain string back.
    ns.respond = lambda user_message, context=None: "Hello there!"

    # generate() async method — checks provider availability
    async def _generate(**kwargs):
        from domain.models._internal.provider import get_provider

        if get_provider("default") is None:
            raise RuntimeError("No model loaded")
        return "Hello there!"

    ns.generate = _generate
    return ns


def _app():
    app = FastAPI()
    app.include_router(router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Tests — patch at domain.companion.* (lazy imports inside handler body)
# ---------------------------------------------------------------------------

PATCH_GET = "domain.companion.get_companion"
PATCH_CREATE = "domain.companion.create_companion"


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
        resp = client.post(
            "/companion/personality",
            json={
                "name": "Alice",
                "warmth": 0.9,
                "curiosity": 0.8,
                "creativity": 0.7,
                "confidence": 0.6,
                "humor": 0.5,
            },
        )
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
    @patch("routers.companion._get_db")
    @patch(PATCH_CREATE)
    @patch(PATCH_GET)
    def test_apply_preset(self, mock_get, mock_create, mock_db):
        mock_get.return_value = _make_companion()
        comp = _make_companion()
        mock_create.return_value = comp
        # Mock the database to return a preset
        mock_col = MagicMock()
        mock_col.find_one.return_value = {"id": "warm", "traits": {"warmth": 0.9}}
        mock_db.return_value.collection.return_value = mock_col
        client = TestClient(_app())
        resp = client.post("/companion/preset", json="warm")
        assert resp.status_code == 200
        data = resp.json()["data"]
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
    @patch("routers.companion._get_db")
    @patch(PATCH_GET)
    def test_returns_preset_list(self, mock_get, mock_db):
        mock_get.return_value = _make_companion()
        # Hermetic on purpose: the handler seeds the collection only when it is
        # empty, then returns whatever is in it. Reading the live collection
        # made this an exact count of a mutable store -- it drifted 4 -> 5 as
        # soon as anything else wrote a preset. Stub the four documented
        # defaults so the assertion means what it says.
        preset_docs = [
            {"id": "warm", "name": "Warm Friend"},
            {"id": "curious", "name": "Curious Friend"},
            {"id": "playful", "name": "Playful Friend"},
            {"id": "balanced", "name": "Balanced Friend"},
        ]
        mock_col = MagicMock()
        mock_col.count.return_value = len(preset_docs)
        mock_col.find.return_value = preset_docs
        mock_db.return_value.collection.return_value = mock_col
        client = TestClient(_app())
        resp = client.get("/companion/presets")
        assert resp.status_code == 200
        presets = resp.json()["data"]["presets"]
        assert len(presets) == 4
        ids = [p["id"] for p in presets]
        assert "warm" in ids
        assert "curious" in ids


class TestChat:
    @patch("domain.models._internal.provider.get_provider")
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
    def test_chat_has_no_model_gate(self, mock_get):
        # These two tests used to assert 503 "No model loaded". The gate is
        # gone: neither the chat route nor Companion.respond() consults a
        # provider -- respond() builds a prompt and returns it
        # (domain/companion/_internal/companion.py:180), and grep for
        # get_provider / "No model loaded" in routers/companion.py is empty.
        # Chat is provider-independent now, so assert that instead.
        comp = _make_companion()
        mock_get.return_value = comp
        with patch("domain.models._internal.provider.get_provider", return_value=None):
            client = TestClient(_app())
            resp = client.post("/companion/chat", json={"message": "Hi"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "Hello there!"

    @patch(PATCH_GET)
    def test_chat_accepts_user_mood(self, mock_get):
        # user_mood is accepted by the request schema but the chat route never
        # reads it -- there is no adjust_for_mood call anywhere on the path.
        # Asserting the 200 records that honestly. Actually wiring mood into
        # chat is a product decision, not test drift, and is left alone here.
        comp = _make_companion()
        mock_get.return_value = comp
        with patch("domain.models._internal.provider.get_provider", return_value=None):
            client = TestClient(_app())
            resp = client.post("/companion/chat", json={"message": "Hi", "user_mood": "happy"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "Hello there!"
