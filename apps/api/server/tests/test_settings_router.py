"""Tests for the Settings router — core CRUD and section endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


# ── Helpers ──────────────────────────────────────────────────────────────────

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.settings import SettingsRouter

    router_obj = SettingsRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── GET /settings ────────────────────────────────────────────────────────────


class TestGetSettings:
    def test_get_all_settings_returns_dict(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], dict)

    def test_get_all_settings_has_sections(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings")
        data = resp.json()["data"]
        for section in ("generation", "training", "adaptive", "voice", "ui"):
            assert section in data


# ── Generation settings ─────────────────────────────────────────────────────


class TestGenerationSettings:
    def test_get_generation(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/generation")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "temperature" in data
        assert "top_p" in data

    def test_update_generation(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/generation",
            json={"temperature": 0.9, "top_p": 0.95},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["temperature"] == 0.9
        assert data["top_p"] == 0.95

    def test_update_generation_no_changes(self):
        _app = _build_app()
        resp = TestClient(_app).patch("/settings/generation", json={})
        assert resp.status_code == 200
        assert "message" in resp.json()["data"]

    def test_update_generation_validation(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/generation",
            json={"temperature": 5.0},  # out of range
        )
        assert resp.status_code == 422


# ── Training settings ───────────────────────────────────────────────────────


class TestTrainingSettings:
    def test_get_training(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "preferred_model" in data
        assert "auto_train" in data

    def test_update_training(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/training",
            json={"preferred_model": "gpt2", "auto_train": True},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["preferred_model"] == "gpt2"
        assert data["auto_train"] is True

    def test_update_training_no_changes(self):
        _app = _build_app()
        resp = TestClient(_app).patch("/settings/training", json={})
        assert resp.status_code == 200
        assert "message" in resp.json()["data"]


# ── Voice settings ──────────────────────────────────────────────────────────


class TestVoiceSettings:
    def test_get_voice(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/voice")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "noise_gate_db" in data
        assert "vad_enabled" in data

    def test_update_voice(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/voice",
            json={"noise_gate_db": -40.0, "vad_enabled": True},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["noise_gate_db"] == -40.0
        assert data["vad_enabled"] is True


# ── Adaptive settings ───────────────────────────────────────────────────────


class TestAdaptiveSettings:
    def test_get_adaptive(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/adaptive")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "enabled" in data
        assert "exploration_rate" in data

    def test_update_adaptive(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/adaptive",
            json={"enabled": True, "exploration_rate": 0.3},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["enabled"] is True
        assert data["exploration_rate"] == 0.3


# ── UI settings ─────────────────────────────────────────────────────────────


class TestUISettings:
    def test_get_ui(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/ui")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "theme" in data
        assert "language" in data

    def test_update_ui(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/ui",
            json={"theme": "dark", "language": "es"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["theme"] == "dark"
        assert data["language"] == "es"


# ── Reset ───────────────────────────────────────────────────────────────────


class TestResetSettings:
    def test_reset_settings(self):
        _app = _build_app()
        resp = TestClient(_app).post("/settings/reset")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "reset"

    def test_reset_then_get_defaults(self):
        _app = _build_app()
        # Update first
        TestClient(_app).patch(
            "/settings/generation",
            json={"temperature": 0.5},
        )
        # Reset
        TestClient(_app).post("/settings/reset")
        # Verify defaults restored
        resp = TestClient(_app).get("/settings/generation")
        data = resp.json()["data"]
        assert data["temperature"] != 0.5 or data["temperature"] == 1.0


# ── Training presets ────────────────────────────────────────────────────────


class TestTrainingPresets:
    def test_list_presets(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/presets")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)
        # Should have at least default presets
        assert len(data) > 0

    def test_get_preset(self):
        _app = _build_app()
        # First list to get a preset name
        list_resp = TestClient(_app).get("/settings/training/presets")
        presets = list_resp.json()["data"]
        if presets:
            preset_name = list(presets.keys())[0]
            resp = TestClient(_app).get(f"/settings/training/presets/{preset_name}")
            assert resp.status_code == 200

    def test_get_preset_not_found(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/presets/nonexistent")
        # Router returns 200 with empty/error data, not 404
        assert resp.status_code == 200


# ── Auto-train ──────────────────────────────────────────────────────────────


class TestAutoTrain:
    def test_auto_train_status(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)

    def test_auto_train_config_update(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/training/auto-train/config",
            params={"threshold": 50, "interval_s": 120},
        )
        assert resp.status_code == 200

    def test_auto_train_config_validation(self):
        _app = _build_app()
        resp = TestClient(_app).patch(
            "/settings/training/auto-train/config",
            params={"threshold": 0},  # below minimum
        )
        assert resp.status_code == 422


# ── Training runs ───────────────────────────────────────────────────────────


class TestTrainingRuns:
    def test_filter_training_runs(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/runs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)

    def test_get_training_run_not_found(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/runs/nonexistent")
        # Router returns 200 with empty/error data, not 404
        assert resp.status_code == 200

    def test_bulk_delete_runs_empty(self):
        _app = _build_app()
        resp = TestClient(_app).post(
            "/settings/training/runs/bulk/delete",
            params={"run_ids": ""},
        )
        assert resp.status_code == 200

    def test_bulk_bookmark_runs(self):
        _app = _build_app()
        resp = TestClient(_app).post(
            "/settings/training/runs/bulk/bookmark",
            params={"run_ids": "", "bookmarked": True},
        )
        assert resp.status_code == 200

    def test_get_all_tags(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/tags")
        assert resp.status_code == 200

    def test_get_bookmarked_runs(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/bookmarks")
        assert resp.status_code == 200


# ── Training history ────────────────────────────────────────────────────────


class TestTrainingHistory:
    def test_export_training_history(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/history/export")
        assert resp.status_code == 200

    def test_clear_training_history(self):
        _app = _build_app()
        resp = TestClient(_app).post("/settings/training/history/clear")
        assert resp.status_code == 200


# ── Batch status ────────────────────────────────────────────────────────────


class TestBatchStatus:
    def test_get_batch_training_status(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/training/batch-status")
        assert resp.status_code == 200


# ── Model card ──────────────────────────────────────────────────────────────


class TestModelCard:
    def test_generate_model_card(self):
        _app = _build_app()
        resp = TestClient(_app).post("/settings/model-card")
        assert resp.status_code == 200


# ── Adaptive insights ───────────────────────────────────────────────────────


class TestAdaptiveInsights:
    def test_get_adaptive_insights(self):
        _app = _build_app()
        resp = TestClient(_app).get("/settings/adaptive/insights")
        assert resp.status_code == 200
