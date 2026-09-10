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


class TestExportTrainingHistory:
    def test_export_json(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/history/export?format=json")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["format"] == "json"
        assert "outcomes" in data
        assert "count" in data

    def test_export_csv(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/history/export?format=csv")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["format"] == "csv"
        assert "content" in data

    def test_export_with_limit(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/history/export?format=json&limit=5")
        assert resp.status_code == 200


class TestGenerateModelCard:
    def test_generate_card(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/model-card?name=test-model&base_model=gpt2&description=A+test+model")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "card" in data
        assert "markdown" in data
        assert data["card"]["model_name"] == "test-model"
        assert "test-model" in data["markdown"]


class TestCompareTrainingRuns:
    def test_compare_nonexistent(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/compare?run_a=nonexistent&run_b=also_nonexistent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data


class TestBatchTrainingStatus:
    def test_batch_status_returns_structure(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/batch-status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "jobs" in data
        assert "summary" in data
        assert "total" in data["summary"]
        assert "running" in data["summary"]
        assert isinstance(data["jobs"], list)


class TestTrainingPresets:
    def test_list_presets(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/presets")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "presets" in data
        assert len(data["presets"]) > 0
        assert "name" in data["presets"][0]
        assert "model" in data["presets"][0]

    def test_get_preset(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/presets/quick-finetune")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "Quick Fine-Tune"
        assert data["model"] == "gpt2"

    def test_get_preset_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/presets/nonexistent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_apply_preset(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/presets/quick-finetune/apply")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["preset"] == "quick-finetune"
        assert "applied" in data
        assert "settings" in data


class TestTrainingRunManagement:
    def test_get_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/runs/nonexistent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_delete_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.delete("/settings/training/runs/nonexistent")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["deleted"] is False

    def test_filter_runs(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/runs")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "runs" in data
        assert "count" in data
        assert isinstance(data["runs"], list)

    def test_filter_runs_by_model(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/runs?model=gpt2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "runs" in data

    def test_clear_history(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/history/clear")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["cleared"] is True
        assert "removed_count" in data


class TestTrainingRunTags:
    def test_add_tag_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/nonexistent/tags?tag=best")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_remove_tag_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.delete("/settings/training/runs/nonexistent/tags/best")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_set_notes_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.put("/settings/training/runs/nonexistent/notes?notes=test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_get_all_tags(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/tags")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "tags" in data
        assert isinstance(data["tags"], list)

    def test_get_runs_by_tag(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/tags/best")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "runs" in data
        assert "count" in data
        assert data["tag"] == "best"


class TestExportTrainingRun:
    def test_export_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/runs/nonexistent/export")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_export_run_json(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/runs/nonexistent/export?format=json")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data


class TestBookmark:
    def test_toggle_bookmark_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/nonexistent/bookmark")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_get_bookmarked_runs(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.get("/settings/training/bookmarks")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "runs" in data
        assert "count" in data
        assert isinstance(data["runs"], list)


class TestDuplicateTrainingRun:
    def test_duplicate_run_not_found(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/nonexistent/duplicate")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data

    def test_duplicate_run_with_id(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/nonexistent/duplicate?new_run_id=test-1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "error" in data


class TestBulkOperations:
    def test_bulk_delete(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/bulk/delete?run_ids=run-1,run-2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "deleted_count" in data
        assert "requested" in data

    def test_bulk_add_tag(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/bulk/tag?run_ids=run-1,run-2&tag=best")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "updated_count" in data
        assert data["tag"] == "best"

    def test_bulk_bookmark(self):
        sr = SettingsRouter()
        client = TestClient(_app(sr))
        resp = client.post("/settings/training/runs/bulk/bookmark?run_ids=run-1,run-2&bookmarked=true")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "updated_count" in data
        assert data["bookmarked"] is True
