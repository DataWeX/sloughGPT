"""
Tests for models router — list, current, hf, cache-usage, export-formats.

Only registers the models router to avoid pulling in heavy dependencies.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers.models import router as models_router

app = FastAPI()
app.include_router(models_router)
client = TestClient(app)


@pytest.fixture(scope="session")
def fake_cache_dir():
    """Create a temp HF cache directory with mock blobs for cache-usage test."""
    tmp = Path(tempfile.mkdtemp())
    # Create a models--gpt2 directory with a blob file
    blob_dir = tmp / "models--gpt2" / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_file = blob_dir / "abc123"
    blob_file.write_text("x" * 1024 * 1024)  # 1 MB blob
    yield tmp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_CURRENT_MODEL = {
    "model_id": "gpt2",
    "device": "cpu",
    "parameters": 124_000_000,
    "vocab_size": 50257,
    "loaded_at": "2025-01-01T00:00:00",
}

SAMPLE_HF_MODELS = [
    {"model_id": "gpt2", "parameters": 124_000_000, "vocab_size": 50257},
    {"model_id": "gpt2-medium", "parameters": 355_000_000, "vocab_size": 50257},
    {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "parameters": 500_000_000, "vocab_size": 151936},
    {"model_id": "gpt2-large", "parameters": 774_000_000, "vocab_size": 50257},
]


@pytest.fixture(autouse=True)
def mock_controller():
    ctrl = MagicMock()
    ctrl.get_current_model.return_value = SAMPLE_CURRENT_MODEL
    ctrl.list_hf_models.return_value = SAMPLE_HF_MODELS
    ctrl.get_cache_usage.return_value = {
        "total_bytes": 2_500_000_000,
        "total_gb": 2.5,
        "model_count": 3,
        "cache_dir": "/home/user/.cache/huggingface/hub",
    }

    with patch("routers.models.get_models_controller", return_value=ctrl):
        yield ctrl


# ── GET /models ────────────────────────────────────────────────────────────

class TestListModels:

    def test_list_includes_loaded_model(self, mock_controller):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4  # 1 loaded + 3 unique HF models (gpt2 deduped)
        loaded = [m for m in data if m["status"] == "loaded"]
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "gpt2"

    def test_list_includes_hf_models(self, mock_controller):
        resp = client.get("/models")
        data = resp.json()
        available = [m for m in data if m["status"] == "available"]
        assert len(available) == 3  # gpt2-medium, Qwen, gpt2-large
        ids = {m["model_id"] for m in available}
        assert "gpt2-medium" in ids
        assert "Qwen/Qwen2.5-0.5B-Instruct" in ids

    def test_list_no_loaded_model(self, mock_controller):
        mock_controller.get_current_model.return_value = None
        resp = client.get("/models")
        data = resp.json()
        assert len(data) == 4  # all 4 HF models (none loaded)
        assert all(m["status"] == "available" for m in data)

    def test_list_model_has_description(self, mock_controller):
        resp = client.get("/models")
        data = resp.json()
        for m in data:
            assert "description" in m
            assert len(m["description"]) > 5

    def test_list_loaded_model_has_vocab_size(self, mock_controller):
        resp = client.get("/models")
        data = resp.json()
        loaded = [m for m in data if m["status"] == "loaded"][0]
        assert loaded["vocab_size"] == 50257


# ── GET /models/current ────────────────────────────────────────────────────

class TestCurrentModel:

    def test_current_returns_loaded(self, mock_controller):
        resp = client.get("/models/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "gpt2"

    def test_current_none_loaded(self, mock_controller):
        mock_controller.get_current_model.return_value = None
        resp = client.get("/models/current")
        assert resp.status_code == 404


# ── GET /models/hf ─────────────────────────────────────────────────────────

class TestHFModels:

    def test_hf_returns_all(self, mock_controller):
        resp = client.get("/models/hf")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "q" in data
        assert len(data["models"]) >= 4

    def test_hf_with_query(self, mock_controller):
        client.get("/models/hf?q=gpt2")
        mock_controller.list_hf_models.assert_called_once()

    def test_hf_model_has_cached_flag(self, mock_controller):
        resp = client.get("/models/hf")
        data = resp.json()
        for m in data["models"]:
            assert "cached" in m


# ── GET /models/cache-usage ────────────────────────────────────────────────

class TestCacheUsage:

    def test_cache_usage(self, mock_controller, fake_cache_dir):
        with patch("routers.models._hf_cache_dir", fake_cache_dir):
            resp = client.get("/models/cache-usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_count"] == 1
        assert data["total_bytes"] >= 1_000_000

    def test_cache_usage_structure(self, mock_controller, fake_cache_dir):
        with patch("routers.models._hf_cache_dir", fake_cache_dir):
            resp = client.get("/models/cache-usage")
        data = resp.json()
        assert "total_bytes" in data
        assert "cache_dir" in data

    def test_cache_usage_empty(self, mock_controller):
        fake_empty = tempfile.mkdtemp()
        with patch("routers.models._hf_cache_dir", Path(fake_empty)):
            resp = client.get("/models/cache-usage")
        data = resp.json()
        assert data["model_count"] == 0
        assert data["total_bytes"] == 0


# ── GET /models/export/formats ────────────────────────────────────────────

class TestExportFormats:

    def test_export_formats(self, mock_controller):
        resp = client.get("/models/export/formats")
        assert resp.status_code == 200
        data = resp.json()
        assert "formats" in data
        assert isinstance(data["formats"], dict)
        assert "safetensors" in data["formats"]
        assert "gguf_q4_k_m" in data["formats"]
        assert "sou" in data["formats"]

    def test_export_formats_has_descriptions(self, mock_controller):
        resp = client.get("/models/export/formats")
        data = resp.json()
        for key, desc in data["formats"].items():
            assert isinstance(key, str)
            assert isinstance(desc, str)
            assert len(desc) > 5
