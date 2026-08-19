from infrastructure.exception_handlers import register_app_error_handler
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
register_app_error_handler(app)
app.include_router(models_router)
client = TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    return body.get("data", body)


@pytest.fixture(scope="session")
def fake_cache_dir():
    """Create a temp HF cache directory with mock blobs for cache-usage test."""
    tmp = Path(tempfile.mkdtemp())
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
        data = _data(resp)
        assert len(data) == 4  # 1 loaded + 3 unique HF models (gpt2 deduped)
        loaded = [m for m in data if m["status"] == "loaded"]
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "gpt2"

    def test_list_includes_hf_models(self, mock_controller):
        resp = client.get("/models")
        data = _data(resp)
        available = [m for m in data if m["status"] == "available"]
        assert len(available) == 3  # gpt2-medium, Qwen, gpt2-large
        ids = {m["model_id"] for m in available}
        assert "gpt2-medium" in ids
        assert "Qwen/Qwen2.5-0.5B-Instruct" in ids

    def test_list_no_loaded_model(self, mock_controller):
        mock_controller.get_current_model.return_value = None
        resp = client.get("/models")
        data = _data(resp)
        assert len(data) == 4  # all 4 HF models (none loaded)
        assert all(m["status"] == "available" for m in data)

    def test_list_model_has_description(self, mock_controller):
        resp = client.get("/models")
        data = _data(resp)
        for m in data:
            assert "description" in m
            assert len(m["description"]) > 5

    def test_list_loaded_model_has_vocab_size(self, mock_controller):
        resp = client.get("/models")
        data = _data(resp)
        loaded = [m for m in data if m["status"] == "loaded"][0]
        assert loaded["vocab_size"] == 50257

    def test_list_non_integral_parameters_coerced_to_int(self, mock_controller):
        """Regression: HF Hub num_parameters can be a fractional float; int field
        must not raise ValidationError (int_from_float) → 422 fields=1."""
        mock_controller.list_hf_models.return_value = [
            {"model_id": "gpt2-xl", "parameters": 1558000000.5, "vocab_size": 50257.0},
        ]
        resp = client.get("/models")
        assert resp.status_code == 200
        data = _data(resp)
        m = [x for x in data if x["model_id"] == "gpt2-xl"][0]
        assert m["parameters"] == 1558000000
        assert isinstance(m["parameters"], int)
        assert m["vocab_size"] == 50257

    def test_list_none_parameters_coerced_to_int(self, mock_controller):
        """Regression: None parameters must not raise int_type validation error."""
        mock_controller.list_hf_models.return_value = [
            {"model_id": "gpt2-medium", "parameters": None, "vocab_size": None},
        ]
        resp = client.get("/models")
        assert resp.status_code == 200
        data = _data(resp)
        m = [x for x in data if x["model_id"] == "gpt2-medium"][0]
        assert m["parameters"] == 0
        assert m["vocab_size"] == 0

    def test_list_string_parameters_coerced_to_int(self, mock_controller):
        """Regression: string parameters must not crash _describe_model (< int)."""
        mock_controller.list_hf_models.return_value = [
            {"model_id": "gpt2-xl", "parameters": "1558000000", "vocab_size": "50257"},
        ]
        resp = client.get("/models")
        assert resp.status_code == 200
        data = _data(resp)
        m = [x for x in data if x["model_id"] == "gpt2-xl"][0]
        assert m["parameters"] == 1558000000
        assert isinstance(m["parameters"], int)
        assert m["vocab_size"] == 50257
        assert len(m["description"]) > 5


# ── GET /models/current ────────────────────────────────────────────────────

class TestCurrentModel:

    def test_current_returns_loaded(self, mock_controller):
        resp = client.get("/models/current")
        assert resp.status_code == 200
        data = _data(resp)
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
        data = _data(resp)
        assert isinstance(data, list)
        assert len(data) >= 4

    def test_hf_with_query(self, mock_controller):
        client.get("/models/hf?q=gpt2")
        mock_controller.list_hf_models.assert_called_once()

    def test_hf_model_has_cached_flag(self, mock_controller):
        resp = client.get("/models/hf")
        data = _data(resp)
        for m in data:
            assert "cached" in m


# ── GET /models/cache-usage ────────────────────────────────────────────────

class TestCacheUsage:

    def test_cache_usage(self, mock_controller, fake_cache_dir):
        with patch("routers.models._hf_cache_dir", fake_cache_dir):
            resp = client.get("/models/cache-usage")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["model_count"] == 1
        assert data["total_bytes"] >= 1_000_000

    def test_cache_usage_structure(self, mock_controller, fake_cache_dir):
        with patch("routers.models._hf_cache_dir", fake_cache_dir):
            resp = client.get("/models/cache-usage")
        data = _data(resp)
        assert "total_bytes" in data
        assert "cache_dir" in data

    def test_cache_usage_empty(self, mock_controller):
        fake_empty = tempfile.mkdtemp()
        with patch("routers.models._hf_cache_dir", Path(fake_empty)):
            resp = client.get("/models/cache-usage")
        data = _data(resp)
        assert data["model_count"] == 0
        assert data["total_bytes"] == 0


# ── GET /models/export/formats ────────────────────────────────────────────

class TestExportFormats:

    def test_export_formats(self, mock_controller):
        resp = client.get("/models/export/formats")
        assert resp.status_code == 200
        data = _data(resp)
        assert isinstance(data, dict)
        assert "safetensors" in data
        assert "gguf_q4_k_m" in data
        assert "sou" in data

    def test_export_formats_has_descriptions(self, mock_controller):
        resp = client.get("/models/export/formats")
        data = _data(resp)
        for key, desc in data.items():
            assert isinstance(key, str)
            assert isinstance(desc, str)
            assert len(desc) > 5


# ── POST /models/load ───────────────────────────────────────────────────────

class TestLoadModel:

    LOADED = {
        "status": "loaded",
        "model_id": "gpt2",
        "type": "slonet",
        "device": "cpu",
        "loaded_at": "2025-01-01T00:00:00",
    }

    def test_load_returns_envelope_with_real_data(self, mock_controller):
        mock_controller.load_model.return_value = self.LOADED
        resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        body = resp.json()
        # Envelope must be preserved — the controller result must NOT be
        # stripped by response_model (regression for Bug A).
        assert body["status"] == "success"
        data = _data(resp)
        assert data["model_id"] == "gpt2"
        assert data["status"] == "loaded"
        assert data["type"] == "slonet"
        assert data["device"] == "cpu"
        assert data["loaded_at"] == "2025-01-01T00:00:00"

    def test_load_preserves_controller_error(self, mock_controller):
        mock_controller.load_model.return_value = {"status": "error", "error": "boom"}
        resp = client.post("/models/load", json={"model_id": "gpt2"})
        body = resp.json()
        assert body["error"] == "boom"
        assert body["code"] == "E_DOMAIN"

    def test_load_records_load_event_on_success(self, mock_controller):
        mock_controller.load_model.return_value = self.LOADED
        ss = MagicMock()
        with patch("domains.infrastructure.server_state.get_server_state", return_value=ss):
            resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        ss.record_model_event.assert_called_once_with("load", "gpt2", "device=cpu")

    def test_load_event_records_requested_device_when_result_has_none(self, mock_controller):
        loaded = dict(self.LOADED)
        loaded["device"] = None
        mock_controller.load_model.return_value = loaded
        ss = MagicMock()
        with patch("domains.infrastructure.server_state.get_server_state", return_value=ss):
            resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        ss.record_model_event.assert_called_once_with("load", "gpt2", "device=auto")

    def test_load_records_error_event_on_failure(self, mock_controller):
        mock_controller.load_model.return_value = {"status": "error", "error": "boom"}
        ss = MagicMock()
        with patch("domains.infrastructure.server_state.get_server_state", return_value=ss):
            resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        ss.record_model_event.assert_called_once_with("error", "gpt2", "boom")

    def test_load_calls_controller_with_args(self, mock_controller):
        mock_controller.load_model.return_value = self.LOADED
        client.post("/models/load", json={"model_id": "gpt2", "device": "cpu", "quantize": "q8"})
        mock_controller.load_model.assert_called_once_with("gpt2", "cpu", "q8")


# ── POST /models/unload ─────────────────────────────────────────────────────

class TestUnloadModel:

    def test_unload_records_event_with_model_id(self, mock_controller):
        mock_controller._current_model = "Qwen/Qwen2.5-0.5B-Instruct"
        ss = MagicMock()
        with patch("domains.infrastructure.server_state.get_server_state", return_value=ss):
            resp = client.post("/models/unload")
        assert resp.status_code == 200
        ss.record_model_event.assert_called_once_with("unload", "Qwen/Qwen2.5-0.5B-Instruct")

    def test_unload_falls_back_to_registry_default(self, mock_controller):
        mock_controller._current_model = None
        registry = MagicMock()
        registry.default_id = "Qwen/Qwen2.5-0.5B-Instruct"
        ss = MagicMock()
        with patch("domains.infrastructure.server_state.get_server_state", return_value=ss), \
             patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry):
            resp = client.post("/models/unload")
        assert resp.status_code == 200
        ss.record_model_event.assert_called_once_with("unload", "Qwen/Qwen2.5-0.5B-Instruct")


# ── GET/POST /models/process-guard ────────────────────────────────────────

class TestProcessGuard:

    def test_get_returns_status(self, mock_controller):
        mock_controller.get_process_guard_status.return_value = {
            "enabled": False, "active": False, "model_id": None, "health": None,
        }
        resp = client.get("/models/process-guard")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["enabled"] is False
        assert data["active"] is False

    def test_get_when_enabled_and_active(self, mock_controller):
        mock_controller.get_process_guard_status.return_value = {
            "enabled": True, "active": True, "model_id": "gpt2",
            "health": {"alive": True, "memory_mb": 512, "restarts": 0},
        }
        resp = client.get("/models/process-guard")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["enabled"] is True
        assert data["active"] is True
        assert data["model_id"] == "gpt2"

    def test_enable_calls_controller(self, mock_controller):
        mock_controller.set_process_guard_enabled.return_value = {
            "enabled": True, "active": False, "model_id": "gpt2", "health": None,
        }
        resp = client.post("/models/process-guard", json={"enabled": True})
        assert resp.status_code == 200
        mock_controller.set_process_guard_enabled.assert_called_once_with(True)
        data = _data(resp)
        assert data["enabled"] is True

    def test_disable_calls_controller(self, mock_controller):
        mock_controller.set_process_guard_enabled.return_value = {
            "enabled": False, "active": False, "model_id": None, "health": None,
        }
        resp = client.post("/models/process-guard", json={"enabled": False})
        assert resp.status_code == 200
        mock_controller.set_process_guard_enabled.assert_called_once_with(False)

    def test_rejects_non_boolean(self, mock_controller):
        resp = client.post("/models/process-guard", json={"enabled": "yes"})
        assert resp.status_code == 422
