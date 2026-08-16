"""
Tests for the models router — list, load, unload, HF models.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.models import ModelsRouter


@pytest.fixture
def router():
    r = ModelsRouter()
    r._hf_cache_dir = MagicMock()
    r._hf_cache_dir.exists.return_value = False
    return r


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListModels:
    """GET /models"""

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    def test_list_models_with_loaded(self, mock_cached, mock_size, mock_get_ctrl, client):
        mock_cached.return_value = False
        mock_size.return_value = 0.5
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = {
            "model_id": "gpt2", "device": "cpu",
            "parameters": 124000000, "vocab_size": 50257,
            "loaded_at": "2026-01-01T00:00:00",
        }
        ctrl.list_hf_models.return_value = []
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        models = body["data"]
        assert any(m["model_id"] == "gpt2" and m["status"] == "loaded" for m in models)

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_list_models_empty(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = None
        ctrl.list_hf_models.return_value = []
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_list_models_adopted_guard_reports_device(self, mock_get_ctrl, client, tmp_path):
        """Regression: after adopting an autoloaded ProcessGuard, GET /models
        must return 200 with a real device on the loaded entry. Previously the
        adopt path left device null and ModelInfo rejected it with a 422."""
        from pathlib import Path
        from apps.api.server.controllers.models import ModelsController

        class _FakeGuard:
            worker_id = "slo-guard"
            alive = True

            def __init__(self):
                self.device = "cpu"

            def start(self):
                pass

            def stop(self):
                pass

            def health(self):
                return {"alive": True}

        ctrl = ModelsController(repo_root=Path(tmp_path))
        ctrl.adopt_process_guard(_FakeGuard(), "gpt2")
        ctrl.list_hf_models = lambda: []
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models")
        assert resp.status_code == 200
        loaded = [m for m in resp.json()["data"] if m["status"] == "loaded"]
        assert len(loaded) == 1
        assert loaded[0]["model_id"] == "gpt2"
        assert loaded[0]["device"] is not None
        assert loaded[0]["device"] == "cpu"


class TestLoadModel:
    """POST /models/load"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_success(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {
            "status": "loaded", "model_id": "gpt2",
            "device": "cpu", "parameters": 124000000,
        }
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/load", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        ctrl.load_model.assert_called_once()

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_failure(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "error", "error": "model not found"}
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/load", json={"model_id": "nonexistent"})
        assert resp.status_code == 200
        ctrl.load_model.assert_called_once()

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_forwards_explicit_device(self, mock_get_ctrl, client):
        """The requested device enum must reach the controller; the controller
        validates availability and the response echoes the resolved device."""
        ctrl = MagicMock()
        ctrl.load_model.return_value = {
            "status": "loaded", "model_id": "gpt2",
            "device": "cpu", "parameters": 124000000,
        }
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/load", json={"model_id": "gpt2", "device": "cuda"})
        assert resp.status_code == 200
        ctrl.load_model.assert_called_once_with("gpt2", "cuda", None)
        assert resp.json()["data"]["device"] == "cpu"

    @patch("domains.infrastructure.server_state.get_server_state")
    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_event_records_resolved_device(self, mock_get_ctrl, mock_ss, client):
        """The model load event must record the resolved device (e.g. cpu after
        a cuda request on a GPU-less box), not the requested device string."""
        ctrl = MagicMock()
        ctrl.load_model.return_value = {
            "status": "loaded", "model_id": "gpt2",
            "device": "cpu", "parameters": 124000000,
        }
        mock_get_ctrl.return_value = ctrl
        ss = MagicMock()
        mock_ss.return_value = ss

        client.post("/models/load", json={"model_id": "gpt2", "device": "cuda"})
        args, kwargs = ss.record_model_event.call_args
        assert args[0] == "load"
        assert args[1] == "gpt2"
        assert args[2] == "device=cpu"

    @patch("domains.infrastructure.server_state.get_server_state")
    @patch("apps.api.server.routers.models.get_models_controller")
    def test_load_model_event_falls_back_to_requested_device(self, mock_get_ctrl, mock_ss, client):
        """If the controller response omits the resolved device, the event falls
        back to the requested device string so the detail is never empty."""
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "loaded", "model_id": "gpt2"}
        mock_get_ctrl.return_value = ctrl
        ss = MagicMock()
        mock_ss.return_value = ss

        client.post("/models/load", json={"model_id": "gpt2", "device": "mps"})
        args, kwargs = ss.record_model_event.call_args
        assert args[2] == "device=mps"


class TestUnloadModel:
    """POST /models/unload"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_unload_success(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.unload_model.return_value = {"status": "unloaded"}
        mock_get_ctrl.return_value = ctrl

        resp = client.post("/models/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"


class TestListHFModels:
    """GET /models/hf"""

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    @patch("apps.api.server.routers.models._hf_cache_dir")
    @patch("domains.infrastructure.resource_manager.get_resource_manager")
    def test_list_hf_models(self, mock_rm, mock_cached, mock_size, mock_cache_dir, mock_get_ctrl, client):
        mock_rm.return_value = MagicMock(inference_pool_size=2)
        mock_cached.return_value = False
        mock_size.return_value = 0.5
        mock_cache_dir.exists.return_value = False

        ctrl = MagicMock()
        ctrl.list_hf_models.return_value = [
            {"model_id": "gpt2", "parameters": 124000000},
            {"model_id": "gpt2-medium", "parameters": 355000000},
        ]
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models/hf")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2
        ids = [m["id"] for m in body["data"]]
        assert "gpt2" in ids
        assert "gpt2-medium" in ids

    @patch("apps.api.server.routers.models.get_models_controller")
    @patch("apps.api.server.routers.models.compute_model_size_gb")
    @patch("apps.api.server.routers.models.is_model_cached")
    @patch("domains.infrastructure.resource_manager.get_resource_manager")
    def test_list_hf_models_search(self, mock_rm, mock_cached, mock_size, mock_get_ctrl, client):
        mock_rm.return_value = MagicMock(inference_pool_size=2)
        mock_cached.return_value = False
        mock_size.return_value = 0.5

        ctrl = MagicMock()
        ctrl.list_hf_models.return_value = [
            {"model_id": "gpt2-xl", "parameters": 1500000000},
        ]
        mock_get_ctrl.return_value = ctrl

        resp = client.get("/models/hf?q=gpt2")
        assert resp.status_code == 200
        ctrl.list_hf_models.assert_called_with("gpt2")


class TestCurrentModel:
    """GET /models/current"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_returns_current(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = {"model_id": "gpt2", "device": "cpu"}
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/models/current")
        assert resp.status_code == 200
        assert resp.json()["data"]["model_id"] == "gpt2"

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_returns_404_when_empty(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_current_model.return_value = None
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/models/current")
        assert resp.status_code == 404


class TestModelLogs:
    """GET /models/logs"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_returns_logs(self, mock_get_ctrl, client):
        resp = client.get("/models/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)


class TestExportFormats:
    """GET /models/export/formats"""

    def test_returns_formats(self, client):
        resp = client.get("/models/export/formats")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], dict)


class TestExportModel:
    """POST /models/export"""

    def test_export_requires_loaded_model(self, client):
        import state as server_state
        prev = server_state.model
        try:
            server_state.model = None
            resp = client.post("/models/export", json={"output_path": "/tmp/x", "format": "sou"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["error"] == "No model loaded"
        finally:
            server_state.model = prev

    @patch("domains.training.export.export_model")
    def test_export_success(self, mock_export, client):
        mock_export.return_value = ["weights.sout", "a.sln"]
        import state as server_state
        prev = server_state.model
        prev_tok = server_state.tokenizer
        try:
            server_state.model = object()
            server_state.tokenizer = object()
            resp = client.post("/models/export", json={"output_path": "/tmp/x", "format": "sou"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"
            assert body["data"]["format"] == "sou"
            assert body["data"]["files"] == ["weights.sout", "a.sln"]
        finally:
            server_state.model = prev
            server_state.tokenizer = prev_tok


class TestUnloadFailure:
    """POST /models/unload — error path"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_unload_error_surfaces(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl._current_model = "gpt2"
        ctrl.unload_model.return_value = {"status": "error", "error": "model busy"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/models/unload")
        assert resp.status_code == 200
        assert resp.json()["error"] == "model busy"


class TestStartDownload:
    """POST /models/download"""

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_already_cached(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.is_cached.return_value = True
        mock_mgr.return_value = mgr
        resp = client.post("/models/download", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "already_cached"

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_already_downloading(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.is_cached.return_value = False
        mgr.is_downloading.return_value = True
        mock_mgr.return_value = mgr
        resp = client.post("/models/download", json={"model_id": "gpt2"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "already_downloading"

    @patch("apps.api.server.routers.models.ModelsRouter._run_download")
    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_started(self, mock_mgr, mock_run, client):
        mgr = MagicMock()
        mgr.is_cached.return_value = False
        mgr.is_downloading.return_value = False
        mock_mgr.return_value = mgr
        resp = client.post("/models/download", json={"model_id": "gpt2", "total_bytes_hint": 100})
        assert resp.status_code == 200
        assert resp.json()["message"] == "started"
        mock_run.assert_called_once_with("gpt2", 100)


class TestDownloadStatus:
    """GET /models/download/{model_id}"""

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_returns_progress(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.get_progress.return_value = {"model_id": "gpt2", "pct": 42.0, "status": "downloading"}
        mock_mgr.return_value = mgr
        resp = client.get("/models/download/gpt2")
        assert resp.status_code == 200
        assert resp.json()["data"]["pct"] == 42.0

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_not_found_reports_cached(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.get_progress.return_value = None
        mgr.is_cached.return_value = True
        mock_mgr.return_value = mgr
        resp = client.get("/models/download/gpt2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "not_found"
        assert body["data"]["cached"] is True


class TestListDownloads:
    """GET /models/downloads"""

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_returns_list_and_cleans_stale(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.list_downloads.return_value = [{"model_id": "gpt2", "pct": 10}]
        mock_mgr.return_value = mgr
        resp = client.get("/models/downloads")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1
        mgr.cleanup_stale.assert_called_once()


class TestCancelDownload:
    """POST /models/download/{model_id}/cancel"""

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_cancel_true(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.cancel.return_value = True
        mock_mgr.return_value = mgr
        resp = client.post("/models/download/gpt2/cancel")
        assert resp.json()["message"] == "cancelled"

    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_cancel_not_found(self, mock_mgr, client):
        mgr = MagicMock()
        mgr.cancel.return_value = False
        mock_mgr.return_value = mgr
        resp = client.post("/models/download/gpt2/cancel")
        assert resp.json()["message"] == "not_found"


class TestRetryDownload:
    """POST /models/download/{model_id}/retry"""

    @patch("apps.api.server.routers.models.ModelsRouter._run_download")
    @patch("domains.infrastructure.download_manager.is_download_complete")
    @patch("domains.infrastructure.download_manager.cleanup_incomplete")
    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_retry_with_complete_cleanup(self, mock_mgr, mock_cleanup, mock_complete, mock_run, client):
        mgr = MagicMock()
        mgr.is_downloading.return_value = False
        mock_mgr.return_value = mgr
        mock_complete.return_value = True
        resp = client.post("/models/download/gpt2/retry")
        assert resp.json()["message"] == "started"
        mock_cleanup.assert_called_once_with("gpt2")

    @patch("domains.infrastructure.download_manager.is_download_complete")
    @patch("domains.infrastructure.download_manager.get_download_manager")
    def test_already_downloading(self, mock_mgr, mock_complete, client):
        mgr = MagicMock()
        mgr.is_downloading.return_value = True
        mock_mgr.return_value = mgr
        mock_complete.return_value = False
        resp = client.post("/models/download/gpt2/retry")
        assert resp.json()["message"] == "already_downloading"


class TestVisualLoad:
    """POST /models/visual-load"""

    def test_returns_400_without_args(self, client):
        resp = client.post("/models/visual-load")
        assert resp.status_code == 400

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_loads_from_model_dir(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model_path.return_value = {"status": "loaded", "model_id": "vision"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/models/visual-load?model_dir=/tmp/vision")
        assert resp.status_code == 200
        ctrl.load_model_path.assert_called_once_with("/tmp/vision")

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_loads_from_model_id(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.load_model.return_value = {"status": "loaded", "model_id": "ans"}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/models/visual-load?model_id=Qwen/Qwen2.5-0.5B-Instruct")
        assert resp.status_code == 200
        ctrl.load_model.assert_called_once()


class TestQuantize:
    """POST /models/quantize — validation and guard paths"""

    def test_rejects_bits_not_4_or_8(self, client):
        resp = client.post("/models/quantize", json={"bits": 3})
        assert resp.status_code == 400
        assert "bits must be 4 or 8" in resp.json()["detail"]

    def test_rejects_invalid_mode(self, client):
        resp = client.post("/models/quantize", json={"bits": 8, "mode": "gaussian"})
        assert resp.status_code == 400
        assert "mode must be symmetric or asymmetric" in resp.json()["detail"]

    @patch("domains.models.provider.get_provider")
    def test_requires_loaded_model(self, mock_provider, client):
        mock_provider.return_value = None
        resp = client.post("/models/quantize", json={"bits": 8, "mode": "symmetric"})
        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]


class TestDequantize:
    """POST /models/dequantize"""

    @patch("domains.models.provider.get_provider")
    def test_requires_loaded_model(self, mock_provider, client):
        mock_provider.return_value = None
        resp = client.post("/models/dequantize")
        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]


class TestPrecision:
    """POST /models/precision"""

    @patch("domains.infrastructure.quantization.Quantine.suggest_format")
    @patch("domains.slolib.gpu.get_accelerator")
    def test_cpu_path_uses_suggestion(self, mock_acc, mock_suggest, client):
        acc = MagicMock()
        acc.name = "cpu"
        acc.device_type = "cpu"
        mock_acc.return_value = acc
        mock_suggest.return_value = {"format": "fp32", "bits": 32, "reason": "fastest", "benchmark": {}}
        resp = client.post("/models/precision", json={"mode": "auto"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["precision"] == "fp32"
        assert data["accelerator"] == "cpu"


class TestCatalog:
    """GET /models/catalog & /models/catalog/stats"""

    @patch("domains.infrastructure.model_catalog.get_model_catalog")
    def test_list_catalog(self, mock_catalog, client):
        cat = MagicMock()
        cat.list_all.return_value = [{"model_id": "gpt2"}]
        mock_catalog.return_value = cat
        resp = client.get("/models/catalog")
        assert resp.status_code == 200
        assert resp.json()["data"] == [{"model_id": "gpt2"}]

    @patch("domains.infrastructure.model_catalog.get_model_catalog")
    def test_catalog_stats(self, mock_catalog, client):
        cat = MagicMock()
        cat.stats.return_value = {"count": 1}
        mock_catalog.return_value = cat
        resp = client.get("/models/catalog/stats")
        assert resp.status_code == 200
        assert resp.json()["data"] == {"count": 1}


class TestConversionStatus:
    """GET /models/conversion-status"""

    @patch("domains.infrastructure.conversion_tracker.get_tracker")
    def test_no_model_id_returns_active(self, mock_tracker, client):
        tracker = MagicMock()
        tracker.get_active.return_value = []
        mock_tracker.return_value = tracker
        resp = client.get("/models/conversion-status")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @patch("domains.infrastructure.conversion_tracker.get_tracker")
    def test_model_id_returns_idle_when_missing(self, mock_tracker, client):
        tracker = MagicMock()
        tracker.get.return_value = None
        mock_tracker.return_value = tracker
        resp = client.get("/models/conversion-status?model_id=gpt2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["stage"] == "idle"


class TestProcessGuard:
    """GET/POST /models/process-guard"""

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_get_status(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.get_process_guard_status.return_value = {"enabled": True, "active": False}
        mock_get_ctrl.return_value = ctrl
        resp = client.get("/models/process-guard")
        assert resp.status_code == 200
        assert resp.json()["data"]["enabled"] is True

    @patch("apps.api.server.routers.models.get_models_controller")
    def test_set_enabled(self, mock_get_ctrl, client):
        ctrl = MagicMock()
        ctrl.set_process_guard_enabled.return_value = {"enabled": True}
        mock_get_ctrl.return_value = ctrl
        resp = client.post("/models/process-guard", json={"enabled": True})
        assert resp.status_code == 200
        ctrl.set_process_guard_enabled.assert_called_once_with(True)

    def test_rejects_non_boolean(self, client):
        resp = client.post("/models/process-guard", json={"enabled": "yes"})
        assert resp.status_code == 422
        assert "must be a boolean" in resp.json()["detail"]


class TestCacheUsage:
    """GET /models/cache-usage"""

    @patch("apps.api.server.routers.models._hf_cache_dir")
    def test_returns_cache_info(self, mock_cache_dir, client):
        mock_cache_dir.exists.return_value = False
        resp = client.get("/models/cache-usage")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_bytes" in data

    def test_counts_real_model_dirs(self, tmp_path, client):
        from apps.api.server.routers import models as models_mod
        blobs = tmp_path / "models--gpt2" / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "w1.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "not-a-model").mkdir()
        with patch.object(models_mod, "_hf_cache_dir", tmp_path):
            resp = client.get("/models/cache-usage")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_bytes"] == 100
        assert data["model_count"] == 1
