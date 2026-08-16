"""
End-to-end smoke test for the training pipeline.

Tests key training endpoints through isolated router apps.
"""

import sys
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add server dir to path for router imports
_server_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server')
if os.path.isdir(_server_dir):
    sys.path.insert(0, os.path.abspath(_server_dir))

from routers.auto_train import router as auto_train_router

auto_app = FastAPI()
auto_app.include_router(auto_train_router)
auto_client = TestClient(auto_app, raise_server_exceptions=False)


class TestAutoTrainStatus:
    """Test GET /auto-train/status."""

    def test_status_returns_dict(self):
        resp = auto_client.get("/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestAutoTrainCheckpoints:
    """Test GET /auto-train/checkpoints."""

    def test_returns_wrapped_list(self):
        resp = auto_client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert isinstance(data["data"], list)

    def test_each_has_required_fields(self):
        resp = auto_client.get("/auto-train/checkpoints")
        for ckpt in resp.json()["data"]:
            assert "name" in ckpt
            assert "download_url" in ckpt

    def test_checkpoint_has_size(self):
        resp = auto_client.get("/auto-train/checkpoints")
        for ckpt in resp.json()["data"]:
            if "size_mb" in ckpt:
                assert isinstance(ckpt["size_mb"], (int, float))


class TestAutoTrainCancel:
    """Test POST /auto-train/cancel."""

    def test_cancel_returns_ok(self):
        resp = auto_client.post("/auto-train/cancel")
        assert resp.status_code in (200, 404, 409)


class TestAutoTrainStart:
    """Test POST /auto-train/start."""

    def test_start_requires_data(self):
        resp = auto_client.post("/auto-train/start", json={})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_start_with_text(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "hello world " * 50,
            "epochs": 1,
            "batch_size": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_invalid_epochs(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test",
            "epochs": -1,
        })
        assert resp.status_code == 422

    def test_start_with_dataset_id(self):
        resp = auto_client.post("/auto-train/start", json={
            "dataset_id": "nonexistent",
            "epochs": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_start_with_checkpoint_name(self):
        resp = auto_client.post("/auto-train/start", json={
            "checkpoint_name": "nonexistent",
            "epochs": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_start_returns_config(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test data",
            "epochs": 2,
            "learning_rate": 0.001,
            "batch_size": 8,
            "soul_name": "test-soul",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert data["config"]["epochs"] == 2
        assert data["config"]["learning_rate"] == 0.001


class TestAutoTrainLog:
    """Test GET /auto-train/log."""

    def test_log_returns_dict(self):
        resp = auto_client.get("/auto-train/log")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "lines" in data
        assert "total" in data


class TestCheckpointDelete:
    """Test DELETE /auto-train/checkpoints/{name}."""

    def test_delete_nonexistent(self):
        resp = auto_client.delete("/auto-train/checkpoints/nonexistent_xyz")
        assert resp.status_code in (200, 404)


class TestCheckpointDownload:
    """Test GET /auto-train/checkpoints/{name}/download."""

    def test_download_nonexistent(self):
        resp = auto_client.get("/auto-train/checkpoints/nonexistent_xyz/download")
        assert resp.status_code in (200, 404, 410)


class TestCheckpointLoad:
    """Test POST /auto-train/checkpoints/{name}/load."""

    def test_load_nonexistent(self):
        resp = auto_client.post("/auto-train/checkpoints/nonexistent_xyz/load")
        assert resp.status_code in (200, 404)


class TestAutoTrainStop:
    """Test POST /auto-train/stop."""

    def test_stop_returns_ok(self):
        resp = auto_client.post("/auto-train/stop")
        assert resp.status_code in (200, 404, 409)


class TestAutoTrainPause:
    """Test POST /auto-train/pause."""

    def test_pause_returns_ok(self):
        resp = auto_client.post("/auto-train/pause")
        assert resp.status_code in (200, 404, 409)


class TestAutoTrainStartValidation:
    """Deterministic request validation — no training starts."""

    def test_epochs_zero_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 0,
        })
        assert resp.status_code == 422

    def test_epochs_over_max_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1001,
        })
        assert resp.status_code == 422

    def test_batch_size_zero_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "batch_size": 0,
        })
        assert resp.status_code == 422

    def test_batch_size_over_max_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "batch_size": 2048,
        })
        assert resp.status_code == 422

    def test_temperature_too_low_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "temperature": 0.05,
        })
        assert resp.status_code == 422

    def test_temperature_too_high_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "temperature": 2.5,
        })
        assert resp.status_code == 422

    def test_learning_rate_zero_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "learning_rate": 0.0,
        })
        assert resp.status_code == 422

    def test_learning_rate_over_max_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "learning_rate": 2.0,
        })
        assert resp.status_code == 422

    def test_early_stopping_over_max_rejected(self):
        resp = auto_client.post("/auto-train/start", json={
            "source_text": "test", "epochs": 1, "early_stopping_patience": 101,
        })
        assert resp.status_code == 422


class TestAutoTrainStatusShape:
    """Deterministic shape assertions on /auto-train/status."""

    def test_has_running_flag(self):
        resp = auto_client.get("/auto-train/status")
        assert "running" in resp.json()

    def test_has_config_dict(self):
        resp = auto_client.get("/auto-train/status")
        config = resp.json().get("config")
        assert isinstance(config, dict)


class TestAutoTrainLogShape:
    """Deterministic shape assertions on /auto-train/log."""

    def test_lines_is_list(self):
        resp = auto_client.get("/auto-train/log")
        assert isinstance(resp.json()["lines"], list)

    def test_total_is_non_negative(self):
        resp = auto_client.get("/auto-train/log")
        assert resp.json()["total"] >= 0
