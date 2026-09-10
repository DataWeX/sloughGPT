"""Tests for the OpenWebUI router — 6 endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.openwebui import OpenWebUIRouter

    router_obj = OpenWebUIRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app


# ── List datasets ────────────────────────────────────────────────────────────


class TestListDatasets:
    def test_list_datasets_returns_list(self):
        _app = _build_app()
        mock_root = MagicMock()
        mock_root.__truediv__ = lambda self, x: MagicMock(exists=lambda: False)
        with patch("routers.openwebui.find_repo_root", return_value=mock_root):
            resp = TestClient(_app).get("/openwebui/datasets")
        assert resp.status_code == 200
        assert "datasets" in resp.json()["data"]

    def test_list_datasets_scans_directories(self, tmp_path):
        _app = _build_app()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "my_dataset").mkdir()
        (data_dir / "my_dataset" / ".metadata.json").write_text('{"workspace_id": "w1"}')
        (data_dir / "some_file.db").write_text("")

        with patch("routers.openwebui.find_repo_root", return_value=tmp_path):
            resp = TestClient(_app).get("/openwebui/datasets")
        assert resp.status_code == 200
        datasets = resp.json()["data"]["datasets"]
        assert len(datasets) == 1
        assert datasets[0]["id"] == "my_dataset"
        assert datasets[0]["workspace_id"] == "w1"


# ── List checkpoints ─────────────────────────────────────────────────────────


class TestListCheckpoints:
    def test_list_checkpoints_returns_list(self):
        _app = _build_app()
        mock_root = MagicMock()
        mock_root.__truediv__ = lambda self, x: MagicMock(exists=lambda: False)
        with patch("routers.openwebui.find_repo_root", return_value=mock_root):
            resp = TestClient(_app).get("/openwebui/checkpoints")
        assert resp.status_code == 200
        assert "checkpoints" in resp.json()["data"]

    def test_list_checkpoints_finds_adapter(self, tmp_path):
        _app = _build_app()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model-v1").mkdir()
        (ckpt_dir / "model-v1" / "adapter.npz").write_bytes(b"fake")

        with patch("routers.openwebui.find_repo_root", return_value=tmp_path):
            resp = TestClient(_app).get("/openwebui/checkpoints")
        assert resp.status_code == 200
        checkpoints = resp.json()["data"]["checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["id"] == "model-v1"


# ── Reload checkpoint ────────────────────────────────────────────────────────


class TestReloadCheckpoint:
    def test_reload_checkpoint(self):
        _app = _build_app()
        resp = TestClient(_app).post("/openwebui/checkpoint/reload?checkpoint_id=ckpt-1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "reloaded"
        assert data["checkpoint_id"] == "ckpt-1"


# ── Start training ───────────────────────────────────────────────────────────


class TestStartTraining:
    def test_start_training(self):
        _app = _build_app()
        resp = TestClient(_app).post(
            "/openwebui/training/start?dataset_id=ds1&method=lora"
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "started"
        assert data["dataset_id"] == "ds1"
        assert data["method"] == "lora"

    def test_start_training_default_method(self):
        _app = _build_app()
        resp = TestClient(_app).post("/openwebui/training/start?dataset_id=ds1")
        assert resp.status_code == 200
        assert resp.json()["data"]["method"] == "finetune"


# ── Stop training ────────────────────────────────────────────────────────────


class TestStopTraining:
    def test_stop_training(self):
        _app = _build_app()
        resp = TestClient(_app).post("/openwebui/training/stop")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "stopped"


# ── Training status ──────────────────────────────────────────────────────────


class TestTrainingStatus:
    def test_training_status(self):
        _app = _build_app()
        resp = TestClient(_app).get("/openwebui/training/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "idle"
        assert data["progress"] == 0
