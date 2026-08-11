"""Tests for the auto_train API router checkpoint endpoints (routers/auto_train.py).

Covers: list_checkpoints, delete_checkpoint, load_checkpoint, checkpoint_info, download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.auto_train import AutoTrainRouter  # noqa: E402


def _app(ar: AutoTrainRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(ar.router)
    return app


class TestListCheckpoints:
    def test_empty(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        ar.LORA_DIR = tmp_path / "lora"
        ar.LORA_DIR.mkdir()
        client = TestClient(_app(ar))
        resp = client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_returns_success_status(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        ar.LORA_DIR = tmp_path / "lora"
        ar.LORA_DIR.mkdir()
        client = TestClient(_app(ar))
        resp = client.get("/auto-train/checkpoints")
        assert resp.json()["status"] == "success"


class TestDeleteCheckpoint:
    def test_not_found(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        client = TestClient(_app(ar))
        resp = client.delete("/auto-train/checkpoints/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["message"] == "not_found"

    def test_delete_existing(self, tmp_path):
        ckpt = tmp_path / "test.soul"
        ckpt.write_bytes(b"\x00" * 100)
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        client = TestClient(_app(ar))
        resp = client.delete("/auto-train/checkpoints/test.soul")
        assert resp.status_code == 200
        assert resp.json()["message"] == "deleted"
        assert not ckpt.exists()


class TestCheckpointInfo:
    def test_info_not_found(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        client = TestClient(_app(ar))
        resp = client.get("/auto-train/checkpoints/missing/info")
        assert resp.status_code in (200, 404)
        body = resp.json()
        assert body.get("message") == "Checkpoint not found" or body.get("detail") == "Checkpoint not found"


class TestLoadCheckpoint:
    def test_load_not_found(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        client = TestClient(_app(ar))
        resp = client.post("/auto-train/checkpoints/missing/load")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "not found" in body["message"].lower()

    def test_load_invalid_soul(self, tmp_path):
        ckpt = tmp_path / "bad.soul"
        ckpt.write_bytes(b"\x00" * 100)
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        client = TestClient(_app(ar))
        resp = client.post("/auto-train/checkpoints/bad/load")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert "data" in body
        assert body["data"]["name"] == "bad.soul"


class TestDownloadCheckpoint:
    def test_download_not_found(self, tmp_path):
        ar = AutoTrainRouter()
        ar.CHECKPOINTS_DIR = tmp_path
        ar.LORA_DIR = tmp_path / "lora"
        ar.LORA_DIR.mkdir()
        client = TestClient(_app(ar))
        resp = client.get("/auto-train/checkpoints/missing/download")
        assert resp.status_code in (200, 404)
