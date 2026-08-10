"""Tests for the auto_train API router checkpoint endpoints (routers/auto_train.py).

Covers: list_checkpoints, delete_checkpoint, load_checkpoint.
"""
from __future__ import annotations

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
