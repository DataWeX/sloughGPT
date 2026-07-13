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
        assert resp.json()["status"] == "error"

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
