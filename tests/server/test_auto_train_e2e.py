"""
End-to-end smoke test for the auto-train pipeline.

Tests: start config → status → checkpoints → schema validation → cancel.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.auto_train import router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)


class TestAutoTrainStart:
    """Test POST /auto-train/start — stores config for stream."""

    def test_start_requires_data(self):
        resp = client.post("/auto-train/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Provide" in data["message"]

    def test_start_with_source_text(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "hello world " * 100,
            "epochs": 1,
            "batch_size": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["epochs"] == 1

    def test_start_config_accessible_via_status(self):
        client.post("/auto-train/start", json={
            "source_text": "test training data",
            "epochs": 3,
            "learning_rate": 0.001,
        })
        resp = client.get("/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("running") is True
        assert data["config"]["epochs"] == 3


class TestAutoTrainCheckpoints:
    """Test GET /auto-train/checkpoints."""

    def test_returns_wrapped_list(self):
        resp = client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert isinstance(data["data"], list)

    def test_each_checkpoint_has_name(self):
        resp = client.get("/auto-train/checkpoints")
        ckpts = resp.json()["data"]
        for ckpt in ckpts:
            assert "name" in ckpt
            assert "download_url" in ckpt


class TestAutoTrainSchema:
    """Test StartRequest schema validation."""

    def test_invalid_epochs(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "test",
            "epochs": -1,
        })
        assert resp.status_code == 422

    def test_invalid_learning_rate(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "test",
            "learning_rate": 0,
        })
        assert resp.status_code == 422

    def test_valid_config(self):
        resp = client.post("/auto-train/start", json={
            "source_text": "hello world",
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "soul_name": "test-soul",
            "algo": "bpe",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestAutoTrainCancel:
    """Test POST /auto-train/cancel."""

    def test_cancel_returns_ok(self):
        resp = client.post("/auto-train/cancel")
        assert resp.status_code in (200, 404, 409)


class TestAutoTrainStatus:
    """Test GET /auto-train/status."""

    def test_status_returns_config(self):
        resp = client.get("/auto-train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "running" in data
        assert "config" in data
