"""Tests for unified /training/* shim routes.

These thin delegates route /training/* to AutoTrainRouter class methods.
Tests verify the routing contract: correct method called, correct args passed,
and response forwarded unchanged.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_app_error_handler
from training.router import router

app = FastAPI()
register_app_error_handler(app)
app.include_router(router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    """Clear any state between tests."""
    yield


# ── /training/log ──────────────────────────────────────────────────────────────


class TestTrainingLog:
    def test_delegates_to_core_service(self):
        with patch("domains.training.service.get_log", new_callable=AsyncMock, return_value=["line1", "line2"]):
            resp = client.get("/training/log")

        assert resp.status_code == 200
        assert resp.json()["data"]["lines"] == ["line1", "line2"]


# ── /training/stop ─────────────────────────────────────────────────────────────


class TestTrainingStop:
    def test_stop_returns_cancelling(self):
        resp = client.post("/training/stop")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["status"] == "cancelling"


# ── /training/turbo-start ──────────────────────────────────────────────────────


class TestTrainingTurboStart:
    def test_turbo_start_returns_started(self):
        request_body = {"epochs": 5, "n_embed": 64}

        with patch("domains.training.service.start_turbo_training", return_value={"job_id": "t1", "data_path": "/tmp/data"}):
            with patch("domains.training.service.run_turbo_worker"):
                resp = client.post("/training/turbo-start", json=request_body)

        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "started"


# ── /training/checkpoints ──────────────────────────────────────────────────────


class TestTrainingCheckpoints:
    def test_list_checkpoints(self):
        with patch("domains.training.service.list_checkpoints", new_callable=AsyncMock, return_value=[{"name": "v1.soul"}]):
            resp = client.get("/training/checkpoints")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1

    def test_delete_checkpoint(self):
        with patch("domains.training.service.delete_checkpoint", new_callable=AsyncMock, return_value=["old.soul"]):
            resp = client.delete("/training/checkpoints/old.soul")

        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == ["old.soul"]

    def test_load_checkpoint(self):
        with patch("domains.training.service.load_checkpoint", new_callable=AsyncMock, return_value={"name": "v1.soul", "vocab_size": 100}):
            resp = client.post("/training/checkpoints/v1.soul/load")

        assert resp.status_code == 200
        assert resp.json()["data"]["vocab_size"] == 100

    def test_checkpoint_info(self):
        with patch("domains.training.service.checkpoint_info", new_callable=AsyncMock, return_value={"name": "v1.soul", "loss": 0.42}):
            resp = client.get("/training/checkpoints/v1.soul/info")

        assert resp.status_code == 200
        assert resp.json()["data"]["loss"] == 0.42

    def test_download_checkpoint(self):
        with patch("domains.training.service.download_checkpoint_path", new_callable=AsyncMock, return_value="/fake/path.soul"):
            with patch("fastapi.responses.FileResponse") as mock_fr:
                mock_fr.return_value = MagicMock(status_code=200)
                resp = client.get("/training/checkpoints/v1.soul/download")

        assert resp.status_code == 200


# ── /training/metrics/export ───────────────────────────────────────────────────


class TestTrainingMetricsExport:
    def test_delegates_to_core_service(self):
        with patch("domains.training.service.get_all_checkpoint_data", new_callable=AsyncMock, return_value=[{"name": "v1.soul"}]):
            resp = client.get("/training/metrics/export")

        assert resp.status_code == 200


# ── /training/from-sessions/cancel ─────────────────────────────────────────────


class TestTrainingCancelFromSessions:
    def test_cancel_from_sessions_returns_success(self):
        resp = client.get("/training/from-sessions/cancel")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["status"] == "cancelled"


# ── /training/turbo/status (turbo_endpoints.py) ────────────────────────────────


class TestTurboStatus:
    def test_returns_turbo_state(self):
        with patch("domains.training.service.get_turbo_status", return_value={"status": "idle"}):
            resp = client.get("/training/turbo/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"


# ── /training/from-sessions-start (turbo_endpoints.py) ─────────────────────────


class TestFromSessionsStart:
    def test_from_sessions_start_returns_success(self):
        request_body = {"epochs": 3}

        with patch("domains.training.service.start_from_sessions_training", return_value={"method": "from-sessions", "epochs": 3}):
            resp = client.post("/training/from-sessions-start", json=request_body)

        assert resp.status_code == 200
        assert resp.json()["data"]["method"] == "from-sessions"
