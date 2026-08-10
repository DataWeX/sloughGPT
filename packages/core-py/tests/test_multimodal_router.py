"""Tests for the multimodal API router (routers/multimodal.py).

Covers: status, list_checkpoints, reset, DPO state, video training state.
MultimodalManager is mocked.
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
from routers.multimodal import MultimodalRouter  # noqa: E402


def _mock_manager() -> MagicMock:
    mgr = MagicMock()
    mgr._initialized = True
    mgr._learning_count = 0
    mgr._caption_history = []
    mgr._accuracy_history = []
    mgr._replay_buffer = MagicMock()
    mgr._replay_buffer.size = 0
    mgr._multimodal_engine = MagicMock()
    mgr._multimodal_engine._trained = False
    mgr._multimodal_engine.text = MagicMock()
    mgr._multimodal_engine.text.vocab_size = 0
    caps = MagicMock()
    caps.speech_to_text = False
    caps.image_caption = True
    caps.speech_model = None
    caps.vision_model = "slonet"
    mgr.capabilities = caps
    return mgr


def _app(mr: MultimodalRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(mr.router)
    return app


class TestStatus:
    @patch("routers.multimodal.get_multimodal_manager")
    def test_status(self, mock_get):
        mock_get.return_value = _mock_manager()
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "engine" in data
        assert "learning" in data
        assert "batch" in data
        assert "dpo" in data
        assert "video" in data

    @patch("routers.multimodal.get_multimodal_manager")
    def test_status_engine_fields(self, mock_get):
        mock_get.return_value = _mock_manager()
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/status")
        engine = resp.json()["data"]["engine"]
        assert engine["speech_to_text"] is False
        assert engine["image_caption"] is True
        assert engine["vision_model"] == "slonet"


class TestListCheckpoints:
    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_list_empty(self, mock_list):
        mock_list.return_value = []
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/checkpoints")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_list_with_checkpoints(self, mock_list):
        mock_list.return_value = [{"name": "ckpt-1", "path": "/tmp/ckpt-1.pt"}]
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/checkpoints")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "ckpt-1"


class TestReset:
    @patch("routers.multimodal.get_multimodal_manager")
    def test_reset(self, mock_get):
        mgr = _mock_manager()
        mock_get.return_value = mgr
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.post("/multimodal/reset")
        assert resp.status_code == 200
        assert mgr._learning_count == 0
        assert mgr._caption_history == []
        assert mgr._accuracy_history == []


class TestDPOState:
    @patch("routers.multimodal.get_multimodal_manager")
    def test_dpo_initial_state(self, mock_get):
        mock_get.return_value = _mock_manager()
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/status")
        dpo = resp.json()["data"]["dpo"]
        assert dpo["status"] == "idle"
        assert dpo["accepted_count"] == 0
        assert dpo["rejected_count"] == 0


class TestVideoTrainingState:
    @patch("routers.multimodal.get_multimodal_manager")
    def test_video_initial_state(self, mock_get):
        mock_get.return_value = _mock_manager()
        mr = MultimodalRouter()
        client = TestClient(_app(mr))
        resp = client.get("/multimodal/status")
        video = resp.json()["data"]["video"]
        assert video["status"] == "idle"
        assert video["current_epoch"] == 0
        assert video["current_step"] == 0
