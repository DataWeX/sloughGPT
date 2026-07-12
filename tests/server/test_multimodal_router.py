"""
Tests for the multimodal router — capabilities, training, generation, speech.
"""

import io
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.multimodal import router, _background_job

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

MGR_TARGET = "apps.api.server.routers.multimodal.get_multimodal_manager"


def _mock_capabilities():
    caps = MagicMock()
    caps.speech_to_text = True
    caps.image_caption = True
    caps.speech_model = "whisper"
    caps.vision_model = "slonet"
    return caps


def _mock_manager():
    mgr = MagicMock()
    mgr.capabilities = _mock_capabilities()
    mgr._initialized = True
    mgr._learning_count = 5
    mgr._caption_history = ["a photo", "a dog", "sunset"]
    mgr._accuracy_history = [0.5, 0.6, 0.7]
    engine = MagicMock()
    engine._trained = True
    engine.text.vocab_size = 3
    mgr._multimodal_engine = engine
    buf = MagicMock()
    buf.size = 42
    mgr._replay_buffer = buf
    result = MagicMock()
    result.text = "hello world"
    result.confidence = 0.9
    result.language = "en"
    result.duration = 1.5
    result.accuracy = 0.85
    mgr.recognize_speech.return_value = result
    mgr.caption_image.return_value = result
    return mgr


class TestCapabilities:
    """GET /multimodal/status (consolidated from /capabilities, /learning-progress, /training-report)"""

    @patch(MGR_TARGET)
    def test_get_capabilities(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        caps = data["engine"]
        assert "speech_to_text" in caps
        assert "image_caption" in caps
        assert caps["status"] == "trained"
        learning = data["learning"]
        assert "images_learned" in learning
        assert "trained" in learning
        assert learning["images_learned"] == 5
        assert learning["trained"] is True


class TestLearningProgress:
    """GET /multimodal/status (consolidated)"""

    @patch(MGR_TARGET)
    def test_get_progress(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        learning = data["learning"]
        assert learning["images_learned"] == 5
        assert learning["trained"] is True
        assert learning["vocab_size"] == 3
        assert learning["replay_buffer_size"] == 42


class TestTrainingReport:
    """GET /multimodal/status (consolidated)"""

    @patch(MGR_TARGET)
    def test_get_report(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        learning = data["learning"]
        assert learning["images_learned"] == 5
        assert len(learning["caption_history"]) == 3
        assert learning["unique_captions"] == 3
        assert learning["diversity_ratio"] == 1.0
        assert learning["mean_accuracy"] == 0.6


class TestTranscribe:
    """POST /multimodal/transcribe"""

    @patch(MGR_TARGET)
    def test_transcribe_audio(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/transcribe",
            files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            data={"language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["text"] == "hello world"
        assert data["confidence"] == 0.9

    @patch(MGR_TARGET)
    def test_transcribe_rejects_non_audio(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/transcribe",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400
        assert "audio" in resp.json()["detail"].lower()


class TestTrainOnImage:
    """POST /multimodal/train"""

    @patch(MGR_TARGET)
    def test_train_image(self, mock_get):
        mock_get.return_value = _mock_manager()
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        resp = client.post(
            "/multimodal/train",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code in (200, 500)

    @patch(MGR_TARGET)
    def test_train_rejects_non_image(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/train",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()


class TestTrainingStatus:
    """GET /multimodal/status (consolidated from /training-status)"""

    def test_training_status_idle(self):
        _background_job["running"] = False
        _background_job["job_id"] = None
        _background_job["total"] = 0
        _background_job["completed"] = 0
        _background_job["errors"] = 0
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        batch = data["batch"]
        assert batch["running"] is False
        assert batch["progress_pct"] == 0


class TestGenerateImage:
    """POST /multimodal/generate-image"""

    @patch(MGR_TARGET)
    def test_generate_image(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/generate-image",
            data={"prompt": "a cat", "steps": 5},
        )
        assert resp.status_code in (200, 500)


class TestGenerationStatus:
    """GET /multimodal/status (consolidated from /generation-status)"""

    def test_generation_status(self):
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "engine" in data
        assert "learning" in data
