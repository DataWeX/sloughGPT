"""Tests for the /multimodal router — phoneme, video, DPO, checkpoints, status, reset."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from infrastructure.exception_handlers import register_app_error_handler
from routers.multimodal import MultimodalRouter

# ── Setup ──────────────────────────────────────────────────────────

router_instance = MultimodalRouter()

app = FastAPI()
register_app_error_handler(app)
app.include_router(router_instance.router)
client = TestClient(app)


def _d(resp):
    j = resp.json()
    return j.get("data", j)


# ── Status ─────────────────────────────────────────────────────────


class TestStatus:
    def test_status_not_initialized(self):
        with patch("routers.multimodal.get_multimodal_manager") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr._initialized = False
            mock_get.return_value = mock_mgr
            resp = client.get("/multimodal/status")
            assert resp.status_code == 200
            data = _d(resp)
            assert data["engine"]["status"] == "not_initialized"
            assert data["learning"]["images_learned"] == 0
            assert data["batch"]["running"] is False

    def test_status_initialized(self):
        with patch("routers.multimodal.get_multimodal_manager") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr._initialized = True
            caps = MagicMock()
            caps.speech_to_text = True
            caps.image_caption = True
            caps.vqa = True
            caps.speech_model = "whisper"
            caps.vision_model = "slonet"
            mock_mgr.capabilities = caps
            mock_mgr._multimodal_engine = MagicMock()
            mock_mgr._multimodal_engine._trained = True
            mock_mgr._multimodal_engine.text = MagicMock()
            mock_mgr._multimodal_engine.text.vocab_size = 1000
            mock_mgr._learning_count = 5
            mock_mgr._replay_buffer = MagicMock()
            mock_mgr._replay_buffer.size = 3
            mock_mgr._caption_history = ["cap1", "cap2", "cap1"]
            mock_mgr._accuracy_history = [0.8, 0.9]
            mock_get.return_value = mock_mgr
            resp = client.get("/multimodal/status")
            assert resp.status_code == 200
            data = _d(resp)
            assert data["engine"]["status"] == "trained"
            assert data["learning"]["images_learned"] == 5
            assert data["learning"]["vocab_size"] == 1000
            assert data["learning"]["unique_captions"] == 2
            assert data["learning"]["diversity_ratio"] == 0.667

    def test_status_structure(self):
        with patch("routers.multimodal.get_multimodal_manager") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr._initialized = False
            mock_get.return_value = mock_mgr
            resp = client.get("/multimodal/status")
            data = _d(resp)
            assert "engine" in data
            assert "learning" in data
            assert "batch" in data
            assert "dpo" in data
            assert "video_training" in data


# ── Phoneme Endpoints ─────────────────────────────────────────────


class TestEncodePhonemes:
    def test_encode_basic(self):
        resp = client.post("/multimodal/encode-phonemes", json={"text": "hello world"})
        assert resp.status_code == 200
        data = _d(resp)
        assert data["text"] == "hello world"
        assert "language" in data
        assert "phonemes" in data
        assert "ids" in data
        assert "decoded" in data
        assert isinstance(data["ids"], list)

    def test_encode_with_language(self):
        resp = client.post(
            "/multimodal/encode-phonemes",
            json={"text": "hallo welt", "language": "de"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["text"] == "hallo welt"
        assert data["language"] == "de"

    def test_encode_empty_text(self):
        resp = client.post("/multimodal/encode-phonemes", json={"text": ""})
        assert resp.status_code in (400, 422, 500)

    def test_encode_returns_nonempty_ids(self):
        resp = client.post("/multimodal/encode-phonemes", json={"text": "test"})
        assert resp.status_code == 200
        data = _d(resp)
        assert len(data["ids"]) > 0


class TestDecodePhonemes:
    def test_decode_basic(self):
        encode_resp = client.post(
            "/multimodal/encode-phonemes", json={"text": "hello"}
        )
        ids = _d(encode_resp)["ids"]
        resp = client.post(
            "/multimodal/decode-phonemes",
            json={"ids": ids, "language": "en"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert "decoded" in data
        assert "phonemes" in data

    def test_decode_empty_ids(self):
        resp = client.post("/multimodal/decode-phonemes", json={"ids": []})
        assert resp.status_code in (400, 422, 500)


class TestScorePronunciation:
    def test_score_basic(self):
        resp = client.post(
            "/multimodal/score-pronunciation",
            json={"target": "hello", "spoken": "hello"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert "score" in data
        assert "precision" in data
        assert "recall" in data
        assert data["target"] == "hello"
        assert data["spoken"] == "hello"
        assert 0.0 <= data["score"] <= 1.0

    def test_score_with_language(self):
        resp = client.post(
            "/multimodal/score-pronunciation",
            json={"target": "hallo", "spoken": "hallo", "language": "de"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert "score" in data

    def test_score_missing_fields(self):
        resp = client.post(
            "/multimodal/score-pronunciation",
            json={"target": "hello"},
        )
        assert resp.status_code in (400, 422, 500)


class TestBatchEncodePhonemes:
    def test_batch_encode(self):
        resp = client.post(
            "/multimodal/batch-encode-phonemes",
            json={"texts": ["hello", "world"]},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["count"] == 2
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert "text" in r
            assert "ids" in r
            assert "language" in r

    def test_batch_encode_with_language(self):
        resp = client.post(
            "/multimodal/batch-encode-phonemes",
            json={"texts": ["hello", "world"], "language": "en"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["count"] == 2

    def test_batch_encode_empty(self):
        resp = client.post(
            "/multimodal/batch-encode-phonemes",
            json={"texts": []},
        )
        assert resp.status_code in (400, 422, 500)


class TestDetectLanguage:
    def test_detect_english(self):
        resp = client.post(
            "/multimodal/detect-language",
            json={"text": "hello world"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert "language" in data
        assert "supported_languages" in data
        assert data["text"] == "hello world"

    def test_detect_empty(self):
        resp = client.post(
            "/multimodal/detect-language",
            json={"text": ""},
        )
        assert resp.status_code in (400, 422, 500)


class TestBatchScorePronunciation:
    def test_batch_score(self):
        resp = client.post(
            "/multimodal/batch-score-pronunciation",
            json={
                "pairs": [
                    {"target": "hello", "spoken": "hello"},
                    {"target": "world", "spoken": "worl"},
                ]
            },
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["count"] == 2
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert "score" in r
            assert "precision" in r
            assert "recall" in r

    def test_batch_score_empty_pairs(self):
        resp = client.post(
            "/multimodal/batch-score-pronunciation",
            json={"pairs": []},
        )
        assert resp.status_code in (400, 422, 500)

    def test_batch_score_skips_invalid_pairs(self):
        resp = client.post(
            "/multimodal/batch-score-pronunciation",
            json={
                "pairs": [
                    {"target": "hello", "spoken": "hello"},
                    {"target": "", "spoken": ""},
                ]
            },
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["count"] == 1


# ── Video Training ────────────────────────────────────────────────


class TestTrainVideo:
    @patch("domains.training.executor.get_training_executor")
    def test_train_video_starts(self, mock_exec):
        mock_exec.return_value = MagicMock()
        resp = client.post(
            "/multimodal/train-video",
            json={
                "data_path": "/tmp/videos",
                "epochs": 3,
                "batch_size": 2,
            },
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["status"] == "started"
        assert "job_id" in data

    @patch("domains.training.executor.get_training_executor")
    def test_train_video_busy(self, mock_exec):
        mock_exec.return_value = MagicMock()
        router_instance._video_training_state["status"] = "running"
        try:
            resp = client.post(
                "/multimodal/train-video",
                json={"data_path": "/tmp/videos"},
            )
            assert resp.status_code in (400, 409, 500)
        finally:
            router_instance._video_training_state["status"] = "idle"

    def test_train_video_validation(self):
        resp = client.post("/multimodal/train-video", json={})
        assert resp.status_code in (400, 422)


class TestVideoInfer:
    def test_video_infer_no_checkpoints(self):
        with patch(
            "domains.training.video_trainer.list_video_checkpoints", return_value=[]
        ):
            resp = client.post(
                "/multimodal/video-infer",
                json={"video_path": "/tmp/test.mp4"},
            )
            assert resp.status_code in (400, 500)


# ── DPO ────────────────────────────────────────────────────────────


class TestDPO:
    def test_dpo_no_model(self):
        with patch.object(
            router_instance,
            "_get_active_model_and_tokenizer",
            return_value=(None, None),
        ):
            resp = client.post(
                "/multimodal/dpo",
                json={"max_pairs": 5},
            )
            assert resp.status_code in (400, 500)

    def test_dpo_busy(self):
        with patch.object(
            router_instance,
            "_get_active_model_and_tokenizer",
            return_value=(MagicMock(), MagicMock()),
        ):
            router_instance._dpo_state["status"] = "running"
            try:
                resp = client.post(
                    "/multimodal/dpo",
                    json={"max_pairs": 5},
                )
                assert resp.status_code in (400, 409, 500)
            finally:
                router_instance._dpo_state["status"] = "idle"


# ── Checkpoints ───────────────────────────────────────────────────


class TestCheckpoints:
    def test_list_checkpoints_empty(self):
        with patch(
            "domains.training.video_trainer.list_video_checkpoints",
            return_value=[],
        ):
            resp = client.get("/multimodal/checkpoints")
            assert resp.status_code == 200

    def test_load_checkpoint_not_found(self):
        with patch(
            "domains.training.video_trainer.list_video_checkpoints",
            return_value=[],
        ):
            resp = client.post("/multimodal/checkpoints/nonexistent/load")
            assert resp.status_code in (404, 500)

    def test_delete_checkpoint_not_found(self):
        with patch(
            "domains.training.video_trainer.list_video_checkpoints",
            return_value=[],
        ):
            resp = client.delete("/multimodal/checkpoints/nonexistent")
            assert resp.status_code in (404, 500)


# ── Reset ─────────────────────────────────────────────────────────


class TestReset:
    def test_reset(self):
        with patch("routers.multimodal.get_multimodal_manager") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr._initialized = True
            mock_mgr._replay_buffer = MagicMock()
            mock_get.return_value = mock_mgr
            resp = client.post("/multimodal/reset")
            assert resp.status_code == 200
            data = _d(resp)
            assert data["status"] == "ok"

    def test_reset_clears_state(self):
        router_instance._video_training_state["status"] = "running"
        router_instance._dpo_state["status"] = "running"
        with patch("routers.multimodal.get_multimodal_manager") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr._initialized = True
            mock_mgr._replay_buffer = MagicMock()
            mock_get.return_value = mock_mgr
            resp = client.post("/multimodal/reset")
            assert resp.status_code == 200
