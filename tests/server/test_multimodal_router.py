"""
Tests for the multimodal router — status, train, batch, transcribe, generate.
"""

import io
import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.multimodal import router, _background_job, multimodal_router

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

MGR_TARGET = "apps.api.server.routers.multimodal.get_multimodal_manager"
ROUTER = "apps.api.server.routers.multimodal.MultimodalRouter"


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
    result.tags = ["outdoor", "animal"]
    mgr.recognize_speech.return_value = result
    mgr.caption_image.return_value = result
    return mgr


def _get_data(resp):
    """Unwrap {status, data} envelope if present."""
    body = resp.json()
    if "data" in body and "status" in body:
        return body["data"]
    return body


def _real_png_bytes():
    """Return valid 1x1 PNG bytes via PIL (mock file content must decode)."""
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (4, 4), (128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()


class TestStatus:
    """GET /multimodal/status — consolidated endpoint"""

    @patch(MGR_TARGET)
    def test_get_status(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = _get_data(resp)
        assert "engine" in data
        assert "learning" in data
        assert "batch" in data

    @patch(MGR_TARGET)
    def test_status_engine_capabilities(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        data = _get_data(resp)
        engine = data["engine"]
        assert engine["speech_to_text"] is True
        assert engine["image_caption"] is True
        assert engine["speech_model"] == "whisper"
        assert engine["vision_model"] == "slonet"
        assert engine["status"] == "trained"

    @patch(MGR_TARGET)
    def test_status_learning_progress(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        data = _get_data(resp)
        learning = data["learning"]
        assert learning["images_learned"] == 5
        assert learning["trained"] is True
        assert learning["vocab_size"] == 3
        assert learning["replay_buffer_size"] == 42
        assert len(learning["caption_history"]) == 3
        assert learning["unique_captions"] == 3
        assert learning["diversity_ratio"] == 1.0
        assert learning["mean_accuracy"] == 0.6

    @patch(MGR_TARGET)
    def test_status_training_report(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        data = _get_data(resp)
        learning = data["learning"]
        assert learning["images_learned"] == 5
        assert len(learning["caption_history"]) == 3
        assert learning["unique_captions"] == 3
        assert learning["diversity_ratio"] == 1.0
        assert learning["mean_accuracy"] == 0.6

    def test_training_status_idle(self):
        _background_job["running"] = False
        _background_job["job_id"] = None
        _background_job["total"] = 0
        _background_job["completed"] = 0
        _background_job["errors"] = 0
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = _get_data(resp)
        assert data["batch"]["running"] is False
        assert data["batch"]["progress_pct"] == 0

    @patch(MGR_TARGET)
    def test_generation_status(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.get("/multimodal/status")
        assert resp.status_code == 200
        data = _get_data(resp)
        assert "engine" in data
        assert "learning" in data


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
        data = _get_data(resp)
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
        img_bytes = _real_png_bytes()
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


class TestTrainBatch:
    """POST /multimodal/train-batch"""

    def _reset_job(self):
        _background_job.update(
            job_id=None, running=False, total=0, completed=0, errors=0,
            current_caption="", current_image="", started_at=None, finished_at=None,
        )

    @patch(MGR_TARGET)
    def test_no_images_returns_400(self, mock_get):
        mock_get.return_value = _mock_manager()
        self._reset_job()
        resp = client.post("/multimodal/train-batch")
        assert resp.status_code == 400
        assert "No images provided" in resp.json()["detail"]

    @patch(MGR_TARGET)
    def test_returns_409_when_running(self, mock_get):
        mock_get.return_value = _mock_manager()
        _background_job["running"] = True
        try:
            resp = client.post(
                "/multimodal/train-batch",
                files={"files": ("t.png", _real_png_bytes(), "image/png")},
            )
            assert resp.status_code == 409
        finally:
            self._reset_job()

    @patch(MGR_TARGET)
    def test_starts_batch_from_uploads(self, mock_get):
        mock_get.return_value = _mock_manager()
        self._reset_job()
        resp = client.post(
            "/multimodal/train-batch",
            files={"files": ("one.png", _real_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "started"
        assert data["total_images"] == 1
        self._reset_job()

    @patch(MGR_TARGET)
    def test_dataset_path_missing_returns_400(self, mock_get):
        mock_get.return_value = _mock_manager()
        self._reset_job()
        resp = client.post("/multimodal/train-batch", data={"dataset_path": "/nonexistent/dir"})
        assert resp.status_code == 400

    @patch(MGR_TARGET)
    def test_dataset_path_no_images_returns_400(self, mock_get, tmp_path):
        mock_get.return_value = _mock_manager()
        self._reset_job()
        d = tmp_path / "empty"
        d.mkdir()
        resp = client.post("/multimodal/train-batch", data={"dataset_path": str(d)})
        assert resp.status_code == 400
        assert "No images found" in resp.json()["detail"]


class TestDPO:
    """POST /multimodal/dpo"""

    def test_requires_loaded_model(self):
        resp = client.post("/multimodal/dpo", json={"max_pairs": 4})
        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]
        multimodal_router._dpo_state["status"] = "idle"

    def test_dpo_run(self):
        import types
        import sys
        import time

        class _FakeTrainer:
            def __init__(self, model, tokenizer, learning_rate):
                self._lr = learning_rate

            def train(self, max_pairs):
                return {"status": "accepted", "steps": 5, "avg_loss": 0.4,
                        "ppl_before": 10.0, "ppl_after": 8.0, "ppl_delta_pct": -20.0,
                        "pairs_trained": max_pairs}

        fake_mod = types.ModuleType("domains.feedback.hf_dpo")
        fake_mod.HFDPOTrainer = _FakeTrainer
        with patch.dict(sys.modules, {"domains.feedback.hf_dpo": fake_mod}):
            with patch("apps.api.server.routers.multimodal.MultimodalRouter._get_active_model_and_tokenizer",
                       return_value=(object(), object())):
                multimodal_router._dpo_state["status"] = "idle"
                resp = client.post("/multimodal/dpo", json={"max_pairs": 4})
                assert resp.status_code == 200
                data = resp.json()["data"]
                assert data["status"] == "accepted"
                assert data["pairs_trained"] == 4
        multimodal_router._dpo_state["status"] = "idle"

    def test_dpo_rejects_invalid_pairs(self):
        resp = client.post("/multimodal/dpo", json={"max_pairs": 999})
        assert resp.status_code == 422


class TestTrainVideo:
    """POST /multimodal/train-video"""

    def _reset_video_job(self):
        multimodal_router._video_training_state.update(
            status="idle", job_id=None, current_epoch=0, current_step=0,
            total_steps=0, current_loss=None, result=None, error=None,
        )

    def test_returns_409_when_running(self):
        multimodal_router._video_training_state["status"] = "running"
        try:
            resp = client.post("/multimodal/train-video", json={
                "data_path": "/tmp/videos", "epochs": 3, "batch_size": 2,
            })
            assert resp.status_code == 409
            assert "already in progress" in resp.json()["detail"].lower()
        finally:
            self._reset_video_job()

    @patch("domains.training.executor.get_training_executor")
    def test_starts_training_job(self, mock_get_exec):
        mock_get_exec.return_value = MagicMock()
        self._reset_video_job()
        try:
            resp = client.post("/multimodal/train-video", json={
                "data_path": "/tmp/videos", "epochs": 3, "batch_size": 2,
                "learning_rate": 0.0003, "output_dir": "models/video-training",
            })
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["status"] == "started"
            assert data["data_path"] == "/tmp/videos"
            assert data["job_id"].startswith("video_")
            mock_get_exec.return_value.submit.assert_called_once()
        finally:
            self._reset_video_job()

    @patch("domains.training.executor.get_training_executor")
    def test_rejects_missing_data_path(self, mock_get_exec):
        mock_get_exec.return_value = MagicMock()
        self._reset_video_job()
        resp = client.post("/multimodal/train-video", json={})
        assert resp.status_code == 422
        self._reset_video_job()


class TestVideoInfer:
    """POST /multimodal/video-infer"""

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_no_checkpoint_returns_400(self, mock_list):
        mock_list.return_value = []
        resp = client.post("/multimodal/video-infer", json={"video_path": "/tmp/a.mp4"})
        assert resp.status_code == 400
        assert "no trained video model" in resp.json()["detail"].lower()

    @patch("domains.training.video_trainer.VideoCaptionTrainer")
    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_generates_caption_from_latest_checkpoint(self, mock_list, mock_trainer_cls):
        mock_list.return_value = [{"name": "ck1", "path": "/tmp/ck1.slnc"}]
        trainer = mock_trainer_cls.return_value
        trainer.generate.return_value = "a dog runs"
        resp = client.post("/multimodal/video-infer", json={"video_path": "/tmp/a.mp4"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["text"] == "a dog runs"
        assert data["checkpoint"] == "ck1"
        assert "elapsed_ms" in data
        trainer.load_checkpoint.assert_called_once_with("/tmp/ck1.slnc")
        trainer.generate.assert_called_once_with(video_path="/tmp/a.mp4", max_len=50, temperature=0.8)


class TestAnalyze:
    """POST /multimodal/analyze"""

    @patch(MGR_TARGET)
    def test_analyze_image(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/analyze",
            files={"file": ("t.png", _real_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["caption"] == "hello world"
        assert data["tags"] == ["outdoor", "animal"]

    @patch(MGR_TARGET)
    def test_analyze_rejects_non_image(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = client.post(
            "/multimodal/analyze",
            files={"file": ("t.txt", b"text", "text/plain")},
        )
        assert resp.status_code == 400


class TestSynthesizeSpeech:
    """POST /multimodal/synthesize-speech"""

    @patch("domains.multimodal.tts.TTSEngine")
    def test_synthesizes_waveform(self, mock_tts_cls):
        import numpy as np
        tts = mock_tts_cls.return_value
        tts.text_to_waveform.return_value = np.zeros(1600)
        tts.sample_rate = 16000
        resp = client.post("/multimodal/synthesize-speech", data={"text": "hi"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "audio" in data
        assert "wav" in data["audio"]


class TestAnalyzePdf:
    """POST /multimodal/pdf/upload"""

    def _fake_pdf_bytes(self):
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor")
    def test_analyzes_pdf_text_extract(self, mock_processor_cls):
        processor = mock_processor_cls.return_value
        processor.analyze.return_value = "extracted summary"
        processor._get_vlm.return_value = None
        resp = client.post(
            "/multimodal/pdf/upload",
            files={"file": ("doc.pdf", self._fake_pdf_bytes(), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["analysis"] == "extracted summary"
        assert data["filename"] == "doc.pdf"
        assert data["method"] == "text_extract"
        processor.analyze.assert_called_once()

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor")
    def test_analyzes_pdf_with_vlm(self, mock_processor_cls):
        processor = mock_processor_cls.return_value
        processor.analyze.return_value = "vlm summary"
        processor._get_vlm.return_value = object()
        resp = client.post(
            "/multimodal/pdf/upload",
            files={"file": ("doc.pdf", self._fake_pdf_bytes(), "application/pdf")},
            data={"question": "What is this?"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["method"] == "vlm"

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor")
    def test_analyzes_pdf_per_page(self, mock_processor_cls):
        processor = mock_processor_cls.return_value
        processor.analyze_pages.return_value = [
            {"page": 1, "text": "intro"},
            {"page": 2, "text": "body"},
        ]
        processor._get_vlm.return_value = None
        resp = client.post(
            "/multimodal/pdf/upload",
            files={"file": ("doc.pdf", self._fake_pdf_bytes(), "application/pdf")},
            data={"per_page": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "--- Page 1 ---" in data["analysis"]
        assert "--- Page 2 ---" in data["analysis"]
        assert "body" in data["analysis"]
        processor.analyze_pages.assert_called_once()


class TestProcessVideo:
    """POST /multimodal/process-video"""

    def _make_engine(self):
        engine = MagicMock()
        engine.vision = MagicMock()
        caption = MagicMock()
        caption.text = "a car driving"
        engine.generate.return_value = caption
        return engine

    @patch(MGR_TARGET)
    @patch("domains.multimodal.video.VideoProcessor")
    def test_processes_video(self, mock_processor_cls, mock_get):
        import numpy as np
        mock_get.return_value = _mock_manager()
        mock_get.return_value._multimodal_engine = self._make_engine()
        processor = mock_processor_cls.return_value
        processor.extract_frames.return_value = [np.zeros((224, 224, 3), dtype=np.float32)] * 2
        emb = MagicMock()
        emb.data.shape = (2, 512)
        processor.encode_video.return_value = emb
        resp = client.post(
            "/multimodal/process-video",
            files={"file": ("clip.mp4", b"fake-mp4", "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        assert data["caption"] == "a car driving"
        assert data["num_frames"] == 2

    @patch(MGR_TARGET)
    @patch("domains.multimodal.video.VideoProcessor")
    def test_returns_500_when_engine_missing(self, mock_processor_cls, mock_get):
        import numpy as np
        mgr = _mock_manager()
        mgr._multimodal_engine = None
        mock_get.return_value = mgr
        processor = mock_processor_cls.return_value
        processor.extract_frames.return_value = [np.zeros((224, 224, 3), dtype=np.float32)]
        resp = client.post(
            "/multimodal/process-video",
            files={"file": ("clip.mp4", b"fake-mp4", "video/mp4")},
        )
        assert resp.status_code == 500


class TestVisualDataset:
    """POST /multimodal/visual-dataset"""

    def test_403_outside_allowed_path(self, tmp_path):
        out = tmp_path / "outside"
        out.mkdir()
        resp = client.post("/multimodal/visual-dataset", json={
            "name": "ds", "image_dir": str(out), "auto_caption": False,
        })
        assert resp.status_code == 403

    def test_400_when_dir_missing(self, tmp_path):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        missing = repo_root / "data" / "mm-vision-missing-xyz"
        resp = client.post("/multimodal/visual-dataset", json={
            "name": "ds", "image_dir": str(missing), "auto_caption": False,
        })
        assert resp.status_code == 400

    @patch(MGR_TARGET)
    def test_creates_dataset(self, mock_get, tmp_path):
        from pathlib import Path
        import shutil
        mock_get.return_value = _mock_manager()
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"
        img_dir = data_dir / "mm-vision-test-xyz"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / "a.png"
        img_path.write_bytes(_real_png_bytes())
        ds_path = repo_root / "datasets" / "mm-vision-test-xyz.jsonl"
        try:
            resp = client.post("/multimodal/visual-dataset", json={
                "name": "mm-vision-test-xyz", "image_dir": str(img_dir),
                "auto_caption": False,
            })
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["entries"] == 1
            assert data["status"] == "created"
        finally:
            if ds_path.exists():
                os.remove(ds_path)
            shutil.rmtree(img_dir, ignore_errors=True)


class TestCheckpoints:
    """GET/DELETE /multimodal/checkpoints"""

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_list_checkpoints_empty(self, mock_list):
        mock_list.return_value = []
        resp = client.get("/multimodal/checkpoints")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_load_missing_returns_404(self, mock_list):
        mock_list.return_value = []
        resp = client.post("/multimodal/checkpoints/nope/load")
        assert resp.status_code == 404

    @patch("domains.training.video_trainer.VideoCaptionTrainer")
    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_load_existing(self, mock_list, mock_trainer):
        mock_list.return_value = [{"name": "ck1", "path": "/tmp/ck1.slnc"}]
        resp = client.post("/multimodal/checkpoints/ck1/load")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "loaded"

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_delete_missing_returns_404(self, mock_list):
        mock_list.return_value = []
        resp = client.delete("/multimodal/checkpoints/nope")
        assert resp.status_code == 404

    @patch("domains.training.video_trainer.list_video_checkpoints")
    def test_delete_existing(self, mock_list, tmp_path):
        ck = tmp_path / "ck1.slnc"
        ck.write_bytes(b"data")
        mock_list.return_value = [{"name": "ck1", "path": str(ck)}]
        resp = client.delete("/multimodal/checkpoints/ck1")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"


class TestReset:
    @patch(MGR_TARGET)
    def test_reset(self, mock_get):
        mgr = _mock_manager()
        mock_get.return_value = mgr
        resp = client.post("/multimodal/reset")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "ok"
