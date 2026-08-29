"""Multimodal Router — vision, speech, DPO, video training, batch training.

Simplified from 24 → 17 endpoints by consolidating 7 status endpoints into 1.
"""

import asyncio
import datetime
import json
import logging
import os
import time
from typing import List, Optional
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from domains.multimodal import get_multimodal_manager
from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.routers.multimodal")


# ── Schemas ────────────────────────────────────────────────────────

class DPOTriggerRequest(BaseModel):
    max_pairs: int = Field(6, ge=1, le=50)
    learning_rate: float = Field(5e-6, ge=1e-8, le=1e-3)


class VideoTrainRequest(BaseModel):
    data_path: str
    epochs: int = Field(5, ge=1, le=100)
    batch_size: int = Field(2, ge=1, le=16)
    learning_rate: float = Field(3e-4, ge=1e-6, le=1.0)
    output_dir: str = "models/video-training"


class VideoInferRequest(BaseModel):
    video_path: str
    max_len: int = Field(50, ge=10, le=200)
    temperature: float = Field(0.8, ge=0.0, le=2.0)


class VisualDatasetRequest(BaseModel):
    name: str
    image_dir: str
    caption_prompt: str = "Describe this image in detail."
    auto_caption: bool = True


# ── Router Class ──────────────────────────────────────────────────

class MultimodalRouter:

    def __init__(self):
        self.router = APIRouter(prefix="/multimodal", tags=["multimodal"])

        self._dpo_state = {
            "last_run": None,
            "status": "idle",
            "result": None,
            "accepted_count": 0,
            "rejected_count": 0,
        }
        self._dpo_lock = Lock()

        self._video_training_state = {
            "status": "idle",
            "job_id": None,
            "current_epoch": 0,
            "current_step": 0,
            "total_steps": 0,
            "current_loss": None,
            "result": None,
            "error": None,
        }
        self._video_training_lock = Lock()

        self._background_job = {
            "job_id": None,
            "running": False,
            "total": 0,
            "completed": 0,
            "errors": 0,
            "current_caption": "",
            "current_image": "",
            "started_at": None,
            "finished_at": None,
        }
        self._bg_lock = Lock()

        self._tts = None
        self._vae = None
        self._diffusion = None
        self._text_encoder = None

        self._register_routes()

    # ── Route Registration ───────────────────────────────────────

    def _register_routes(self):
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/train", self.train_on_image, methods=["POST"])
        self.router.add_api_route("/train-batch", self.train_batch, methods=["POST"])
        self.router.add_api_route("/train-video", self.train_video, methods=["POST"])
        self.router.add_api_route("/video-infer", self.video_infer, methods=["POST"])
        self.router.add_api_route("/dpo", self.trigger_dpo, methods=["POST"])
        self.router.add_api_route("/analyze", self.analyze_image, methods=["POST"])
        self.router.add_api_route("/detect", self.detect_objects, methods=["POST"])
        self.router.add_api_route("/pdf/upload", self.analyze_pdf, methods=["POST"])
        self.router.add_api_route("/process-video", self.process_video, methods=["POST"])
        self.router.add_api_route("/transcribe", self.transcribe_audio, methods=["POST"])
        self.router.add_api_route("/synthesize-speech", self.synthesize_speech, methods=["POST"])
        self.router.add_api_route("/generate-image", self.generate_image, methods=["POST"])
        self.router.add_api_route("/visual-dataset", self.create_visual_dataset, methods=["POST"])
        self.router.add_api_route("/checkpoints", self.list_checkpoints, methods=["GET"])
        self.router.add_api_route("/checkpoints/{name}/load", self.load_checkpoint, methods=["POST"])
        self.router.add_api_route("/checkpoints/{name}", self.delete_checkpoint, methods=["DELETE"])
        self.router.add_api_route("/reset", self.reset, methods=["POST"])

    # ── Helpers ──────────────────────────────────────────────────────

    def _ensure_initialized(self):
        mgr = get_multimodal_manager()
        if not getattr(mgr, "_initialized", False):
            speech_server = False
            try:
                mgr.initialize(vision_model="slonet", speech_server=speech_server)
            except Exception as e:
                logger.debug("Multimodal init deferred: %s", e)
        return mgr

    def _get_initialized_manager(self):
        """Return manager only if already initialized; avoids blocking on lazy init."""
        mgr = get_multimodal_manager()
        if not getattr(mgr, "_initialized", False):
            return None
        return mgr

    @staticmethod
    def _get_active_model_and_tokenizer():
        try:
            import state as server_state
            return server_state.model, server_state.tokenizer
        except Exception as exc:
            logger.debug("Model/state unavailable for multimodal: %s", exc)
            return None, None

    # ── Unified Status ────────────────────────────────────────────────

    async def status(self) -> dict:
        try:
            """status."""
            mgr = self._get_initialized_manager()
            if mgr is None:
                return success_response(data={
                    "engine": {"status": "not_initialized"},
                    "learning": {"images_learned": 0},
                    "batch": {"running": False, "job_id": None, "total": 0, "completed": 0, "errors": 0},
                    "video_training": {},
                    "dpo": {"status": "idle"},
                })
            caps = mgr.capabilities
            engine = getattr(mgr, "_multimodal_engine", None)
            learning = getattr(mgr, "_learning_count", 0)
            trained = getattr(engine, "_trained", False) if engine else False
            vocab_size = getattr(getattr(engine, "text", None), "vocab_size", 0) if engine else 0
            buf = getattr(mgr, "_replay_buffer", None)
            history = getattr(mgr, "_caption_history", [])
            accuracy_history = getattr(mgr, "_accuracy_history", [])
            unique = len(set(history)) if history else 0

            with self._video_training_lock:
                video = dict(self._video_training_state)

            with self._dpo_lock:
                dpo = dict(self._dpo_state)

            with self._bg_lock:
                bg = dict(self._background_job)

            return success_response(data={
                "engine": {
                    "speech_to_text": caps.speech_to_text,
                    "image_caption": caps.image_caption,
                    "speech_model": caps.speech_model,
                    "vision_model": caps.vision_model,
                    "status": "trained" if trained else ("learning" if learning > 0 else "ready"),
                },
                "learning": {
                    "images_learned": learning,
                    "trained": trained,
                    "vocab_size": vocab_size,
                    "replay_buffer_size": buf.size if buf else 0,
                    "learning_method": "contrastive + self-training",
                    "caption_history": history[-50:],
                    "unique_captions": unique,
                    "diversity_ratio": round(unique / max(len(history), 1), 3),
                    "accuracy_history": [round(a, 2) for a in accuracy_history[-50:]],
                    "mean_accuracy": round(sum(accuracy_history) / max(len(accuracy_history), 1), 2),
                    "last_accuracy": round(accuracy_history[-1], 2) if accuracy_history else 0.0,
                },
                "batch": {
                    "running": bg["running"],
                    "job_id": bg["job_id"],
                    "total": bg["total"],
                    "completed": bg["completed"],
                    "errors": bg["errors"],
                    "progress_pct": round(bg["completed"] / max(bg["total"], 1) * 100, 1) if bg["total"] > 0 else 0,
                    "current_caption": bg["current_caption"],
                    "current_image": bg["current_image"],
                },
                "dpo": dpo,
                "video": video,
            })

        # ── Training ───────────────────────────────────────────────────────

        except Exception as e:
            classify_and_raise(e, source="multimodal.status")
    async def train_on_image(self, file: UploadFile = File(...), label: Optional[str] = Form(None), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """train_on_image."""
        if not file.content_type or not file.content_type.startswith("image/"):
            raise_error("Only image files accepted", "E_BAD_REQUEST")
        mgr = self._ensure_initialized()
        try:
            contents = await file.read()
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            caption = await asyncio.to_thread(mgr.caption_image, img, ground_truth=label)
            safe_audit_log("multimodal.train", resource="image", detail="single", supervised=label is not None and label.strip() != "", accuracy=caption.accuracy)
            return success_response(data={
                "status": "ok",
                "caption": caption.text,
                "confidence": caption.confidence,
                "images_learned": getattr(mgr, "_learning_count", 0),
                "accuracy": caption.accuracy,
                "supervised": label is not None and label.strip() != "",
            })
        except Exception as e:
            logger.warning("Multimodal caption failed: %s", e)
            classify_and_raise(e, source="multimodal_caption")

    async def train_batch(
        self,
        files: Optional[List[UploadFile]] = File(None),
        dataset_path: Optional[str] = Form(None),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """train_batch."""
        mgr = self._ensure_initialized()
        with self._bg_lock:
            if self._background_job["running"]:
                raise_error("Training job already running", "E_INFRA_BUSY")

        image_paths = []
        if dataset_path:
            import glob
            dataset_path = os.path.expanduser(dataset_path)
            resolved_path = Path(dataset_path).resolve()
            _REPO_ROOT = Path(__file__).resolve().parents[4]
            allowed_bases = {_REPO_ROOT / "data", Path.home() / "Pictures", Path.home() / "Downloads"}
            if not any(resolved_path == base or str(resolved_path).startswith(str(base) + "/") for base in allowed_bases):
                raise_error(f"Directory not in allowed paths: {dataset_path}", "E_AUTH_FORBIDDEN")

            def _scan_dataset():
                if not os.path.isdir(dataset_path):
                    return None
                paths = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                    paths.extend(glob.glob(os.path.join(dataset_path, ext)))
                    paths.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
                return paths

            image_paths = await asyncio.to_thread(_scan_dataset)
            if image_paths is None:
                raise_error(f"Directory not found: {dataset_path}", "E_BAD_REQUEST")
            if not image_paths:
                raise_error(f"No images found in {dataset_path}", "E_BAD_REQUEST")
        if files:
            for f in files:
                if f.content_type and f.content_type.startswith("image/"):
                    image_paths.append(("upload", f))
        if not image_paths:
            raise_error("No images provided", "E_BAD_REQUEST")

        job_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self._bg_lock:
            self._background_job.update(
                job_id=job_id, running=True, total=len(image_paths),
                completed=0, errors=0, current_caption="", current_image="",
                started_at=datetime.datetime.now().isoformat(), finished_at=None,
            )
        asyncio.create_task(self._run_batch_training(mgr, image_paths))
        safe_audit_log("multimodal.train", resource=job_id, detail="batch", total_images=len(image_paths), dataset_path=dataset_path or "")
        return success_response(data={"status": "started", "job_id": job_id, "total_images": len(image_paths)})

    async def _run_batch_training(self, mgr, image_sources: list):
        import time as _time
        _batch_t0 = _time.monotonic()
        loop = asyncio.get_event_loop()
        for src in image_sources:
            with self._bg_lock:
                if not self._background_job["running"]:
                    break
            try:
                is_upload = isinstance(src, tuple) and src[0] == "upload"
                if is_upload:
                    contents = await src[1].read()
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(contents)).convert("RGB")
                    name = src[1].filename or "upload"
                    caption = await loop.run_in_executor(None, mgr.caption_image, img)
                else:
                    name = src.split("/")[-1]
                    caption = await loop.run_in_executor(None, mgr.train_on_path, src)
                with self._bg_lock:
                    self._background_job["current_image"] = name
                    self._background_job["current_caption"] = caption.text
                    self._background_job["completed"] += 1
            except Exception as exc:
                with self._bg_lock:
                    self._background_job["errors"] += 1
                    self._background_job["completed"] += 1
                logger.warning("Batch image training failed for %s: %s", name, exc)
            with self._bg_lock:
                completed = self._background_job["completed"]
            if completed % 5 == 0:
                await asyncio.sleep(0)
        with self._bg_lock:
            self._background_job["running"] = False
            self._background_job["finished_at"] = datetime.datetime.now().isoformat()
            _job_id = self._background_job.get("job_id", "unknown")
            _completed = self._background_job["completed"]
            _errors = self._background_job["errors"]
        _batch_elapsed_ms = (_time.monotonic() - _batch_t0) * 1000
        safe_audit_log("multimodal.train.complete", resource=_job_id, detail=f"elapsed={_batch_elapsed_ms:.0f}ms completed={_completed} errors={_errors}")

    async def train_video(self, req: VideoTrainRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """train_video."""
            import time as _time
            _t0 = _time.monotonic()
            with self._video_training_lock:
                if self._video_training_state["status"] == "running":
                    raise_error("Video training already in progress", "E_INFRA_BUSY")
                self._video_training_state["status"] = "running"
                self._video_training_state["error"] = None
                self._video_training_state["result"] = None

            job_id = f"video_{int(time.time())}"
            self._video_training_state["job_id"] = job_id

            def _progress(epoch, step, loss, total):
                with self._video_training_lock:
                    self._video_training_state["current_epoch"] = epoch
                    self._video_training_state["current_step"] = step
                    self._video_training_state["total_steps"] = total
                    self._video_training_state["current_loss"] = float(loss)

            def _run():
                try:
                    from domains.training.video_trainer import VideoCaptionTrainer
                    trainer = VideoCaptionTrainer(max_frames=8, lr=req.learning_rate)
                    result = trainer.train(
                        data_path=req.data_path, epochs=req.epochs, batch_size=req.batch_size,
                        lr=req.learning_rate, output_dir=req.output_dir, progress_callback=_progress,
                    )
                    with self._video_training_lock:
                        self._video_training_state["status"] = "completed" if result.get("status") == "completed" else "error"
                        self._video_training_state["result"] = result
                except Exception as e:
                    with self._video_training_lock:
                        self._video_training_state["status"] = "error"
                        self._video_training_state["error"] = str(e)

            from domains.training.executor import get_training_executor
            executor = get_training_executor()
            executor.submit(_run, f"vtrain_{job_id}")
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("multimodal.train", resource=req.data_path or job_id, detail=f"video elapsed={_elapsed_ms:.0f}ms", epochs=req.epochs, batch_size=req.batch_size)
            return success_response(data={"status": "started", "job_id": job_id, "data_path": req.data_path})

        except Exception as e:
            classify_and_raise(e, source="multimodal.train_video")
    async def video_infer(self, req: VideoInferRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """video_infer."""
        try:
            from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints
            checkpoints = list_video_checkpoints()
            if not checkpoints:
                checkpoints = list_video_checkpoints(str(Path(__file__).resolve().parents[4] / "models" / "video-training"))
            if not checkpoints:
                raise_error("No trained video model. Train via /multimodal/train-video first.", "E_BAD_REQUEST")
            latest = checkpoints[0]
            trainer = VideoCaptionTrainer()
            await asyncio.to_thread(trainer.load_checkpoint, latest["path"])
            t0 = time.time()
            text = await asyncio.to_thread(trainer.generate, video_path=req.video_path, max_len=req.max_len, temperature=req.temperature)
            return success_response(data={"text": text, "checkpoint": latest["name"], "elapsed_ms": round((time.time() - t0) * 1000, 1)})
        except HTTPException:
            classify_and_raise(e, source="multimodal.video_infer")
        except Exception as e:
            logger.warning("Multimodal video generate failed: %s", e)
            classify_and_raise(e, source="multimodal_video_generate")

    # ── DPO ────────────────────────────────────────────────────────────

    async def trigger_dpo(self, req: DPOTriggerRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """trigger_dpo."""
        model, tokenizer = self._get_active_model_and_tokenizer()
        if model is None or tokenizer is None:
            raise_error("No model loaded.", "E_BAD_REQUEST")
        with self._dpo_lock:
            if self._dpo_state["status"] == "running":
                raise_error("DPO already in progress", "E_INFRA_BUSY")
            self._dpo_state["status"] = "running"
            self._dpo_state["result"] = None
        try:
            from domains.feedback.hf_dpo import HFDPOTrainer
            trainer = HFDPOTrainer(model=model, tokenizer=tokenizer, learning_rate=req.learning_rate)
            t0 = time.time()
            result = trainer.train(max_pairs=req.max_pairs)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            with self._dpo_lock:
                self._dpo_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._dpo_state["result"] = result
                self._dpo_state["status"] = result["status"]
                if result["status"] == "accepted":
                    self._dpo_state["accepted_count"] += 1
                elif result["status"] == "rejected":
                    self._dpo_state["rejected_count"] += 1
            safe_audit_log("multimodal.dpo", resource="hf-model", detail=result["status"], steps=result.get("steps", 0), pairs_trained=result.get("pairs_trained", 0))
            return success_response(data={
                "status": result["status"], "steps": result.get("steps", 0),
                "avg_loss": result.get("avg_loss"), "ppl_before": result.get("ppl_before"),
                "ppl_after": result.get("ppl_after"), "ppl_delta_pct": result.get("ppl_delta_pct"),
                "pairs_trained": result.get("pairs_trained", 0), "elapsed_seconds": round(elapsed, 1),
            })
        except Exception as e:
            with self._dpo_lock:
                self._dpo_state["status"] = "error"
                self._dpo_state["result"] = {"error": "DPO training failed", "error_type": type(e).__name__}
            logger.warning("Multimodal DPO failed: %s", e)
            classify_and_raise(e, source="multimodal_dpo")

    # ── Analysis ──────────────────────────────────────────────────────

    async def analyze_image(self, file: UploadFile = File(...), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """analyze_image."""
        import time as _time
        _t0 = _time.monotonic()
        if not file.content_type or not file.content_type.startswith("image/"):
            raise_error("Only image files accepted", "E_BAD_REQUEST")
        mgr = self._ensure_initialized()
        try:
            contents = await file.read()
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            cap = await asyncio.to_thread(mgr.caption_image, img)
            learning = getattr(mgr, "_learning_count", 0)
            engine = getattr(mgr, "_multimodal_engine", None)
            buf = getattr(mgr, "_replay_buffer", None)
            accuracy_history = getattr(mgr, "_accuracy_history", [])
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            return success_response(data={
                "caption": cap.text, "confidence": cap.confidence, "tags": cap.tags or [],
                "accuracy": cap.accuracy, "supervised": cap.accuracy > 0,
                "images_learned": learning, "trained": getattr(engine, "_trained", False) if engine else False,
                "replay_buffer_size": buf.size if buf else 0,
                "mean_accuracy": round(sum(accuracy_history) / max(len(accuracy_history), 1), 2),
                "elapsed_ms": round(_elapsed_ms, 1),
            })
        except Exception as e:
            logger.warning("Multimodal analyze image failed: %s", e)
            classify_and_raise(e, source="multimodal_analyze_image")

    async def detect_objects(self, file: UploadFile = File(...)) -> dict:
        """detect_objects."""
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files accepted")
        mgr = self._ensure_initialized()
        try:
            contents = await file.read()
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            objects = mgr.detect_objects(img)
            return success_response(data={
                "objects": [
                    {
                        "label": obj.label,
                        "bbox": obj.bbox,
                        "confidence": obj.confidence,
                    }
                    for obj in objects
                ]
            })
        except Exception as e:
            classify_and_raise(e, source="multimodal_detect_objects")

    async def analyze_pdf(
        self,
        file: UploadFile = File(...),
        question: str = Form("Summarize this document."),
        per_page: bool = Form(False),
        max_new_tokens: int = Form(512),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """analyze_pdf."""
        import tempfile
        from domains.inference.pdf_vlm import PDFVLMProcessor
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            processor = PDFVLMProcessor(max_pages=10)
            if per_page:
                results = await asyncio.to_thread(processor.analyze_pages, tmp_path, question=question, max_new_tokens=max_new_tokens)
                text = "\n\n".join(f"--- Page {r['page']} ---\n{r['text']}" for r in results)
            else:
                text = await asyncio.to_thread(processor.analyze, tmp_path, question=question, max_new_tokens=max_new_tokens)
            return success_response(data={
                "analysis": text, "filename": file.filename, "pages_analyzed": 10,
                "method": "vlm" if processor._get_vlm() is not None else "text_extract",
            })
        except Exception as e:
            logger.warning("Multimodal analyze PDF failed: %s", e)
            classify_and_raise(e, source="multimodal_analyze_pdf")
        finally:
            await asyncio.to_thread(os.unlink, tmp_path)

    async def process_video(self, file: UploadFile = File(...), num_frames: int = Form(16), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """process_video."""
        try:
            from domains.multimodal.video import VideoProcessor
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            try:
                processor = VideoProcessor(max_frames=num_frames)
                frames = await asyncio.to_thread(processor.extract_frames, tmp_path, num_frames)
                mgr = get_multimodal_manager()
                engine = getattr(mgr, "_multimodal_engine", None)
                if engine is None:
                    raise_error("Multimodal engine not initialized", "E_INTERNAL", status_code=500)
                video_embedding = await asyncio.to_thread(processor.encode_video, frames, engine.vision)
                first_frame = frames[0].reshape(1, 224, 224, 3)
                caption = await asyncio.to_thread(engine.generate, first_frame, max_len=20, temperature=0.8)
                return success_response(data={"caption": caption.text, "num_frames": len(frames),
                        "video_embedding_shape": list(video_embedding.data.shape)})
            finally:
                await asyncio.to_thread(os.unlink, tmp_path)
        except Exception as e:
            logger.warning("Multimodal process video failed: %s", e)
            classify_and_raise(e, source="multimodal_process_video")

    # ── Speech ────────────────────────────────────────────────────────

    async def transcribe_audio(self, file: UploadFile = File(...), language: str = Form("en"), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """transcribe_audio."""
        import time as _time
        _t0 = _time.monotonic()
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise_error("Only audio files accepted", "E_BAD_REQUEST")
        mgr = self._ensure_initialized()
        if not mgr.capabilities.speech_to_text:
            raise_error("Server ASR not available.", "E_NOT_IMPLEMENTED", status_code=501)
        try:
            audio_data = await file.read()
            result = await asyncio.to_thread(mgr.recognize_speech, audio_data, language=language)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            return success_response(data={"text": result.text, "confidence": result.confidence,
                    "language": result.language or language, "duration": result.duration, "elapsed_ms": round(_elapsed_ms, 1)})
        except Exception as e:
            logger.warning("Multimodal transcribe failed: %s", e)
            classify_and_raise(e, source="multimodal_transcribe")

    async def synthesize_speech(self, text: str = Form(...), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """synthesize_speech."""
        import time as _time
        _t0 = _time.monotonic()
        try:
            from domains.multimodal.tts import TTSEngine
            import base64, io, wave, numpy as np
            if self._tts is None:
                self._tts = TTSEngine()
            tts = self._tts
            waveform = await asyncio.to_thread(tts.text_to_waveform, text)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(tts.sample_rate)
                wf.writeframes((waveform * 32767).astype(np.int16).tobytes())
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            return success_response(data={"audio": f"data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode()}",
                    "text": text, "duration_sec": len(waveform) / tts.sample_rate, "elapsed_ms": round(_elapsed_ms, 1)})
        except Exception as e:
            logger.warning("Multimodal TTS failed: %s", e)
            classify_and_raise(e, source="multimodal_tts")

    # ── Generation ────────────────────────────────────────────────────

    async def generate_image(self, prompt: str = Form(...), steps: int = Form(20),
                            guidance_scale: float = Form(7.5), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """generate_image."""
        try:
            from domains.multimodal.diffusion import LatentDiffusionModel
            from domains.multimodal.vae import SloVAE
            from domains.multimodal.text_encoder import TextEncoder
            import base64, io, numpy as np
            from PIL import Image
            if self._vae is None:
                self._vae = SloVAE(latent_dim=64)
                self._diffusion = LatentDiffusionModel(latent_dim=64)
                self._text_encoder = TextEncoder(vocab_size=4096, embed_dim=256)
            text_embeddings = await asyncio.to_thread(self._text_encoder.encode_text, [prompt])
            latents = await asyncio.to_thread(self._diffusion.sample, text_embeddings, num_steps=steps, guidance_scale=guidance_scale)
            image_np = await asyncio.to_thread(self._vae.decode, latents)
            image_np = np.clip(image_np[0].transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return success_response(data={"image": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}",
                    "prompt": prompt, "steps": steps})
        except Exception as e:
            logger.warning("Multimodal generate image failed: %s", e)
            classify_and_raise(e, source="multimodal_generate_image")

    # ── Dataset ───────────────────────────────────────────────────────

    async def create_visual_dataset(self, req: VisualDatasetRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """create_visual_dataset."""
            import time as _time
            _t0 = _time.monotonic()
            image_dir = Path(req.image_dir).resolve()
            _REPO_ROOT = Path(__file__).resolve().parents[4]
            allowed_bases = {_REPO_ROOT / "data", Path.home() / "Pictures", Path.home() / "Downloads"}
            if not any(image_dir == base or str(image_dir).startswith(str(base) + "/") for base in allowed_bases):
                        raise_error(f"Directory not in allowed paths: {req.image_dir}", "E_AUTH_FORBIDDEN")

            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            datasets_dir = Path(__file__).resolve().parents[4] / "data"
            output_path = datasets_dir / f"{req.name}.jsonl"

            def _validate_and_list():
                if not image_dir.exists():
                    raise_error(f"Directory not found: {req.image_dir}", "E_BAD_REQUEST")
                files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])
                if not files:
                    raise_error(f"No images in {req.image_dir}", "E_BAD_REQUEST")
                datasets_dir.mkdir(parents=True, exist_ok=True)
                return files

            image_files = await asyncio.to_thread(_validate_and_list)
            auto_captioned = False
            if req.auto_caption:
                try:
                    mgr = self._ensure_initialized()
                    auto_captioned = True
                except Exception as exc:
                    auto_captioned = False
                    logger.debug("Auto-caption init failed: %s", exc)
            def _build_dataset():
                entries = 0
                with open(output_path, "w") as f:
                    for img_path in image_files:
                        entry = {"image_path": str(img_path), "caption": ""}
                        if req.auto_caption and auto_captioned:
                            try:
                                caps = mgr.caption_image(str(img_path))
                                if caps:
                                    entry["caption"] = caps[0].text
                            except Exception as exc:
                                logger.debug("Per-image caption failed for %s: %s", img_path, exc)
                        f.write(json.dumps(entry) + "\n")
                        entries += 1
                return entries

            entries = await asyncio.to_thread(_build_dataset)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("multimodal.visual_dataset", resource=req.name, detail=f"entries={entries} auto_caption={auto_captioned} elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={"status": "created", "dataset": req.name, "path": str(output_path),
                    "entries": entries, "auto_captioned": auto_captioned, "elapsed_ms": round(_elapsed_ms, 1)})

        # ── Checkpoints ──────────────────────────────────────────────────

        except Exception as e:
            classify_and_raise(e, source="multimodal.create_visual_dataset")
    async def list_checkpoints(self):
        try:
            """list_checkpoints."""
            try:
                from domains.training.video_trainer import list_video_checkpoints

                def _list():
                    ckpts = list_video_checkpoints()
                    if not ckpts:
                        ckpts = list_video_checkpoints(str(Path(__file__).resolve().parents[4] / "models" / "video-training"))
                    return ckpts

                return await asyncio.to_thread(_list)
            except Exception as exc:
                logger.warning("list_checkpoints failed: %s", exc)
                return []

        except Exception as e:
            classify_and_raise(e, source="multimodal.list_checkpoints")
    async def load_checkpoint(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """load_checkpoint."""
        try:
            from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints

            def _find_checkpoint():
                ckpts = list_video_checkpoints()
                if not ckpts:
                    ckpts = list_video_checkpoints(str(Path(__file__).resolve().parents[4] / "models" / "video-training"))
                return [c for c in ckpts if c["name"] == name]

            match = await asyncio.to_thread(_find_checkpoint)
            if not match:
                raise_error(f"Checkpoint '{name}' not found", "E_NOT_FOUND")
            trainer = VideoCaptionTrainer()
            await asyncio.to_thread(trainer.load_checkpoint, match[0]["path"])
            safe_audit_log("multimodal.checkpoint.load", resource=name)
            return success_response(data={"status": "loaded", "checkpoint": name})
        except HTTPException:
            classify_and_raise(exc, source="multimodal._find_checkpoint")
        except Exception as e:
            logger.warning("Multimodal load checkpoint failed: %s", e)
            classify_and_raise(e, source="multimodal_load_checkpoint")

    async def delete_checkpoint(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """delete_checkpoint."""
        try:
            from domains.training.video_trainer import list_video_checkpoints

            def _find_and_delete():
                ckpts = list_video_checkpoints()
                if not ckpts:
                    ckpts = list_video_checkpoints(str(Path(__file__).resolve().parents[4] / "models" / "video-training"))
                match = [c for c in ckpts if c["name"] == name]
                if not match:
                    return False
                path = Path(match[0]["path"])
                for p in [path, path.with_suffix(".npz"), path.parent / f"{path.stem}_meta.json"]:
                    if p.exists():
                        os.remove(p)
                return True

            found = await asyncio.to_thread(_find_and_delete)
            if not found:
                raise_error(f"Checkpoint '{name}' not found", "E_NOT_FOUND")
            safe_audit_log("multimodal.checkpoint.delete", resource=name)
            return success_response(data={"status": "deleted", "checkpoint": name})
        except HTTPException:
            classify_and_raise(e, source="multimodal._find_and_delete")
        except Exception as e:
            logger.warning("Multimodal delete checkpoint failed: %s", e)
            classify_and_raise(e, source="multimodal_delete_checkpoint")

    # ── Reset ─────────────────────────────────────────────────────────

    async def reset(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """reset."""
            mgr = self._ensure_initialized()
            mgr._learning_count = 0
            mgr._caption_history = []
            mgr._accuracy_history = []
            if getattr(mgr, "_replay_buffer", None):
                mgr._replay_buffer.clear()
            mgr._multimodal_engine = None
            logger.info("Multimodal engine reset: all state cleared")
            safe_audit_log("multimodal.reset", resource="all")
            return success_response(data={"status": "ok", "message": "Multimodal engine reset"})


        except Exception as e:
            classify_and_raise(e, source="multimodal.reset")
# ── Module-level exports ──────────────────────────────────────────

multimodal_router = MultimodalRouter()
router = multimodal_router.router
_background_job = multimodal_router._background_job