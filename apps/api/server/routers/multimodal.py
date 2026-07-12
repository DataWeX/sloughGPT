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
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from domains.multimodal import get_multimodal_manager
from schemas.common import success_response

logger = logging.getLogger("man.routers.multimodal")

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


# ── State ──────────────────────────────────────────────────────────

_dpo_state = {
    "last_run": None,
    "status": "idle",
    "result": None,
    "accepted_count": 0,
    "rejected_count": 0,
}
_dpo_lock = Lock()

_VIDEO_TRAINING_STATE = {
    "status": "idle",
    "job_id": None,
    "current_epoch": 0,
    "current_step": 0,
    "total_steps": 0,
    "current_loss": None,
    "result": None,
    "error": None,
}
_video_training_lock = Lock()

_background_job = {
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


# ── Helpers ────────────────────────────────────────────────────────

def _ensure_initialized():
    mgr = get_multimodal_manager()
    if not getattr(mgr, "_initialized", False):
        speech_server = False
        try:
            import transformers  # noqa: F401
            from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
            speech_server = True
        except ImportError:
            pass
        mgr.initialize(vision_model="slonet", speech_server=speech_server)
    return mgr


def _get_active_model_and_tokenizer():
    try:
        import state as server_state
        return server_state.model, server_state.tokenizer
    except Exception:
        return None, None


# ── Unified Status (replaces 7 separate status endpoints) ──────────

@router.get("/status")
async def status():
    """Get combined status of all multimodal subsystems.

    Replaces: /capabilities, /learning-progress, /training-report,
    /training-status, /dpo/status, /train-video/status, /generation-status
    """
    mgr = _ensure_initialized()
    caps = mgr.capabilities
    engine = getattr(mgr, "_multimodal_engine", None)
    learning = getattr(mgr, "_learning_count", 0)
    trained = getattr(engine, "_trained", False) if engine else False
    vocab_size = getattr(getattr(engine, "text", None), "vocab_size", 0) if engine else 0
    buf = getattr(mgr, "_replay_buffer", None)
    history = getattr(mgr, "_caption_history", [])
    accuracy_history = getattr(mgr, "_accuracy_history", [])
    unique = len(set(history)) if history else 0

    with _video_training_lock:
        video = dict(_VIDEO_TRAINING_STATE)

    with _dpo_lock:
        dpo = dict(_dpo_state)

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
            "running": _background_job["running"],
            "job_id": _background_job["job_id"],
            "total": _background_job["total"],
            "completed": _background_job["completed"],
            "errors": _background_job["errors"],
            "progress_pct": round(_background_job["completed"] / max(_background_job["total"], 1) * 100, 1) if _background_job["total"] > 0 else 0,
            "current_caption": _background_job["current_caption"],
            "current_image": _background_job["current_image"],
        },
        "dpo": dpo,
        "video": video,
    })


# ── Training ───────────────────────────────────────────────────────

@router.post("/train")
async def train_on_image(file: UploadFile = File(...), label: Optional[str] = Form(None)):
    """Train on a single image. Returns caption + accuracy."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted")
    mgr = _ensure_initialized()
    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        caption = mgr.caption_image(img, ground_truth=label)
        return success_response(data={
            "status": "ok",
            "caption": caption.text,
            "confidence": caption.confidence,
            "images_learned": getattr(mgr, "_learning_count", 0),
            "accuracy": caption.accuracy,
            "supervised": label is not None and label.strip() != "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-batch")
async def train_batch(
    files: Optional[List[UploadFile]] = File(None),
    dataset_path: Optional[str] = Form(None),
):
    """Start background batch training. Poll GET /status for progress."""
    mgr = _ensure_initialized()
    if _background_job["running"]:
        raise HTTPException(status_code=409, detail="Training job already running")

    image_paths = []
    if dataset_path:
        import glob
        dataset_path = os.path.expanduser(dataset_path)
        if not os.path.isdir(dataset_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {dataset_path}")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
            image_paths.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        if not image_paths:
            raise HTTPException(status_code=400, detail=f"No images found in {dataset_path}")
    if files:
        for f in files:
            if f.content_type and f.content_type.startswith("image/"):
                image_paths.append(("upload", f))
    if not image_paths:
        raise HTTPException(status_code=400, detail="No images provided")

    job_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _background_job.update(
        job_id=job_id, running=True, total=len(image_paths),
        completed=0, errors=0, current_caption="", current_image="",
        started_at=datetime.datetime.now().isoformat(), finished_at=None,
    )
    asyncio.create_task(_run_batch_training(mgr, image_paths))
    return success_response(data={"status": "started", "job_id": job_id, "total_images": len(image_paths)})


async def _run_batch_training(mgr, image_sources: list):
    loop = asyncio.get_event_loop()
    for src in image_sources:
        if not _background_job["running"]:
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
            _background_job["current_image"] = name
            _background_job["current_caption"] = caption.text
            _background_job["completed"] += 1
        except Exception:
            _background_job["errors"] += 1
            _background_job["completed"] += 1
        if _background_job["completed"] % 5 == 0:
            await asyncio.sleep(0)
    _background_job["running"] = False
    _background_job["finished_at"] = datetime.datetime.now().isoformat()


@router.post("/train-video")
async def train_video(req: VideoTrainRequest):
    """Start video captioning training in background."""
    with _video_training_lock:
        if _VIDEO_TRAINING_STATE["status"] == "running":
            raise HTTPException(status_code=409, detail="Video training already in progress")
        _VIDEO_TRAINING_STATE["status"] = "running"
        _VIDEO_TRAINING_STATE["error"] = None
        _VIDEO_TRAINING_STATE["result"] = None

    job_id = f"video_{int(time.time())}"
    _VIDEO_TRAINING_STATE["job_id"] = job_id

    def _progress(epoch, step, loss, total):
        with _video_training_lock:
            _VIDEO_TRAINING_STATE["current_epoch"] = epoch
            _VIDEO_TRAINING_STATE["current_step"] = step
            _VIDEO_TRAINING_STATE["total_steps"] = total
            _VIDEO_TRAINING_STATE["current_loss"] = float(loss)

    def _run():
        try:
            from domains.training.video_trainer import VideoCaptionTrainer
            trainer = VideoCaptionTrainer(max_frames=8, lr=req.learning_rate)
            result = trainer.train(
                data_path=req.data_path, epochs=req.epochs, batch_size=req.batch_size,
                lr=req.learning_rate, output_dir=req.output_dir, progress_callback=_progress,
            )
            with _video_training_lock:
                _VIDEO_TRAINING_STATE["status"] = "completed" if result.get("status") == "completed" else "error"
                _VIDEO_TRAINING_STATE["result"] = result
        except Exception as e:
            with _video_training_lock:
                _VIDEO_TRAINING_STATE["status"] = "error"
                _VIDEO_TRAINING_STATE["error"] = str(e)

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return success_response(data={"status": "started", "job_id": job_id, "data_path": req.data_path})


@router.post("/video-infer")
async def video_infer(req: VideoInferRequest):
    """Generate a caption for a video file."""
    try:
        from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints
        checkpoints = list_video_checkpoints()
        if not checkpoints:
            checkpoints = list_video_checkpoints(str(Path("models/video-training")))
        if not checkpoints:
            raise HTTPException(status_code=400, detail="No trained video model. Train via /multimodal/train-video first.")
        latest = checkpoints[0]
        trainer = VideoCaptionTrainer()
        trainer.load_checkpoint(latest["path"])
        t0 = time.time()
        text = trainer.generate(video_path=req.video_path, max_len=req.max_len, temperature=req.temperature)
        return success_response(data={"text": text, "checkpoint": latest["name"], "elapsed_ms": round((time.time() - t0) * 1000, 1)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DPO ────────────────────────────────────────────────────────────

@router.post("/dpo")
async def trigger_dpo(req: DPOTriggerRequest):
    """Trigger DPO training on the active HF model."""
    global _dpo_state
    model, tokenizer = _get_active_model_and_tokenizer()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    with _dpo_lock:
        if _dpo_state["status"] == "running":
            raise HTTPException(status_code=409, detail="DPO already in progress")
        _dpo_state["status"] = "running"
        _dpo_state["result"] = None
    try:
        from domains.feedback.hf_dpo import HFDPOTrainer
        trainer = HFDPOTrainer(model=model, tokenizer=tokenizer, learning_rate=req.learning_rate)
        t0 = time.time()
        result = trainer.train(max_pairs=req.max_pairs)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        with _dpo_lock:
            _dpo_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _dpo_state["result"] = result
            _dpo_state["status"] = result["status"]
            if result["status"] == "accepted":
                _dpo_state["accepted_count"] += 1
            elif result["status"] == "rejected":
                _dpo_state["rejected_count"] += 1
        return success_response(data={
            "status": result["status"], "steps": result.get("steps", 0),
            "avg_loss": result.get("avg_loss"), "ppl_before": result.get("ppl_before"),
            "ppl_after": result.get("ppl_after"), "ppl_delta_pct": result.get("ppl_delta_pct"),
            "pairs_trained": result.get("pairs_trained", 0), "elapsed_seconds": round(elapsed, 1),
        })
    except Exception as e:
        with _dpo_lock:
            _dpo_state["status"] = "error"
            _dpo_state["result"] = {"error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


# ── Analysis ───────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an image — returns caption, confidence, tags, model state."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted")
    mgr = _ensure_initialized()
    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        cap = mgr.caption_image(img)
        learning = getattr(mgr, "_learning_count", 0)
        engine = getattr(mgr, "_multimodal_engine", None)
        buf = getattr(mgr, "_replay_buffer", None)
        accuracy_history = getattr(mgr, "_accuracy_history", [])
        return success_response(data={
            "caption": cap.text, "confidence": cap.confidence, "tags": cap.tags or [],
            "accuracy": cap.accuracy, "supervised": cap.accuracy > 0,
            "images_learned": learning, "trained": getattr(engine, "_trained", False) if engine else False,
            "replay_buffer_size": buf.size if buf else 0,
            "mean_accuracy": round(sum(accuracy_history) / max(len(accuracy_history), 1), 2),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@router.post("/pdf/upload")
async def analyze_pdf(
    file: UploadFile = File(...),
    question: str = Form("Summarize this document."),
    per_page: bool = Form(False),
    max_new_tokens: int = Form(512),
):
    """Analyze an uploaded PDF — VLM first, text extraction fallback."""
    import tempfile
    from domains.inference.pdf_vlm import PDFVLMProcessor
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        processor = PDFVLMProcessor(max_pages=10)
        if per_page:
            results = processor.analyze_pages(tmp_path, question=question, max_new_tokens=max_new_tokens)
            text = "\n\n".join(f"--- Page {r['page']} ---\n{r['text']}" for r in results)
        else:
            text = processor.analyze(tmp_path, question=question, max_new_tokens=max_new_tokens)
        return success_response(data={
            "analysis": text, "filename": file.filename, "pages_analyzed": 10,
            "method": "vlm" if processor._get_vlm() is not None else "text_extract",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {e}")
    finally:
        os.unlink(tmp_path)


@router.post("/process-video")
async def process_video(file: UploadFile = File(...), num_frames: int = Form(16)):
    """Process uploaded video and generate caption."""
    try:
        from domains.multimodal.video import VideoProcessor
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            processor = VideoProcessor(max_frames=num_frames)
            frames = processor.extract_frames(tmp_path, num_frames)
            mgr = get_multimodal_manager()
            engine = getattr(mgr, "_multimodal_engine", None)
            if engine is None:
                raise HTTPException(status_code=500, detail="Multimodal engine not initialized")
            video_embedding = processor.encode_video(frames, engine.vision)
            first_frame = frames[0].reshape(1, 224, 224, 3)
            caption = engine.generate(first_frame, max_len=20, temperature=0.8)
            return success_response(data={"status": "success", "caption": caption.text, "num_frames": len(frames),
                    "video_embedding_shape": list(video_embedding.data.shape)})
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")


# ── Speech ─────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("en")):
    """Transcribe audio via server-side Whisper (if available)."""
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files accepted")
    mgr = _ensure_initialized()
    if not mgr.capabilities.speech_to_text:
        raise HTTPException(status_code=501, detail="Server ASR not available.")
    try:
        result = mgr.recognize_speech(await file.read(), language=language)
        return success_response(data={"text": result.text, "confidence": result.confidence,
                "language": result.language or language, "duration": result.duration})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@router.post("/synthesize-speech")
async def synthesize_speech(text: str = Form(...)):
    """Synthesize speech from text. Returns base64 WAV."""
    try:
        from domains.multimodal.tts import TTSEngine
        import base64, io, wave
        if not hasattr(synthesize_speech, "_tts"):
            synthesize_speech._tts = TTSEngine()
        tts = synthesize_speech._tts
        waveform = tts.text_to_waveform(text)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts.sample_rate)
            wf.writeframes((waveform * 32767).astype(np.int16).tobytes())
        return success_response(data={"status": "success", "audio": f"data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode()}",
                "text": text, "duration_sec": len(waveform) / tts.sample_rate})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ── Generation ─────────────────────────────────────────────────────

@router.post("/generate-image")
async def generate_image(prompt: str = Form(...), steps: int = Form(20),
                        guidance_scale: float = Form(7.5)):
    """Generate an image from text prompt using latent diffusion."""
    try:
        from domains.multimodal.diffusion import LatentDiffusionModel
        from domains.multimodal.vae import SloVAE
        from domains.multimodal.text_encoder import TextEncoder
        import base64
        from PIL import Image
        import io
        if not hasattr(generate_image, "_vae"):
            generate_image._vae = SloVAE(latent_dim=64)
            generate_image._diffusion = LatentDiffusionModel(latent_dim=64)
            generate_image._text_encoder = TextEncoder(vocab_size=4096, embed_dim=256)
        text_embeddings = generate_image._text_encoder.encode_text([prompt])
        latents = generate_image._diffusion.sample(text_embeddings, num_steps=steps, guidance_scale=guidance_scale)
        image_np = generate_image._vae.decode(latents)
        image_np = np.clip(image_np[0].transpose(1, 2, 0), 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return success_response(data={"status": "success", "image": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}",
                "prompt": prompt, "steps": steps})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


# ── Dataset ────────────────────────────────────────────────────────

@router.post("/visual-dataset")
async def create_visual_dataset(req: VisualDatasetRequest):
    """Create a visual training dataset from a directory of images."""
    image_dir = Path(req.image_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.image_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])
    if not image_files:
        raise HTTPException(status_code=400, detail=f"No images in {req.image_dir}")
    datasets_dir = Path(__file__).resolve().parents[4] / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    output_path = datasets_dir / f"{req.name}.jsonl"
    entries = 0
    auto_captioned = False
    if req.auto_caption:
        try:
            mgr = _ensure_initialized()
            auto_captioned = True
        except Exception:
            auto_captioned = False
    with open(output_path, "w") as f:
        for img_path in image_files:
            entry = {"image_path": str(img_path), "caption": ""}
            if req.auto_caption and auto_captioned:
                try:
                    caps = mgr.caption_image(str(img_path))
                    if caps:
                        entry["caption"] = caps[0].text
                except Exception:
                    pass
            f.write(json.dumps(entry) + "\n")
            entries += 1
    return success_response(data={"status": "created", "dataset": req.name, "path": str(output_path),
            "entries": entries, "auto_captioned": auto_captioned})


# ── Checkpoints ────────────────────────────────────────────────────

@router.get("/checkpoints")
async def list_checkpoints():
    """List video training checkpoints."""
    try:
        from domains.training.video_trainer import list_video_checkpoints
        ckpts = list_video_checkpoints()
        if not ckpts:
            ckpts = list_video_checkpoints(str(Path("models/video-training")))
        return ckpts
    except Exception:
        return []


@router.post("/checkpoints/{name}/load")
async def load_checkpoint(name: str):
    """Load a video training checkpoint."""
    try:
        from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints
        ckpts = list_video_checkpoints()
        if not ckpts:
            ckpts = list_video_checkpoints(str(Path("models/video-training")))
        match = [c for c in ckpts if c["name"] == name]
        if not match:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{name}' not found")
        trainer = VideoCaptionTrainer()
        trainer.load_checkpoint(match[0]["path"])
        return success_response(data={"status": "loaded", "checkpoint": name})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    """Delete a video training checkpoint."""
    try:
        from domains.training.video_trainer import list_video_checkpoints
        ckpts = list_video_checkpoints()
        if not ckpts:
            ckpts = list_video_checkpoints(str(Path("models/video-training")))
        match = [c for c in ckpts if c["name"] == name]
        if not match:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{name}' not found")
        path = Path(match[0]["path"])
        for p in [path, path.with_suffix(".npz"), path.parent / f"{path.stem}_meta.json"]:
            if p.exists():
                os.remove(p)
        return success_response(data={"status": "deleted", "checkpoint": name})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Reset ──────────────────────────────────────────────────────────

@router.post("/reset")
async def reset():
    """Reset the multimodal engine — clears learned data."""
    mgr = _ensure_initialized()
    mgr._learning_count = 0
    mgr._caption_history = []
    mgr._accuracy_history = []
    if getattr(mgr, "_replay_buffer", None):
        mgr._replay_buffer.clear()
    mgr._multimodal_engine = None
    return success_response(data={"status": "ok", "message": "Multimodal engine reset"})
