"""Multimodal Router - vision, speech, and background batch training."""

import asyncio
import datetime
import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from domains.multimodal import get_multimodal_manager

logger = logging.getLogger("sloughgpt.routers.multimodal")

router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# ── Background training state ──────────────────────────────────────────────

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


def _ensure_initialized():
    mgr = get_multimodal_manager()
    if not getattr(mgr, "_initialized", False):
        speech_server = False
        try:
            import transformers  # noqa: F401
            from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
            speech_server = True
            logger.info("Whisper detected — enabling server-side ASR")
        except ImportError:
            logger.info("Whisper not available — using browser Web Speech API for voice")
        mgr.initialize(vision_model="slonet", speech_server=speech_server)
    return mgr


# ── Capabilities & Progress ────────────────────────────────────────────────

@router.get("/capabilities")
async def get_capabilities():
    """Get current multimodal capabilities and learning state."""
    mgr = _ensure_initialized()
    caps = mgr.capabilities
    learning = getattr(mgr, "_learning_count", 0)
    engine = getattr(mgr, "_multimodal_engine", None)
    trained = getattr(engine, "_trained", False) if engine else False
    buf = getattr(mgr, "_replay_buffer", None)
    return {
        "speech_to_text": caps.speech_to_text,
        "image_caption": caps.image_caption,
        "speech_model": caps.speech_model,
        "vision_model": caps.vision_model,
        "images_learned": learning,
        "trained": trained,
        "replay_buffer_size": buf.size if buf else 0,
        "learning_method": "contrastive + self-training",
        "background_job_running": _background_job["running"],
        "status": "trained" if trained else ("learning" if learning > 0 else "ready"),
    }


@router.get("/learning-progress")
async def get_learning_progress():
    """Get learning progress of the multimodal engine."""
    mgr = _ensure_initialized()
    learning = getattr(mgr, "_learning_count", 0)
    engine = getattr(mgr, "_multimodal_engine", None)
    trained = getattr(engine, "_trained", False) if engine else False
    vocab_size = len(engine.text.vocab) if engine and hasattr(engine, "text") else 0
    buf = getattr(mgr, "_replay_buffer", None)
    return {
        "images_learned": learning,
        "trained": trained,
        "vocab_size": vocab_size,
        "replay_buffer_size": buf.size if buf else 0,
    }


@router.get("/training-report")
async def get_training_report():
    """Get detailed training report with caption evolution."""
    mgr = _ensure_initialized()
    history = getattr(mgr, "_caption_history", [])
    learning = getattr(mgr, "_learning_count", 0)
    engine = getattr(mgr, "_multimodal_engine", None)
    vocab_size = len(engine.text.vocab) if engine and hasattr(engine, "text") else 0
    buf = getattr(mgr, "_replay_buffer", None)
    unique = len(set(history)) if history else 0
    return {
        "images_learned": learning,
        "vocab_size": vocab_size,
        "replay_buffer_size": buf.size if buf else 0,
        "caption_history": history[-50:],
        "unique_captions": unique,
        "diversity_ratio": round(unique / max(len(history), 1), 3),
        "trained": getattr(engine, "_trained", False) if engine else False,
    }


# ── Speech recognition ─────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("en")):
    """Transcribe audio file using server-side Whisper (if available).

    Accepts WAV/MP3/OGG audio uploads. Returns transcribed text.
    Falls back gracefully if server ASR is not initialized.
    """
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files accepted")

    mgr = _ensure_initialized()

    if not mgr.capabilities.speech_to_text:
        raise HTTPException(
            status_code=501,
            detail="Server-side speech recognition not available. "
                   "Install transformers + whisper to enable, or use browser-based Web Speech API.",
        )

    try:
        contents = await file.read()
        result = mgr.recognize_speech(contents, language=language)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language or language,
            "duration": result.duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


# ── Single image training ──────────────────────────────────────────────────

@router.post("/train")
async def train_on_image(file: UploadFile = File(...)):
    """Upload a single image to train the multimodal engine."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted")

    mgr = _ensure_initialized()

    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        caption = mgr.caption_image(img)
        return {
            "status": "ok",
            "caption": caption.text,
            "confidence": caption.confidence,
            "images_learned": getattr(mgr, "_learning_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Background batch training ──────────────────────────────────────────────

@router.post("/train-batch")
async def train_batch(
    files: Optional[List[UploadFile]] = File(None),
    dataset_path: Optional[str] = Form(None),
):
    """Start background batch training on multiple images.

    Accepts either uploaded image files or a local directory path.
    Returns immediately with a job_id. Poll /training-status for progress.
    """
    mgr = _ensure_initialized()

    if _background_job["running"]:
        raise HTTPException(status_code=409, detail="Training job already running")

    image_paths = []

    if dataset_path:
        import glob, os
        dataset_path = os.path.expanduser(dataset_path)
        if not os.path.isdir(dataset_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {dataset_path}")
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        for ext in exts:
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
        job_id=job_id,
        running=True,
        total=len(image_paths),
        completed=0,
        errors=0,
        current_caption="",
        current_image="",
        started_at=datetime.datetime.now().isoformat(),
        finished_at=None,
    )

    asyncio.create_task(_run_batch_training(mgr, image_paths))
    return {
        "status": "started",
        "job_id": job_id,
        "total_images": len(image_paths),
    }


@router.get("/training-status")
async def get_training_status():
    """Get current background batch training status."""
    return {
        "running": _background_job["running"],
        "job_id": _background_job["job_id"],
        "total": _background_job["total"],
        "completed": _background_job["completed"],
        "errors": _background_job["errors"],
        "progress_pct": round(
            _background_job["completed"] / max(_background_job["total"], 1) * 100, 1
        ) if _background_job["total"] > 0 else 0,
        "current_caption": _background_job["current_caption"],
        "current_image": _background_job["current_image"],
        "started_at": _background_job["started_at"],
        "finished_at": _background_job["finished_at"],
    }


async def _run_batch_training(mgr, image_sources: list):
    """Process images in background, updating job state."""
    loop = asyncio.get_event_loop()

    for src in image_sources:
        if not _background_job["running"]:
            break

        try:
            is_upload = isinstance(src, tuple) and src[0] == "upload"
            if is_upload:
                upload_file = src[1]
                contents = await upload_file.read()
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                name = upload_file.filename or "upload"
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
