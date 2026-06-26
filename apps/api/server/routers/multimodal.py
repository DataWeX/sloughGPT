"""Multimodal Router - vision, speech, and background batch training."""

import asyncio
import datetime
import json
import logging
from typing import List, Optional
from pathlib import Path
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from domains.multimodal import get_multimodal_manager, ImageCaption

logger = logging.getLogger("man.routers.multimodal")

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
    vocab_size = len(engine.text.bpe.vocab) if engine and hasattr(engine, "text") else 0
    buf = getattr(mgr, "_replay_buffer", None)
    return {
        "images_learned": learning,
        "trained": trained,
        "vocab_size": vocab_size,
        "replay_buffer_size": buf.size if buf else 0,
    }


@router.get("/training-report")
async def get_training_report():
    """Get detailed training report with caption evolution and accuracy."""
    mgr = _ensure_initialized()
    history = getattr(mgr, "_caption_history", [])
    learning = getattr(mgr, "_learning_count", 0)
    engine = getattr(mgr, "_multimodal_engine", None)
    vocab_size = len(engine.text.bpe.vocab) if engine and hasattr(engine, "text") else 0
    buf = getattr(mgr, "_replay_buffer", None)
    unique = len(set(history)) if history else 0
    accuracy_history = getattr(mgr, "_accuracy_history", [])
    mean_accuracy = round(sum(accuracy_history) / max(len(accuracy_history), 1), 2)
    last_accuracy = round(accuracy_history[-1], 2) if accuracy_history else 0.0
    return {
        "images_learned": learning,
        "vocab_size": vocab_size,
        "replay_buffer_size": buf.size if buf else 0,
        "caption_history": history[-50:],
        "unique_captions": unique,
        "diversity_ratio": round(unique / max(len(history), 1), 3),
        "trained": getattr(engine, "_trained", False) if engine else False,
        "accuracy_history": [round(a, 2) for a in accuracy_history[-50:]],
        "mean_accuracy": mean_accuracy,
        "last_accuracy": last_accuracy,
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
async def train_on_image(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
):
    """Upload a single image to train the multimodal engine.

    When label is provided, uses supervised training with that as ground truth.
    Returns BLEU accuracy between model's generated caption and the label.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted")

    mgr = _ensure_initialized()

    try:
        contents = await file.read()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        caption = mgr.caption_image(img, ground_truth=label)
        return {
            "status": "ok",
            "caption": caption.text,
            "confidence": caption.confidence,
            "images_learned": getattr(mgr, "_learning_count", 0),
            "accuracy": caption.accuracy,
            "supervised": label is not None and label.strip() != "",
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


# ── Text-to-Image Generation ───────────────────────────────────────────────

@router.post("/generate-image")
async def generate_image(prompt: str = Form(...), steps: int = Form(20), 
                        guidance_scale: float = Form(7.5), width: int = Form(224),
                        height: int = Form(224)):
    """Generate an image from text prompt using latent diffusion."""
    try:
        from domains.multimodal.diffusion import LatentDiffusionModel
        from domains.multimodal.vae import SloVAE
        from domains.multimodal.text_encoder import TextEncoder
        import base64
        from PIL import Image
        import io
        
        # Initialize models (lazy load)
        if not hasattr(generate_image, "_vae"):
            generate_image._vae = SloVAE(latent_dim=64)
            generate_image._diffusion = LatentDiffusionModel(latent_dim=64)
            generate_image._text_encoder = TextEncoder(vocab_size=4096, embed_dim=256)
        
        vae = generate_image._vae
        diffusion = generate_image._diffusion
        text_encoder = generate_image._text_encoder
        
        # Encode text
        text_embeddings = text_encoder.encode_text([prompt])
        
        # Sample latents
        latents = diffusion.sample(text_embeddings, num_steps=steps, 
                                  guidance_scale=guidance_scale)
        
        # Decode to image
        image_np = vae.decode(latents)
        
        # Convert to PIL and then to base64
        # image_np is (1, 3, 224, 224) in [0, 1]
        image_np = np.clip(image_np[0].transpose(1, 2, 0), 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "status": "success",
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "steps": steps,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ── Analyze Image ──────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Upload and analyze an image — returns caption, confidence, features, and model status.

    Runs the image through the multimodal engine's training+caption pipeline,
    returning structured analysis alongside current model state.
    """
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
        trained = getattr(engine, "_trained", False) if engine else False
        buf = getattr(mgr, "_replay_buffer", None)
        accuracy_history = getattr(mgr, "_accuracy_history", [])
        mean_accuracy_ = round(sum(accuracy_history) / max(len(accuracy_history), 1), 2)

        return {
            "caption": cap.text,
            "confidence": cap.confidence,
            "tags": cap.tags or [],
            "accuracy": cap.accuracy,
            "supervised": cap.accuracy > 0,
            "images_learned": learning,
            "trained": trained,
            "replay_buffer_size": buf.size if buf else 0,
            "mean_accuracy": mean_accuracy_,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ── Reset model ────────────────────────────────────────────────────────────

@router.post("/reset")
async def reset_multimodal():
    """Reset the multimodal engine — clears learned images, replay buffer, and accuracy history."""
    mgr = _ensure_initialized()
    mgr._learning_count = 0
    mgr._caption_history = []
    mgr._accuracy_history = []
    if getattr(mgr, "_replay_buffer", None):
        mgr._replay_buffer.clear()
    mgr._multimodal_engine = None
    return {"status": "ok", "message": "Multimodal engine reset"}


@router.get("/generation-status")
async def get_generation_status():
    """Get status of text-to-image generation models."""
    has_models = hasattr(generate_image, "_vae")
    return {
        "models_loaded": has_models,
        "capabilities": {
            "text_to_image": True,
            "image_to_image": False,
            "inpainting": False,
        } if has_models else {
            "text_to_image": "loading",
            "image_to_image": False,
            "inpainting": False,
        }
    }


# ── Video Processing ───────────────────────────────────────────────────────

@router.post("/process-video")
async def process_video(file: UploadFile = File(...), num_frames: int = Form(16)):
    """Process video and generate caption."""
    try:
        from domains.multimodal.video import VideoProcessor
        from domains.multimodal import get_multimodal_manager
        import tempfile
        import os
        
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Initialize video processor
            processor = VideoProcessor(max_frames=num_frames)
            
            # Extract frames
            frames = processor.extract_frames(tmp_path, num_frames)
            
            # Get vision encoder from multimodal manager
            mgr = get_multimodal_manager()
            engine = getattr(mgr, "_multimodal_engine", None)
            if engine is None:
                raise HTTPException(status_code=500, detail="Multimodal engine not initialized")
            
            # Encode video
            video_embedding = processor.encode_video(frames, engine.vision)
            
            # Generate caption (using first frame for now)
            first_frame = frames[0].reshape(1, 224, 224, 3)
            caption = engine.generate(first_frame, max_len=20, temperature=0.8)
            
            return {
                "status": "success",
                "caption": caption.text,
                "num_frames": len(frames),
                "video_embedding_shape": list(video_embedding.data.shape),
            }
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


# ── Text-to-Speech ─────────────────────────────────────────────────────────

@router.post("/synthesize-speech")
async def synthesize_speech(text: str = Form(...)):
    """Synthesize speech from text."""
    try:
        from domains.multimodal.tts import TTSEngine
        import base64
        import io
        import wave
        
        # Initialize TTS engine
        if not hasattr(synthesize_speech, "_tts"):
            synthesize_speech._tts = TTSEngine()
        
        tts = synthesize_speech._tts
        
        # Generate waveform
        waveform = tts.text_to_waveform(text)
        
        # Convert to WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(tts.sample_rate)
            wf.writeframes((waveform * 32767).astype(np.int16).tobytes())
        
        # Convert to base64
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "status": "success",
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "text": text,
            "duration_sec": len(waveform) / tts.sample_rate,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# ── Visual Dataset ─────────────────────────────────────────────────

class VisualDatasetRequest(BaseModel):
    name: str
    image_dir: str
    caption_prompt: str = "Describe this image in detail."
    auto_caption: bool = True


@router.post("/visual-dataset")
async def create_visual_dataset(req: VisualDatasetRequest):
    """Create a visual training dataset from a directory of images.

    Scans ``image_dir`` for image files, optionally auto-captions them
    using the multimodal manager, and saves a JSONL dataset.
    """
    image_dir = Path(req.image_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"Image directory not found: {req.image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])
    if not image_files:
        raise HTTPException(status_code=400, detail=f"No image files found in {req.image_dir}")

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
                    entry["caption"] = ""
            f.write(json.dumps(entry) + "\n")
            entries += 1

    logger.info("Created visual dataset: %s (%d entries)", req.name, entries)

    return {
        "status": "created",
        "dataset": req.name,
        "path": str(output_path),
        "entries": entries,
        "auto_captioned": auto_captioned,
    }

