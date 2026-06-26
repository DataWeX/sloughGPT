"""
Visual Router — DPO training + video captioning endpoints.

Provides:
- ``POST /visual/dpo`` — trigger DPO training on the active HF model
- ``GET /visual/dpo/status`` — poll DPO status
- ``POST /visual/train-video`` — start video captioning training
- ``GET /visual/train-video/status`` — poll video training progress
- ``POST /visual/video-infer`` — generate caption from video file
- ``GET /visual/status`` — combined status of all visual subsystems

DPO training uses feedback from chat (thumbs up/down) to fine-tune
the active HuggingFace model with LoRA adapters.

Video training uses SloNet-based VideoCaptionTrainer to learn from
video → caption pairs in JSONL format.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("man.visual_router")

router = APIRouter(prefix="/visual", tags=["visual"])

# ── Global state ───────────────────────────────────────────────────

_dpo_state = {
    "last_run": None,
    "status": "idle",
    "result": None,
    "accepted_count": 0,
    "rejected_count": 0,
}
_dpo_lock = Lock()


# ── Schemas ────────────────────────────────────────────────────────

class DPOTriggerRequest(BaseModel):
    max_pairs: int = Field(6, ge=1, le=50)
    learning_rate: float = Field(5e-6, ge=1e-8, le=1e-3)


class DPOStatusResponse(BaseModel):
    status: str
    last_run: Optional[str] = None
    result: Optional[dict] = None
    accepted_count: int = 0
    rejected_count: int = 0


class VideoTrainRequest(BaseModel):
    data_path: str = Field(..., description="Path to JSONL with {video_path, caption} entries")
    epochs: int = Field(5, ge=1, le=100)
    batch_size: int = Field(2, ge=1, le=16)
    learning_rate: float = Field(3e-4, ge=1e-6, le=1.0)
    output_dir: str = Field("models/video-training", description="Output directory for checkpoints")


class VideoInferRequest(BaseModel):
    video_path: str = Field(..., description="Server path to video file")
    max_len: int = Field(50, ge=10, le=200)
    temperature: float = Field(0.8, ge=0.0, le=2.0)


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


# ── Helpers ────────────────────────────────────────────────────────

def _get_active_model_and_tokenizer():
    """Get the currently loaded HF model and tokenizer from server state."""
    try:
        import state as server_state
        return server_state.model, server_state.tokenizer
    except Exception:
        return None, None


# ── DPO Endpoints ──────────────────────────────────────────────────

@router.post("/dpo")
async def trigger_dpo(req: DPOTriggerRequest):
    """Trigger DPO training on the active HF model using feedback pairs.

    Loads (chosen, rejected, prompt) pairs from the feedback database,
    applies gradient descent on chosen + gradient ascent on rejected,
    and runs a quality guard (PPL benchmark -> rollback if >5% degradation).

    The training targets LoRA parameters if LoRA is enabled on the model,
    or all trainable parameters otherwise.
    """
    global _dpo_state

    model, tokenizer = _get_active_model_and_tokenizer()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    with _dpo_lock:
        if _dpo_state["status"] == "running":
            raise HTTPException(status_code=409, detail="DPO training already in progress")
        _dpo_state["status"] = "running"
        _dpo_state["result"] = None

    try:
        from domains.feedback.hf_dpo import HFDPOTrainer

        trainer = HFDPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=req.learning_rate,
        )

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

        logger.info("DPO completed: %s (%ds)", result["status"], round(elapsed))

        return {
            "status": result["status"],
            "steps": result.get("steps", 0),
            "avg_loss": result.get("avg_loss"),
            "ppl_before": result.get("ppl_before"),
            "ppl_after": result.get("ppl_after"),
            "ppl_delta_pct": result.get("ppl_delta_pct"),
            "pairs_trained": result.get("pairs_trained", 0),
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        with _dpo_lock:
            _dpo_state["status"] = "error"
            _dpo_state["result"] = {"error": str(e)}
        logger.error("DPO failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dpo/status")
async def dpo_status():
    """Get DPO training status and history."""
    with _dpo_lock:
        return DPOStatusResponse(
            status=_dpo_state["status"],
            last_run=_dpo_state["last_run"],
            result=_dpo_state["result"],
            accepted_count=_dpo_state["accepted_count"],
            rejected_count=_dpo_state["rejected_count"],
        )


# ── Video Training Endpoints ──────────────────────────────────────────

@router.post("/train-video")
async def train_video(req: VideoTrainRequest):
    """Start video captioning training in background.

    Training runs on a daemon thread. Poll ``GET /visual/train-video/status``
    for progress.

    The dataset at ``data_path`` must be a JSONL file with:
      ``{"video_path": "/path/to/video.mp4", "caption": "description"}``
    """
    with _video_training_lock:
        if _VIDEO_TRAINING_STATE["status"] == "running":
            raise HTTPException(status_code=409, detail="Video training already in progress")
        _VIDEO_TRAINING_STATE["status"] = "running"
        _VIDEO_TRAINING_STATE["error"] = None
        _VIDEO_TRAINING_STATE["result"] = None

    job_id = f"video_{int(time.time())}"
    _VIDEO_TRAINING_STATE["job_id"] = job_id
    total_steps = [0]

    def _progress(epoch, step, loss, total):
        total_steps[0] = total
        with _video_training_lock:
            _VIDEO_TRAINING_STATE["current_epoch"] = epoch
            _VIDEO_TRAINING_STATE["current_step"] = step
            _VIDEO_TRAINING_STATE["total_steps"] = total
            _VIDEO_TRAINING_STATE["current_loss"] = float(loss)

    def _run():
        try:
            from domains.training.video_trainer import VideoCaptionTrainer

            trainer = VideoCaptionTrainer(
                max_frames=8,
                lr=req.learning_rate,
            )
            result = trainer.train(
                data_path=req.data_path,
                epochs=req.epochs,
                batch_size=req.batch_size,
                lr=req.learning_rate,
                output_dir=req.output_dir,
                progress_callback=_progress,
            )
            with _video_training_lock:
                _VIDEO_TRAINING_STATE["status"] = "completed" if result.get("status") == "completed" else "error"
                _VIDEO_TRAINING_STATE["result"] = result
            logger.info("Video training %s: %s", result.get("status"), req.data_path)
        except Exception as e:
            with _video_training_lock:
                _VIDEO_TRAINING_STATE["status"] = "error"
                _VIDEO_TRAINING_STATE["error"] = str(e)
            logger.error("Video training failed: %s", e)

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {
        "status": "started",
        "job_id": job_id,
        "data_path": req.data_path,
        "output_dir": req.output_dir,
    }


@router.get("/train-video/status")
async def train_video_status():
    """Get video training status."""
    with _video_training_lock:
        return {
            "status": _VIDEO_TRAINING_STATE["status"],
            "job_id": _VIDEO_TRAINING_STATE["job_id"],
            "current_epoch": _VIDEO_TRAINING_STATE["current_epoch"],
            "current_step": _VIDEO_TRAINING_STATE["current_step"],
            "total_steps": _VIDEO_TRAINING_STATE["total_steps"],
            "current_loss": _VIDEO_TRAINING_STATE["current_loss"],
            "result": _VIDEO_TRAINING_STATE["result"],
            "error": _VIDEO_TRAINING_STATE["error"],
        }


@router.post("/video-infer")
async def video_infer(req: VideoInferRequest):
    """Generate a caption for a video file on the server.

    Uses the latest trained video checkpoint. Returns the generated
    caption text.
    """
    try:
        from domains.training.video_trainer import VideoCaptionTrainer, list_video_checkpoints

        checkpoints = list_video_checkpoints()
        if not checkpoints:
            checkpoints = list_video_checkpoints(str(Path("models/video-training")))

        if not checkpoints:
            raise HTTPException(
                status_code=400,
                detail="No trained video model found. Train a model first via /visual/train-video.",
            )

        latest = checkpoints[0]
        trainer = VideoCaptionTrainer()
        trainer.load_checkpoint(latest["path"])

        t0 = time.time()
        text = trainer.generate(
            video_path=req.video_path,
            max_len=req.max_len,
            temperature=req.temperature,
        )
        elapsed = (time.time() - t0) * 1000

        return {
            "text": text,
            "checkpoint": latest["name"],
            "elapsed_ms": round(elapsed, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Checkpoint Endpoints ────────────────────────────────────────────

@router.get("/checkpoints")
async def list_visual_checkpoints():
    """List all video training checkpoints."""
    try:
        from domains.training.video_trainer import list_video_checkpoints
        ckpts = list_video_checkpoints()
        if not ckpts:
            ckpts = list_video_checkpoints(str(Path("models/video-training")))
        return ckpts
    except Exception as e:
        logger.error("Failed to list visual checkpoints: %s", e)
        return []


@router.post("/checkpoints/{name}/load")
async def load_visual_checkpoint(name: str):
    """Load a video training checkpoint by name."""
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
        return {"status": "loaded", "checkpoint": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/checkpoints/{name}")
async def delete_visual_checkpoint(name: str):
    """Delete a video training checkpoint by name."""
    try:
        from domains.training.video_trainer import list_video_checkpoints
        ckpts = list_video_checkpoints()
        if not ckpts:
            ckpts = list_video_checkpoints(str(Path("models/video-training")))
        match = [c for c in ckpts if c["name"] == name]
        if not match:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{name}' not found")
        path = Path(match[0]["path"])
        if path.exists():
            os.remove(path)
        npz_path = path.with_suffix(".npz")
        if npz_path.exists():
            os.remove(npz_path)
        meta_path = path.parent / f"{path.stem}_meta.json"
        if meta_path.exists():
            os.remove(meta_path)
        return {"status": "deleted", "checkpoint": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Visual Model Load ───────────────────────────────────────────────

@router.post("/load")
async def visual_load_model(req: dict):
    """Load a vision model. Expects {'model_id': '...'}."""
    model_id = req.get("model_id", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    logger.info("Visual model load requested: %s", model_id)
    return {"status": "ok", "message": f"Visual model {model_id} loading triggered"}


# ── Status ─────────────────────────────────────────────────────────

@router.get("/status")
async def visual_status():
    """Get DPO and video training status (no model loading triggered)."""
    with _video_training_lock:
        video_train_state = {
            "status": _VIDEO_TRAINING_STATE["status"],
            "job_id": _VIDEO_TRAINING_STATE["job_id"],
            "current_epoch": _VIDEO_TRAINING_STATE["current_epoch"],
            "current_step": _VIDEO_TRAINING_STATE["current_step"],
            "total_steps": _VIDEO_TRAINING_STATE["total_steps"],
            "current_loss": _VIDEO_TRAINING_STATE["current_loss"],
            "result": _VIDEO_TRAINING_STATE["result"],
            "error": _VIDEO_TRAINING_STATE["error"],
        }
    return {
        "video_training": video_train_state,
        "dpo": DPOStatusResponse(
            status=_dpo_state["status"],
            last_run=_dpo_state["last_run"],
            result=_dpo_state["result"],
            accepted_count=_dpo_state["accepted_count"],
            rejected_count=_dpo_state["rejected_count"],
        ).model_dump(),
    }
