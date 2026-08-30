"""Visual training route — visual-start.

Extracted from execution.py to keep each module focused.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fastapi import APIRouter

from schemas.common import raise_error

from .schemas import VisualTrainRequest
from .jobs import training_jobs
from .helpers import _finish_job
from domains.training.executor import get_training_executor
from domains.shared import find_repo_root

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-visual"])


@router.post("/training/visual-start")
async def start_visual_training(request: VisualTrainRequest):
    """Start a vision-language model (VLM) fine-tune on a visual dataset.

    Uses the video trainer's VideoCaptionTrainer on an image-caption
    dataset, running stage-1 connector tuning then stage-2 full fine-tune.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    datasets_dir = find_repo_root(Path(__file__).resolve()) / "data"
    data_path = datasets_dir / request.dataset
    if not data_path.exists():
        data_path = datasets_dir / f"{request.dataset}.jsonl"
    if not data_path.exists():
        raise_error(f"Dataset not found: {request.dataset}", "E_BAD_REQUEST", status_code=400)
    data_path_str = str(data_path)

    out_stem = request.name or f"vlm_{job_id}"
    output_dir = find_repo_root(Path(__file__).resolve()) / "models" / "video-training" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"VLM-{job_id}",
        "model": f"{request.vision_encoder}+{request.llm}",
        "dataset": request.dataset,
        "status": "queued",
        "progress": 0,
        "epochs": request.stage1_epochs + request.stage2_epochs,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "error": None,
        "result": None,
        "loss_history": [],
    }
    cancel_event = threading.Event()
    job["_cancel_event"] = cancel_event
    training_jobs[job_id] = job

    # Register with CancelManager
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _mgr = get_cancel_manager()
        op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=str(request.name or f"vlm-{job_id}"),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "visual"},
            op_id=job_id,
        )
        _mgr.start(job_id)
    except Exception as e:
        logger.warning("CancelManager registration failed for visual training %s: %s", job_id, e)

    def _run_visual(job_id_: str = job_id):
        try:
            training_jobs[job_id_]["status"] = "running"
            from domains.training.video_trainer import VideoCaptionTrainer, VideoTrainConfig

            config = VideoTrainConfig(
                data_path=data_path_str,
                output_dir=str(output_dir),
                vision_encoder=request.vision_encoder,
                llm=request.llm,
                connector_hidden_dim=request.connector_hidden_dim,
                max_seq_length=request.max_seq_length,
                stage1_epochs=request.stage1_epochs,
                stage2_epochs=request.stage2_epochs,
                stage1_lr=request.stage1_lr,
                stage2_lr=request.stage2_lr,
                batch_size=request.batch_size,
                use_lora=request.use_lora,
                lora_rank=request.lora_rank,
                freeze_vision=request.freeze_vision,
                cancel_event=cancel_event,
            )

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(job_id_)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                loss = info.get("loss")
                if loss is not None:
                    rec["loss"] = float(loss)
                    rec.setdefault("loss_history", []).append({"step": rec["global_step"], "value": float(loss), "type": "train"})

            trainer = VideoCaptionTrainer(config)
            result = trainer.train(on_progress=on_progress)

            if cancel_event.is_set():
                _finish_job(job_id_, "cancelled")
                return

            training_jobs[job_id_].update({
                "progress": 100,
                "status": "completed",
                "loss": result.get("final_loss"),
                "result": result,
            })
            _finish_job(job_id_, "completed")
            logger.info("Visual training complete: job %s", job_id_, extra={"tag": "TRAIN"})

        except Exception as e:
            logger.exception("Visual training job %s failed", job_id_, extra={"tag": "TRAIN"})
            _finish_job(job_id_, "failed", str(e))

    executor = get_training_executor()
    executor.submit(_run_visual, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "message": f"Visual training started: encoder={request.vision_encoder} llm={request.llm}",
    }
