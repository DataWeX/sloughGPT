"""Feedback training route — from-feedback.

Extracted from execution.py to keep each module focused.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from domains.shared import find_repo_root
from domains.training.executor import get_training_executor
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import raise_error

from .controller import get_training_controller
from .helpers import _finish_job, _run_async
from .jobs import training_jobs
from .webhooks import notify_training_event

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-feedback"])


class FromFeedbackRequest(BaseModel):
    """Request schema for training from feedback data."""

    epochs: int = Field(default=3, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(default=16, ge=1, le=512, description="Batch size")


@router.post("/training/from-feedback")
async def train_from_feedback(
    req: FromFeedbackRequest | None = None,
    auth_user: dict = Depends(require_auth_if_enabled),
):
    """Train a model from collected feedback data."""
    import uuid

    epochs = req.epochs if req else 3
    learning_rate = req.learning_rate if req else 1e-4
    batch_size = req.batch_size if req else 16

    try:
        from domains.feedback.training import FeedbackTrainer

        trainer = FeedbackTrainer()

        # Export feedback data
        timestamp = int(time.time())
        export_dir = find_repo_root(Path(__file__).resolve()) / "data" / "training_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export as SFT format for training
        sft_path = export_dir / f"feedback_sft_{timestamp}.jsonl"
        count = trainer.export_sft(str(sft_path))

        if count == 0:
            return {"status": "no_data", "message": "No feedback data available for training"}

        # Create training job
        jid = f"feedback_train_{uuid.uuid4().hex[:8]}"
        data_path = str(sft_path)
        out_stem = f"feedback_model_{timestamp}"

        cancel_event = threading.Event()
        training_jobs[jid] = {
            "id": jid,
            "name": f"Feedback Training {timestamp}",
            "status": "running",
            "progress": 0,
            "dataset": str(sft_path),
            "data_source": "feedback",
            "epochs": epochs,
            "checkpoint_interval": 100,
            "output_checkpoint_stem": out_stem,
            "_cancel_event": cancel_event,
        }

        # Register with CancelManager
        try:
            from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

            _mgr = get_cancel_manager()
            op_id = _mgr.register(
                op_type=OpType.TRAINING,
                label=f"feedback:{out_stem}",
                cancel_fn=lambda: cancel_event.set(),
            )
            _mgr.start(op_id)
            training_jobs[jid]["_cancel_manager_op_id"] = op_id
        except Exception as exc:
            logger.warning(
                "CancelManager registration failed for feedback training %s: %s", jid, exc
            )

        # Update global training controller
        get_training_controller().start(jid, f"Feedback Training {timestamp}")

        def run_feedback_training(job_id_: str = jid):
            try:
                from domains.training.train_pipeline import SloughGPTTrainer

                trainer = SloughGPTTrainer(
                    data_path=data_path,
                    n_embed=256,
                    n_layer=6,
                    n_head=8,
                    block_size=256,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate,
                    use_lora=True,
                    lora_rank=8,
                    lora_alpha=16,
                    checkpoint_dir="models",
                    checkpoint_interval=100,
                )

                def on_progress(info: dict) -> None:
                    training_jobs[jid]["progress"] = min(99, int(info.get("progress_percent", 0)))
                    training_jobs[jid]["current_epoch"] = int(
                        info.get("epoch", training_jobs[jid].get("current_epoch", 0))
                    )
                    tl = info.get("train_loss")
                    if tl is not None:
                        training_jobs[jid]["train_loss"] = float(tl)
                        training_jobs[jid].setdefault("loss_history", []).append(
                            {
                                "step": info.get("global_step", 0),
                                "value": float(tl),
                                "type": "train",
                            }
                        )
                    el = info.get("eval_loss")
                    if el is not None:
                        training_jobs[jid]["eval_loss"] = float(el)
                        training_jobs[jid]["loss"] = float(el)
                        training_jobs[jid].setdefault("loss_history", []).append(
                            {"step": info.get("global_step", 0), "value": float(el), "type": "eval"}
                        )

                trainer.train(on_progress=on_progress)
                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
                trainer.save(f"models/{safe_stem}.soul")

                _finish_job(jid, "completed")
                training_jobs[jid]["progress"] = 100
                training_jobs[jid]["checkpoint"] = f"models/{safe_stem}.soul"
                training_jobs[jid]["samples_used"] = count
                get_training_controller().complete()

                # Trigger webhook notification (fire and forget)
                try:
                    _run_async(
                        notify_training_event(
                            "training.completed",
                            {
                                "job_id": jid,
                                "job_name": training_jobs[jid].get("name", "feedback_training"),
                                "status": "completed",
                                "samples_used": count,
                                "checkpoint": training_jobs[jid]["checkpoint"],
                            },
                        )
                    )
                except Exception as e:
                    logger.debug("Feedback training webhook failed: %s", e, extra={"tag": "TRAIN"})

            except Exception as e:
                logger.exception("Feedback training job %s failed", jid, extra={"tag": "TRAIN"})
                _finish_job(jid, "failed", str(e))
                get_training_controller().fail(str(e))

                # Trigger webhook notification
                try:
                    _run_async(
                        notify_training_event(
                            "training.failed",
                            {
                                "job_id": jid,
                                "job_name": training_jobs[jid].get("name", "feedback_training"),
                                "status": "failed",
                                "error": str(e),
                            },
                        )
                    )
                except Exception as e:
                    logger.warning("Feedback training failure webhook failed: %s: %s", jid, e)

        executor = get_training_executor()
        executor.submit(run_feedback_training, jid)

        safe_out_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
        return {
            "status": "started",
            "job_id": jid,
            "samples": count,
            "data_path": str(sft_path),
            "checkpoint": f"models/{safe_out_stem}.soul",
            "message": "Training started from feedback data",
        }

    except Exception as e:
        logger.exception("Failed to start feedback training", extra={"tag": "TRAIN"})
        raise_error(str(e), "E_INFRA_STARTUP", status_code=500)
