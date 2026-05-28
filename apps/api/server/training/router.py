"""FastAPI routes for char-level training and job orchestration.

Trainer ``step_*.pt`` charset maps: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from training.jobs import training_jobs
from training.resolution import resolve_training_inputs
from training.schemas import TrainingRequest, TrainRequest, TrainResolveRequest
from training.controller import get_training_controller, TrainingState
from training.webhooks import (
    get_webhook_store,
    TRAINING_EVENTS,
    notify_training_event,
)
from training.job_store import get_job_store

logger = logging.getLogger("man")

router = APIRouter(tags=["training"])


def _sloughgpt_trainer_kwds(req_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build ``SloughGPTTrainer`` keyword arguments from a request ``model_dump()`` (except ``data_path``)."""
    device = req_snapshot.get("device")
    return {
        "n_embed": int(req_snapshot.get("n_embed") or 128),
        "n_layer": int(req_snapshot.get("n_layer") or 4),
        "n_head": int(req_snapshot.get("n_head") or 4),
        "block_size": int(req_snapshot.get("block_size") or 128),
        "dropout": float(
            req_snapshot.get("dropout") if req_snapshot.get("dropout") is not None else 0.1
        ),
        "batch_size": int(req_snapshot.get("batch_size") or 32),
        "epochs": int(req_snapshot.get("epochs") or 3),
        "lr": float(req_snapshot.get("learning_rate") or 1e-3),
        "max_steps": req_snapshot.get("max_steps"),
        "gradient_accumulation_steps": int(req_snapshot.get("gradient_accumulation_steps") or 1),
        "max_grad_norm": float(
            req_snapshot.get("max_grad_norm")
            if req_snapshot.get("max_grad_norm") is not None
            else 1.0
        ),
        "use_mixed_precision": bool(req_snapshot.get("use_mixed_precision", True)),
        "mixed_precision_dtype": str(req_snapshot.get("mixed_precision_dtype") or "bf16"),
        "checkpoint_dir": str(req_snapshot.get("checkpoint_dir") or "checkpoints"),
        "checkpoint_interval": int(req_snapshot.get("checkpoint_interval") or 500),
        "save_best_only": bool(req_snapshot.get("save_best_only", False)),
        "max_checkpoints": int(req_snapshot.get("max_checkpoints") or 5),
        "scheduler_type": str(req_snapshot.get("scheduler") or "cosine"),
        "warmup_steps": int(
            req_snapshot.get("warmup_steps")
            if req_snapshot.get("warmup_steps") is not None
            else 100
        ),
        "min_lr": float(
            req_snapshot.get("min_lr") if req_snapshot.get("min_lr") is not None else 1e-5
        ),
        "weight_decay": float(
            req_snapshot.get("weight_decay")
            if req_snapshot.get("weight_decay") is not None
            else 0.01
        ),
        "use_lora": bool(req_snapshot.get("use_lora", False)),
        "lora_rank": int(req_snapshot.get("lora_rank") or 8),
        "lora_alpha": int(req_snapshot.get("lora_alpha") or 16),
        "log_interval": int(req_snapshot.get("log_interval") or 10),
        "eval_interval": int(req_snapshot.get("eval_interval") or 100),
        "device": device if device is not None and str(device).strip() != "" else None,
    }


@router.post("/train")
async def train(request: TrainRequest):
    """Start a training job (background thread).

    ``SloughGPTTrainer`` writes periodic ``step_*.pt`` under ``checkpoint_dir`` with
    ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError
    from domains.training.train_pipeline import SloughGPTTrainer

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    req_snapshot = request.model_dump()

    def train_model() -> None:
        try:
            trainer = SloughGPTTrainer(
                data_path=data_path_str,
                **_sloughgpt_trainer_kwds(req_snapshot),
            )
            trainer.train()
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
        except Exception as e:
            logger.exception("Background /train failed: %s", e)

    thread = threading.Thread(target=train_model, daemon=True)
    thread.start()

    out: dict[str, Any] = {
        "status": "started",
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "epochs": request.epochs,
        "message": "Training started in background",
    }
    if request.dataset is not None:
        out["dataset"] = request.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.post("/train/resolve")
async def train_resolve(body: TrainResolveRequest) -> dict[str, Any]:
    """Resolve ``data_path`` and checkpoint stem (dry run; no training).

    Does not write ``.pt`` artifacts. After ``POST /train`` or ``POST /training/start``,
    native ``step_*.pt`` includes char vocab; see ``docs/policies/CONTRIBUTING.md``
    (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            body.dataset,
            body.manifest_uri,
            body.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out: dict[str, Any] = {
        "ok": True,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
    }
    if body.dataset is not None:
        out["dataset"] = body.dataset.strip()
    if manifest_meta is not None:
        out["manifest"] = manifest_meta
    return out


@router.get("/train/status")
async def train_status():
    """Legacy training status stub."""
    return {"status": "ready", "message": "Use /train endpoint to start training"}


@router.get("/training/jobs")
async def list_training_jobs():
    """List all tracked training jobs.

    Completed jobs may expose ``checkpoint``; native ``step_*.pt`` uses charset maps for
    ``cli.py eval`` — ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    return list(training_jobs.values())


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get one training job by id.

    See list endpoint docstring for ``checkpoint`` / ``step_*.pt`` vocabulary semantics.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@router.delete("/training/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job and optionally its checkpoint files.

    Removes job from registry. If ``delete_files`` is true, removes checkpoint
    files associated with the job from disk.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    deleted_files = []

    if job.get("checkpoint"):
        checkpoint_path = Path(job["checkpoint"])
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                deleted_files.append(str(checkpoint_path))
            except OSError:
                pass

    if job.get("checkpoint_dir"):
        checkpoint_dir = Path(job["checkpoint_dir"])
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            try:
                shutil.rmtree(checkpoint_dir)
                deleted_files.append(str(checkpoint_dir))
            except OSError:
                pass

    del training_jobs[job_id]

    return {
        "status": "deleted",
        "job_id": job_id,
        "deleted_files": deleted_files,
    }


@router.get("/training/export/{job_id}")
async def export_training_job(job_id: str):
    """Export a completed training job's checkpoint file."""
    from fastapi.responses import FileResponse

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]

    if job.get("status") not in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Job must be completed before export")

    checkpoint = job.get("checkpoint")
    if not checkpoint:
        raise HTTPException(status_code=404, detail="No checkpoint found for this job")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")

    return FileResponse(
        path=checkpoint_path,
        filename=checkpoint_path.name,
        media_type="application/octet-stream",
    )


@router.post("/training/export-text")
async def export_feedback_pairs(request: Request):
    """Export feedback conversation pairs with a minimum quality threshold."""
    from controllers.feedback import get_feedback_controller
    body = await request.json()
    min_quality = body.get("min_quality", 0)
    target_count = body.get("target_count", 100)
    ctrl = get_feedback_controller()
    pairs = []
    feedback_file = ctrl.feedback_dir / "feedback.jsonl"
    if feedback_file.exists():
        with open(feedback_file) as f:
            for line in f:
                fb = json.loads(line)
                if fb.get("user_message") and fb.get("assistant_response"):
                    pairs.append(fb)
                    if len(pairs) >= target_count:
                        break
    output_file = ctrl.feedback_dir / f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({"pairs": pairs, "total": len(pairs)}, f, indent=2)
    return {"pairs_count": len(pairs), "filepath": str(output_file), "status": "exported"}


@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """Start a tracked training job (web UI).

    ``step_*.pt`` files saved on the server include ``stoi`` / ``itos`` / ``chars``
    for char-LM eval; see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    from domains.training.dataset_manifest import ManifestError

    try:
        data_path_str, out_stem, manifest_meta, source_kind = resolve_training_inputs(
            request.dataset,
            request.manifest_uri,
            request.dataset_ref,
        )
    except ManifestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job_id = f"job_{len(training_jobs) + 1}"
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name,
        "model": request.model,
        "dataset": request.dataset.strip() if request.dataset else out_stem,
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "data_source": source_kind,
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
    }
    if manifest_meta is not None:
        job["manifest"] = manifest_meta
    training_jobs[job_id] = job

    # Update global training controller
    controller = get_training_controller()
    controller.start(job_id, request.name or "training")

    # Trigger webhook notification for training started
    try:
        import asyncio

        async def notify_async():
            await notify_training_event(
                "training.started",
                {
                    "job_id": job_id,
                    "job_name": request.name or "training",
                    "dataset": request.dataset,
                    "epochs": request.epochs,
                },
            )

        asyncio.create_task(notify_async())
    except Exception:
        pass

    req_snapshot = request.model_dump()
    data_path_for_thread = data_path_str
    out_stem_for_thread = out_stem
    jid = job_id

    def run_training() -> None:
        from domains.training.train_pipeline import SloughGPTTrainer
        from domains.training.wandb_helpers import create_training_tracker_for_api_job

        tracker = None
        try:
            tracker = create_training_tracker_for_api_job(
                job_id=jid,
                job_name=str(req_snapshot.get("name") or "training"),
                data_path=data_path_for_thread,
                hyperparams=dict(req_snapshot),
            )

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                tl = info.get("train_loss")
                if tl is not None:
                    rec["train_loss"] = float(tl)
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe

            trainer = SloughGPTTrainer(
                data_path=data_path_for_thread,
                **_sloughgpt_trainer_kwds(req_snapshot),
                experiment_tracker=tracker,
            )
            result = trainer.train(on_progress=on_progress)
            safe_stem = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in out_stem_for_thread
            )[:120]
            trainer.save(f"models/{safe_stem}_trained.pt")
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["current_epoch"] = int(req_snapshot.get("epochs") or 3)
            bel = result.get("best_eval_loss")
            training_jobs[jid]["loss"] = bel if bel is not None and bel < float("inf") else None
            training_jobs[jid]["checkpoint"] = f"models/{safe_stem}_trained.pt"
            get_training_controller().complete()

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name", "training"),
                            "status": "completed",
                            "loss": training_jobs[jid]["loss"],
                            "checkpoint": training_jobs[jid]["checkpoint"],
                        },
                    )
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Training job %s failed", jid)
            training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(e)
            training_jobs[jid]["progress"] = 0
            get_training_controller().fail(str(e))

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.failed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name", "training"),
                            "status": "failed",
                            "error": str(e),
                        },
                    )
                )
            except Exception:
                pass
        finally:
            if tracker is not None:
                try:
                    tracker.end_run()
                except Exception:
                    logger.exception("W&B end_run failed for job %s", jid)

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()


@router.post("/training/from-feedback")
async def train_from_feedback():
    """Train a model from collected feedback data.

    This endpoint:
    1. Exports feedback as training data (DPO format)
    2. Starts training with the exported data
    3. Returns the job ID for tracking
    """
    import os
    import uuid
    from pathlib import Path
    from pydantic import BaseModel

    class TrainFromFeedbackRequest(BaseModel):
        epochs: int = 3
        batch_size: int = 16
        learning_rate: float = 1e-4
        use_lora: bool = True

    try:
        from domains.feedback.training import FeedbackTrainer

        trainer = FeedbackTrainer()

        # Export feedback data
        timestamp = int(time.time())
        export_dir = Path("data/training_exports")
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

        training_jobs[jid] = {
            "id": jid,
            "name": f"Feedback Training {timestamp}",
            "status": "running",
            "progress": 0,
            "dataset": str(sft_path),
            "data_source": "feedback",
            "epochs": 3,
            "checkpoint_interval": 100,
            "output_checkpoint_stem": out_stem,
        }

        # Update global training controller
        get_training_controller().start(jid, f"Feedback Training {timestamp}")

        def run_feedback_training():
            try:
                from domains.training.train_pipeline import SloughGPTTrainer

                trainer = SloughGPTTrainer(
                    data_path=data_path,
                    n_embed=256,
                    n_layer=6,
                    n_head=8,
                    block_size=256,
                    epochs=3,
                    batch_size=16,
                    lr=1e-4,
                    use_lora=True,
                    lora_rank=8,
                    lora_alpha=16,
                    checkpoint_dir="models",
                    checkpoint_interval=100,
                    use_mixed_precision=True,
                )

                def on_progress(
                    step: int, epoch: int, loss: Optional[float], loss_type: str = "train"
                ):
                    training_jobs[jid]["progress"] = min(99, int((epoch / 3) * 100))
                    training_jobs[jid]["current_epoch"] = epoch
                    if loss is not None:
                        training_jobs[jid][loss_type] = float(loss)

                result = trainer.train(on_progress=on_progress)
                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
                trainer.save(f"models/{safe_stem}.pt")

                training_jobs[jid]["status"] = "completed"
                training_jobs[jid]["progress"] = 100
                training_jobs[jid]["checkpoint"] = f"models/{safe_stem}.pt"
                training_jobs[jid]["samples_used"] = count
                get_training_controller().complete()

                # Trigger webhook notification (fire and forget)
                try:
                    import asyncio

                    asyncio.get_event_loop().run_until_complete(
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
                except Exception:
                    pass

            except Exception as e:
                logger.exception("Feedback training job %s failed", jid)
                training_jobs[jid]["status"] = "failed"
                training_jobs[jid]["error"] = str(e)
                get_training_controller().fail(str(e))

                # Trigger webhook notification
                try:
                    import asyncio

                    asyncio.get_event_loop().run_until_complete(
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
                except Exception:
                    pass

        thread = threading.Thread(target=run_feedback_training, daemon=True)
        thread.start()

        return {
            "status": "started",
            "job_id": jid,
            "samples": count,
            "data_path": str(sft_path),
            "message": "Training started from feedback data",
        }

    except Exception as e:
        logger.exception("Failed to start feedback training")
        raise HTTPException(status_code=500, detail=str(e))

    return job


# ===== TRAINING STATE CONTROLLER =====


@router.get("/training/status")
async def get_training_status():
    """
    Get comprehensive training system status.

    Returns current state (idle/running/paused), current job info,
    and statistics about completed/failed jobs.
    """
    controller = get_training_controller()
    status = controller.get_status()

    # Also include any running jobs from the job registry
    running_jobs = [
        {"id": jid, "name": job.get("name"), "progress": job.get("progress", 0)}
        for jid, job in training_jobs.items()
        if job.get("status") == "running"
    ]

    status["running_jobs"] = running_jobs
    status["total_tracked_jobs"] = len(training_jobs)

    return status


@router.post("/training/control/start")
async def control_start_training():
    """
    Request to start training.

    Returns success/failure with current state.
    Note: Actual training start happens via POST /training/start
    """
    controller = get_training_controller()

    if controller.is_running():
        return {
            "success": False,
            "message": "Training is already running",
            **controller.get_status(),
        }

    if controller.is_paused():
        return {
            "success": False,
            "message": "Training is paused. Use /training/control/resume to continue.",
            **controller.get_status(),
        }

    return {
        "success": True,
        "message": "Ready to start training",
        **controller.get_status(),
    }


@router.post("/training/control/pause")
async def control_pause_training():
    """
    Pause current training.

    Pauses the training loop if running.
    """
    controller = get_training_controller()
    result = controller.pause()

    # Notify the training job if it's listening
    if result["success"]:
        logger.info("Training pause requested")

    return result


@router.post("/training/control/resume")
async def control_resume_training():
    """
    Resume paused training.

    Continues training from where it was paused.
    """
    controller = get_training_controller()
    result = controller.resume()

    if result["success"]:
        logger.info("Training resumed")

    return result


@router.post("/training/control/stop")
async def control_stop_training():
    """
    Stop current training.

    Gracefully stops the training job.
    """
    controller = get_training_controller()
    result = controller.stop()

    # Update all running jobs to stopping
    if result["success"]:
        for jid, job in training_jobs.items():
            if job.get("status") == "running":
                job["status"] = "stopping"
        logger.info("Training stop requested")

    return result


@router.post("/training/control/reset")
async def control_reset_training():
    """
    Reset training controller to idle state.

    Use after training completes or fails to clear state.
    """
    controller = get_training_controller()
    return controller.reset()


@router.get("/training/is-running")
async def is_training_running():
    """
    Quick check if training is currently running.

    Useful for UI to conditionally show controls.
    """
    controller = get_training_controller()
    return {
        "is_running": controller.is_running(),
        "is_paused": controller.is_paused(),
        "is_idle": controller.is_idle(),
        "state": controller.state.value,
        "current_job": controller.current_job_id,
    }


# ===== WEBHOOK NOTIFICATIONS =====


@router.get("/training/webhooks")
async def list_webhooks():
    """
    List all registered webhooks.
    """
    store = get_webhook_store()
    webhooks = store.list()

    return {
        "webhooks": [
            {
                "id": w.id,
                "url": w.url,
                "events": w.events,
                "description": w.description,
                "is_active": w.is_active,
                "created_at": w.created_at.isoformat(),
            }
            for w in webhooks
        ],
        "available_events": TRAINING_EVENTS,
    }


@router.post("/training/webhooks")
async def register_webhook(
    url: str,
    events: str,  # JSON stringified array
    description: str = "",
    secret: Optional[str] = None,
):
    """
    Register a new webhook endpoint.

    Args:
        url: The URL to send notifications to
        events: JSON stringified list of events (e.g., '["training.completed","training.failed"]')
        description: Optional description
        secret: Optional HMAC secret (generated if not provided)
    """
    # Parse events from JSON string
    import json

    try:
        events_list = json.loads(events) if isinstance(events, str) else events
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid events format. Must be JSON array.")

    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Validate events
    invalid_events = [e for e in events_list if e not in TRAINING_EVENTS]
    if invalid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid events: {invalid_events}. Available: {TRAINING_EVENTS}",
        )

    store = get_webhook_store()
    webhook_id = store.register(
        url=url,
        events=events_list,
        secret=secret,
        description=description,
        headers=None,
    )

    webhook = store.get(webhook_id)

    return {
        "id": webhook_id,
        "url": url,
        "events": events,
        "secret": webhook.secret if webhook else None,
        "message": "Webhook registered successfully",
    }


@router.delete("/training/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Unregister a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    store.unregister(webhook_id)

    return {"status": "deleted", "webhook_id": webhook_id}


@router.get("/training/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    """Get webhook details (without secret)."""
    store = get_webhook_store()
    webhook = store.get(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {
        "id": webhook.id,
        "url": webhook.url,
        "events": webhook.events,
        "description": webhook.description,
        "is_active": webhook.is_active,
        "created_at": webhook.created_at.isoformat(),
    }


@router.get("/training/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(webhook_id: str, limit: int = 50):
    """Get delivery log for a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")

    deliveries = store.get_deliveries(webhook_id, limit=limit)

    return {
        "deliveries": [
            {
                "id": d.id,
                "event": d.event,
                "success": d.success,
                "status_code": d.status_code,
                "attempted_at": d.attempted_at.isoformat(),
                "error": d.error,
            }
            for d in deliveries
        ]
    }


@router.get("/training/webhooks/stats")
async def get_webhook_stats():
    """Get webhook statistics."""
    store = get_webhook_store()
    return store.get_stats()


class TestWebhookRequest(BaseModel):
    url: str

@router.post("/training/webhooks/test")
async def test_webhook(req: TestWebhookRequest):
    """
    Send a test notification to a URL.

    Useful for verifying webhook setup.
    """
    store = get_webhook_store()

    # Register temporary webhook for test
    webhook_id = store.register(
        url=req.url,
        events=TRAINING_EVENTS,
        description="Temporary test webhook",
    )

    # Send test event
    delivery = await store.deliver(
        webhook_id=webhook_id,
        event="training.completed",
        payload={
            "job_id": "test",
            "job_name": "Test Training",
            "status": "completed",
            "message": "This is a test webhook notification",
        },
        retries=1,
    )

    # Clean up
    store.unregister(webhook_id)

    return {
        "success": delivery.success,
        "status_code": delivery.status_code,
        "error": delivery.error,
        "response_body": delivery.response_body,
    }


# ===== JOB RECOVERY =====


@router.get("/recovery/check")
async def check_crashed_jobs(timeout_seconds: int = 300):
    """
    Check for jobs that may have crashed.

    Jobs that are 'running' but haven't sent a heartbeat in timeout_seconds
    are considered potentially crashed.
    """
    store = get_job_store()
    crashed = store.detect_crashed_jobs(timeout_seconds)

    return {
        "detected_crashes": len(crashed),
        "jobs": crashed,
        "message": f"Found {len(crashed)} potentially crashed job(s)",
    }


@router.get("/recovery/recoverable")
async def get_recoverable_jobs():
    """
    Get all jobs that can be recovered.

    Includes interrupted and crashed jobs.
    """
    store = get_job_store()
    jobs = store.get_recoverable_jobs()

    return {
        "count": len(jobs),
        "jobs": jobs,
    }


@router.post("/recovery/recover/{job_id}")
async def recover_job(job_id: str):
    """
    Recover and restart an interrupted/crashed job.

    Resumes training from the last checkpoint if available.
    """
    store = get_job_store()
    job = store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ("interrupted", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', only 'interrupted' or 'failed' jobs can be recovered",
        )

    # Get config and checkpoint
    config = job.get("config", {})
    data_path = job.get("data_path", "")
    checkpoint_path = job.get("checkpoint_path", "")
    checkpoint_dir = job.get("checkpoint_dir", "checkpoints")
    job_name = job.get("name", "recovered_job")

    # Find checkpoint
    from pathlib import Path

    if checkpoint_path and Path(checkpoint_path).exists():
        pass  # Use existing checkpoint_path
    else:
        # Try to find any checkpoint in the checkpoint dir
        checkpoint_dir_path = Path(checkpoint_dir)
        if checkpoint_dir_path.exists():
            checkpoints = list(checkpoint_dir_path.glob("step_*.pt")) + list(
                checkpoint_dir_path.glob("*.pt")
            )
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                checkpoint_path = str(latest)

    # Create recovery job in training_jobs
    recovery_job_id = f"recovery_{job_id}"
    recovery_job = {
        "id": recovery_job_id,
        "name": f"Recovered: {job_name}",
        "model": config.get("model", "sloughgpt"),
        "dataset": job.get("dataset", ""),
        "data_path": data_path,
        "status": "running",
        "progress": job.get("progress", 0),
        "current_epoch": job.get("current_epoch", 0),
        "global_step": job.get("global_step", 0),
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
        "original_job_id": job_id,
        **config,
    }
    training_jobs[recovery_job_id] = recovery_job

    # Update job store
    store.update(job_id, status="recovering", crashed=0)

    # Update controller
    controller = get_training_controller()
    controller.start(recovery_job_id, f"Recovered: {job_name}")

    # Start recovery in background thread
    jid = recovery_job_id
    checkpoint_for_recovery = checkpoint_path

    def run_recovery():
        try:
            from domains.training.train_pipeline import SloughGPTTrainer

            trainer_config = {
                "data_path": recovery_job.get("data_path", ""),
                "epochs": recovery_job.get("epochs", 10),
                "batch_size": recovery_job.get("batch_size", 32),
                "lr": recovery_job.get("learning_rate", 1e-3),
                "n_embed": recovery_job.get("n_embed", 256),
                "n_layer": recovery_job.get("n_layer", 6),
                "n_head": recovery_job.get("n_head", 8),
                "block_size": recovery_job.get("block_size", 128),
                "checkpoint_dir": recovery_job.get("checkpoint_dir", "checkpoints"),
                "checkpoint_interval": recovery_job.get("checkpoint_interval", 500),
            }

            def on_progress(info: dict):
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                store.update_progress(
                    jid, rec["progress"], epoch=rec["current_epoch"], step=rec["global_step"]
                )

            trainer = SloughGPTTrainer(**trainer_config)

            # Resume from checkpoint if available
            result = trainer.train(
                on_progress=on_progress,
                resume=True,
                resume_path=checkpoint_for_recovery,
            )

            # Mark as completed
            training_jobs[jid]["status"] = "completed"
            training_jobs[jid]["progress"] = 100
            store.mark_completed(
                jid, checkpoint_for_recovery or trainer_config["checkpoint_dir"] + "/final.pt"
            )
            store.update(job_id, status="recovered")
            controller.complete()

            # Trigger webhook
            try:
                import asyncio

                asyncio.get_event_loop().run_until_complete(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": jid,
                            "job_name": training_jobs[jid].get("name"),
                            "status": "completed",
                            "recovered_from": job_id,
                        },
                    )
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            training_jobs[jid]["status"] = "failed"
            training_jobs[jid]["error"] = str(e)
            store.mark_failed(jid, str(e))
            controller.fail()

    thread = threading.Thread(target=run_recovery, daemon=True)
    thread.start()

    return {
        "status": "recovered",
        "original_job_id": job_id,
        "recovery_job_id": recovery_job_id,
        "checkpoint_path": checkpoint_path,
        "message": f"Recovery started. Training restarting from checkpoint: {checkpoint_path or 'beginning'}",
    }


@router.delete("/recovery/abandon/{job_id}")
async def abandon_recovery(job_id: str):
    """
    Abandon a crashed job and mark it as permanently failed.
    """
    store = get_job_store()
    job = store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    store.update(job_id, status="abandoned")

    return {
        "status": "abandoned",
        "job_id": job_id,
        "message": "Job marked as abandoned",
    }


@router.get("/recovery/stats")
async def get_recovery_stats():
    """Get recovery statistics."""
    store = get_job_store()
    stats = store.get_stats()

    return {
        **stats,
        "crashed_jobs": store.detect_crashed_jobs().__len__(),
        "recoverable_jobs": len(store.get_recoverable_jobs()),
    }
