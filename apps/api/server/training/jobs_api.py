"""Training job management endpoints — list, get, stop, delete, purge, export."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from domains.training.executor import get_training_executor
from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import raise_error

from .jobs import training_jobs

logger = logging.getLogger("slo")

router = APIRouter(tags=["training", "jobs"])


def _job_summary(job: dict) -> dict:
    """Build a human-readable job summary, stripping internal (underscore) fields."""
    status = job.get("status", "unknown")
    model = job.get("model", "unknown")
    dataset = job.get("dataset", "")
    progress = job.get("progress", 0)

    if status == "completed":
        status_message = f"Training complete! Model saved to {job.get('checkpoint', 'unknown')}"
    elif status == "running":
        status_message = f"Training {model} on {dataset}, {progress}% done"
    elif status == "failed":
        status_message = f"Training failed: {job.get('error', 'unknown error')}"
    elif status == "queued":
        status_message = f"Training {model} on {dataset} is queued"
    elif status == "stopping":
        status_message = f"Stopping training {model}..."
    else:
        status_message = f"Training status: {status}"

    summary = {
        "id": job.get("id"),
        "name": job.get("name"),
        "status": status,
        "status_message": status_message,
        "model": model,
        "dataset": dataset,
        "progress": progress,
        "current_epoch": job.get("current_epoch"),
        "epochs": job.get("epochs"),
        "global_step": job.get("global_step"),
        "loss": job.get("loss"),
        "train_loss": job.get("train_loss"),
        "total_steps": job.get("total_steps"),
        "steps_per_sec": job.get("steps_per_sec"),
        "eta_s": job.get("eta_s"),
        "elapsed_s": job.get("elapsed_s"),
        "avg_quality": job.get("avg_quality"),
        "eval_loss": job.get("eval_loss"),
        "checkpoint": job.get("checkpoint"),
        "checkpoint_dir": job.get("checkpoint_dir"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }
    return {k: v for k, v in summary.items() if v is not None}


@router.get("/training/jobs")
async def list_training_jobs():
    """List all tracked training jobs with plain-language status."""
    return [_job_summary(j) for j in training_jobs.values()]


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get one training job by id with plain-language status."""
    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    return _job_summary(training_jobs[job_id])


@router.post("/training/jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    """Stop a specific training job by id."""
    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    job = training_jobs[job_id]
    if job.get("status") not in ("running", "queued", "starting"):
        raise_error(
            f"Job is not running (status: {job.get('status', 'unknown')})",
            "E_BAD_REQUEST",
            status_code=400,
        )
    prev_status = job.get("status", "unknown")
    job["status"] = "stopping"
    cancel_event = job.get("_cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    executor = get_training_executor()
    executor.cancel(job_id)
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager

        get_cancel_manager().cancel(job_id)
    except Exception as exc:
        logger.debug("cancel_manager.cancel failed: %s", exc)
    try:
        from infrastructure.auth import get_audit_logger

        get_audit_logger().log("training.stop", resource=job_id, detail=f"from={prev_status}")
    except Exception as exc:
        logger.debug("audit log failed: %s", exc)
    return {"status": "stopping", "job_id": job_id}


@router.get("/training/jobs/{job_id}/summary")
async def get_training_summary(job_id: str):
    """Plain-language summary of a training job."""
    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    job = training_jobs[job_id]

    model = job.get("model", "unknown model")
    dataset = job.get("dataset", "unknown dataset")
    status = job.get("status", "unknown")
    checkpoint = job.get("checkpoint", "")
    epochs = job.get("epochs")
    current_epoch = job.get("current_epoch")
    final_loss = job.get("loss")
    rl = bool(job.get("reward_history"))

    lines = []

    if status == "completed":
        lines.append(f"You trained {model} on {dataset}.")
        if epochs:
            lines.append(f"It learned from {epochs} passes over your data.")
        if final_loss is not None:
            if final_loss < 1.5:
                lines.append(f"Loss is low ({final_loss:.2f}) — your AI learned well.")
            elif final_loss < 3.0:
                lines.append(
                    f"Loss is moderate ({final_loss:.2f}) — your AI learned something, but could do better."
                )
            else:
                lines.append(
                    f"Loss is high ({final_loss:.2f}) — your AI may need more data or more training."
                )
        if rl:
            lines.append(
                "Personality reinforcement was applied — your AI learned to give better answers."
            )
        if checkpoint:
            lines.append(f"Your trained model is at: {checkpoint}")
            lines.append("Load it in the Models page to use it in chat.")
        else:
            lines.append("The model was saved but the checkpoint path isn't available yet.")
    elif status == "running":
        progress = job.get("progress", 0)
        lines.append(f"Training {model} on {dataset}... {progress}% done.")
        if current_epoch is not None and epochs:
            lines.append(f"Epoch {current_epoch} of {epochs}.")
    elif status == "failed":
        error = job.get("error", "Unknown error")
        lines.append(f"Training failed: {error}")
        lines.append("Try using a smaller model, or check that you have enough disk space.")
    elif status == "queued":
        lines.append(f"Training {model} on {dataset} is queued. It will start shortly.")
    else:
        lines.append(f"Training status: {status}")

    return {
        "job_id": job_id,
        "summary": " ".join(lines),
        "status": status,
        "model": model,
        "dataset": dataset,
    }


@router.delete("/training/jobs/{job_id}")
async def delete_training_job(job_id: str, auth_user: dict = Depends(require_auth_if_enabled)):
    """Delete a training job and optionally its checkpoint files."""
    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
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

    try:
        from infrastructure.auth import get_audit_logger

        get_audit_logger().log(
            "training.delete",
            resource=job_id,
            detail=f"deleted_files={len(deleted_files)}",
        )
    except Exception as exc:
        logger.debug("audit log failed: %s", exc)

    return {
        "status": "deleted",
        "job_id": job_id,
        "deleted_files": deleted_files,
    }


@router.post("/training/jobs/purge")
async def purge_training_jobs(
    status: str | None = Query(None, description="Only purge jobs in this terminal status"),
    auth_user: dict = Depends(require_auth_if_enabled),
):
    """Remove all terminal training jobs from the in-memory tracker."""
    terminal = {"completed", "failed", "cancelled", "interrupted"}
    if status:
        if status not in terminal:
            raise_error(
                f"Cannot purge non-terminal status '{status}'",
                "E_INVALID_STATUS",
                status_code=400,
            )
        terminal = {status}

    before = len(training_jobs)
    to_remove = [jid for jid, j in training_jobs.items() if j.get("status") in terminal]
    for jid in to_remove:
        del training_jobs[jid]
    purged = before - len(training_jobs)

    return {
        "status": "purged",
        "purged": purged,
        "remaining": len(training_jobs),
        "statuses_purged": sorted(terminal),
    }


@router.get("/training/export/{job_id}")
async def export_training_job(job_id: str):
    """Export a completed training job's checkpoint file."""
    from fastapi.responses import FileResponse

    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    job = training_jobs[job_id]

    if job.get("status") not in ("completed", "failed", "cancelled"):
        raise_error("Job must be completed before export", "E_BAD_REQUEST", status_code=400)
    checkpoint = job.get("checkpoint")
    if not checkpoint:
        raise_error("No checkpoint found for this job", "E_NOT_FOUND", status_code=404)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise_error("Checkpoint file not found on disk", "E_NOT_FOUND", status_code=404)
    return FileResponse(
        path=checkpoint_path,
        filename=checkpoint_path.name,
        media_type="application/octet-stream",
    )


class ExportTextRequest(BaseModel):
    """Request model for /training/export-text."""

    min_quality: float = Field(default=0, ge=0.0, le=1.0)
    target_count: int = Field(default=100, ge=1, le=10000)


@router.post("/training/export-text")
async def export_feedback_pairs(request: ExportTextRequest):
    """Export feedback conversation pairs with a minimum quality threshold."""
    from controllers.feedback import get_feedback_controller

    ctrl = get_feedback_controller()
    pairs = []
    feedback_file = ctrl.feedback_dir / "feedback.jsonl"
    if feedback_file.exists():
        with open(feedback_file) as f:
            for line in f:
                try:
                    fb = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if fb.get("user_message") and fb.get("assistant_response"):
                    pairs.append(fb)
                    if len(pairs) >= request.target_count:
                        break
    output_file = ctrl.feedback_dir / f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(pairs, f, indent=2)
    return {"pairs_count": len(pairs), "file": str(output_file), "status": "exported"}
