"""FastAPI routes for char-level training and job orchestration.

Trainer ``*.soul`` checkpoint charset maps: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from infrastructure.auth import require_auth_if_enabled, audit_user, get_audit_logger
from schemas.common import raise_error

try:
    from domains.api.sse_envelope import sse_event, sse_error, sse_complete
except ImportError:
    def sse_event(stream, phase, status, data=None, meta=None, message=""):
        import json as _j
        return "data: " + _j.dumps({
            "stream": stream, "phase": phase, "status": status,
            "data": data or {}, "meta": meta or {}, "message": message
        }) + "\n\n"
    def sse_error(stream, phase, error, meta=None):
        return sse_event(stream, phase, "error", {"error": error}, meta or {}, f"Error: {error}")
    def sse_complete(stream, phase="COMPLETE", data=None, meta=None, message="Done"):
        return sse_event(stream, phase, "complete", data or {}, meta or {}, message)

from .jobs import training_jobs
from .resolution import resolve_training_inputs
from .schemas import TrainingRequest, TrainRequest, TrainResolveRequest, DistillStartRequest, LoraFinetuneRequest
from .controller import get_training_controller, TrainingState
from .webhooks import (
    get_webhook_store,
    TRAINING_EVENTS,
    notify_training_event,
)
from .job_store import get_job_store
from domains.training.executor import get_training_executor
from domains.shared import find_repo_root
from domains.mobile.notifications import get_notification_service

logger = logging.getLogger("slo")

router = APIRouter(tags=["training"])


def _finish_job(job_id: str, status: str, error: str | None = None) -> None:
    """Set job status and notify CancelManager so operations store stays in sync."""
    job = training_jobs.get(job_id)
    if job is not None:
        job["status"] = status
        if error:
            job["error"] = error
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpStatus
        mgr = get_cancel_manager()
        op = mgr.get(job_id)
        if op is not None and op.status not in (
            OpStatus.CANCELLED, OpStatus.COMPLETED, OpStatus.FAILED,
        ):
            mgr.finish(job_id, error=error or "")
    except Exception as exc:
        logger.warning("_finish_job CancelManager.finish failed for %s: %s", job_id, exc)


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Add plain-language status message to a training job dict.

    Translates raw numbers into human-readable descriptions:
    - "Training... 60% done, about 2 minutes left"
    - "Training complete! Model saved to models/..."
    - "Training failed: CUDA out of memory"
    """
    summary = {k: v for k, v in job.items() if not k.startswith("_")}
    status = job.get("status", "unknown")
    progress = job.get("progress", 0)
    model = job.get("model", "")
    dataset = job.get("dataset", "")
    method = job.get("data_source", "")
    explanation = job.get("explanation", "")

    if status == "running":
        parts = [f"Training {model} on {dataset}"]
        if progress > 0:
            parts.append(f"{progress}% done")
        epoch = job.get("current_epoch")
        epochs = job.get("epochs")
        if epoch is not None and epochs:
            parts.append(f"epoch {epoch}/{epochs}")
        summary["status_message"] = ", ".join(parts) + "..."
    elif status == "completed":
        if explanation:
            summary["status_message"] = explanation
        else:
            summary["status_message"] = f"Training complete! Model: {model}"
    elif status == "failed":
        error = job.get("error", "Unknown error")
        summary["status_message"] = f"Training failed: {error}"
    elif status == "queued":
        summary["status_message"] = f"Queued: {model} on {dataset}"
    elif status == "stopping":
        summary["status_message"] = "Stopping..."
    else:
        summary["status_message"] = f"Status: {status}"

    return summary


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

    ``SloughGPTTrainer`` writes periodic ``<dataset>_<timestamp>.soul`` checkpoints under
    ``checkpoint_dir`` with ``stoi`` / ``itos`` / ``chars`` for char-LM eval; see
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
        raise_error(str(e), "E_BAD_REQUEST", status_code=400)

    req_snapshot = request.model_dump()

    def train_model(job_id: str) -> None:
        try:
            training_jobs[job_id]["status"] = "running"
            trainer = SloughGPTTrainer(
                data_path=data_path_str,
                **_sloughgpt_trainer_kwds(req_snapshot),
            )
            if not cancel_event.is_set():
                trainer.train(cancel_event=cancel_event)
            else:
                _finish_job(job_id, "cancelled", "Cancelled before start")
                return
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            trainer.save(f"models/{safe_stem}_trained.soul")
            _finish_job(job_id, "completed")
            training_jobs[job_id]["checkpoint"] = f"models/{safe_stem}_trained.soul"
        except Exception as e:
            logger.exception("Background /train failed: %s", e, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(e))

    executor = get_training_executor()
    job_id = f"train_{int(time.time())}"
    cancel_event = threading.Event()
    training_jobs[job_id] = {
        "status": "queued",
        "data_path": data_path_str,
        "output_checkpoint_stem": out_stem,
        "epochs": request.epochs,
        "_cancel_event": cancel_event,
    }

    # Register with CancelManager
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _mgr = get_cancel_manager()
        op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=f"train:{out_stem}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(op_id)
        training_jobs[job_id]["_cancel_manager_op_id"] = op_id
    except Exception as exc:
        logger.warning("CancelManager registration failed for %s: %s", job_id, exc)

    executor.submit(train_model, job_id)

    out: dict[str, Any] = {
        "status": "started",
        "job_id": job_id,
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

    Does not write checkpoint artifacts. After ``POST /train`` or ``POST /training/start``,
    native ``*.soul`` checkpoints include char vocab; see ``docs/policies/CONTRIBUTING.md``
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
        raise_error(str(e), "E_BAD_REQUEST", status_code=400)

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


@router.get("/training/jobs")
async def list_training_jobs():
    """List all tracked training jobs with plain-language status.

    Each job includes a ``status_message`` field with a human-readable
    description of what's happening: "Training... 60% done" or
    "Training complete! Model saved to models/...".
    """
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
        raise_error(f"Job is not running (status: {job.get('status', 'unknown')})", "E_BAD_REQUEST", status_code=400)
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
    """Plain-language summary of a training job.

    Returns what was trained, what data was used, how it went,
    and what to do next.  No ML jargon — just facts Alex can use.
    """
    if job_id not in training_jobs:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    job = training_jobs[job_id]

    model = job.get("model", "unknown model")
    dataset = job.get("dataset", "unknown dataset")
    status = job.get("status", "unknown")
    explanation = job.get("explanation", "")
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
                lines.append(f"Loss is moderate ({final_loss:.2f}) — your AI learned something, but could do better.")
            else:
                lines.append(f"Loss is high ({final_loss:.2f}) — your AI may need more data or more training.")
        if rl:
            lines.append("Personality reinforcement was applied — your AI learned to give better answers.")
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
    """Delete a training job and optionally its checkpoint files.

    Removes job from registry. If ``delete_files`` is true, removes checkpoint
    files associated with the job from disk.
    """
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
    """Remove all terminal training jobs from the in-memory tracker.

    By default purges jobs with status completed, failed, cancelled, or
    interrupted.  Pass ``?status=completed`` to purge only completed jobs.
    Only terminal statuses are accepted — running/queued jobs cannot be purged.
    """
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
async def start_training(request: TrainingRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    """Start a tracked training job (web UI).

    ``*.soul`` files saved on the server include ``stoi`` / ``itos`` / ``chars``
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
        raise_error(str(e), "E_BAD_REQUEST", status_code=400)

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
        "loss_history": [],
    }
    if manifest_meta is not None:
        job["manifest"] = manifest_meta
    training_jobs[job_id] = job

    # Update global training controller
    controller = get_training_controller()
    controller.start(job_id, request.name or "training")

    # Audit trail — training job start (char-level fine-tune)
    try:
        from infrastructure.auth import get_audit_logger
        get_audit_logger().log(
            "training.start",
            resource=request.dataset.strip() if request.dataset else out_stem,
            detail="char",
            extra={"job_id": job_id, "model": request.model, "epochs": request.epochs, "source_kind": source_kind},
        )
    except Exception as exc:
        logger.debug("audit log failed: %s", exc)

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
    except Exception as e:
        logger.debug("Training webhook notification failed: %s", e, extra={"tag": "TRAIN"})

    req_snapshot = request.model_dump()
    data_path_for_thread = data_path_str
    out_stem_for_thread = out_stem
    jid = job_id
    cancel_event = threading.Event()
    training_jobs[job_id]["_cancel_event"] = cancel_event
    pause_event = threading.Event()
    training_jobs[job_id]["_pause_event"] = pause_event
    from training.runtime import get_training_runtime
    get_training_runtime().register(job_id, training_jobs[job_id], cancel_event, req_snapshot)

    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(req_snapshot.get("name") or job_id),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "slonet"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception:
        pass

    def run_training(job_id_: str = jid) -> None:
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
                rec["total_steps"] = int(info.get("total_steps", rec.get("total_steps", 0)))
                rec["steps_per_sec"] = info.get("steps_per_sec", rec.get("steps_per_sec"))
                rec["eta_s"] = info.get("eta_s", rec.get("eta_s"))
                rec["elapsed_s"] = info.get("elapsed_s", rec.get("elapsed_s"))
                aq = info.get("avg_quality")
                if aq is not None:
                    rec["avg_quality"] = float(aq)
                tl = info.get("train_loss")
                if tl is not None:
                    rec["train_loss"] = float(tl)
                    rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": float(tl), "type": "train"})
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe
                    rec.setdefault("loss_history", []).append({"step": rec.get("global_step", 0), "value": fe, "type": "eval"})
                get_training_runtime().sync(jid)

            trainer = SloughGPTTrainer(
                data_path=data_path_for_thread,
                **_sloughgpt_trainer_kwds(req_snapshot),
                experiment_tracker=tracker,
            )
            result = trainer.train(
                on_progress=on_progress,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )
            safe_stem = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in out_stem_for_thread
            )[:120]
            if cancel_event.is_set():
                _finish_job(jid, "cancelled")
                training_jobs[jid]["progress"] = 0
                get_training_runtime().sync(jid)
                get_training_controller().complete()
                return
            trainer.save(f"models/{safe_stem}_trained.soul")
            _finish_job(jid, "completed")
            training_jobs[jid]["progress"] = 100
            training_jobs[jid]["current_epoch"] = int(req_snapshot.get("epochs") or 3)
            bel = result.get("best_eval_loss")
            training_jobs[jid]["loss"] = bel if bel is not None and bel < float("inf") else None
            training_jobs[jid]["checkpoint"] = f"models/{safe_stem}_trained.soul"
            get_training_runtime().sync(jid)
            get_training_controller().complete()

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.run(
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
            except Exception as e:
                logger.debug("Training completion webhook failed: %s", e, extra={"tag": "TRAIN"})

            # Push notification to mobile devices
            try:
                loss = training_jobs[jid].get("loss")
                loss_str = f"Final loss: {loss:.4f}" if loss is not None else "Training"
                get_notification_service().send_notification_sync(
                    title="Training Complete",
                    body=f"{loss_str}. Checkpoint saved.",
                    data={"screen": "Training", "job_id": jid},
                    topics=["training"],
                )
            except Exception as e:
                logger.debug("Training completion push failed: %s", e, extra={"tag": "TRAIN"})
        except Exception as e:
            logger.exception("Training job %s failed", jid, extra={"tag": "TRAIN"})
            _finish_job(jid, "failed", str(e))
            training_jobs[jid]["progress"] = 0
            get_training_runtime().sync(jid)
            get_training_controller().fail(str(e))

            # Trigger webhook notification (fire and forget)
            try:
                import asyncio

                asyncio.run(
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

            # Push notification to mobile devices
            try:
                get_notification_service().send_notification_sync(
                    title="Training Failed",
                    body=f"Job {training_jobs[jid].get('name', jid)} failed: {str(e)[:100]}",
                    data={"screen": "Training", "job_id": jid},
                    topics=["training"],
                )
            except Exception:
                pass
        finally:
            if tracker is not None:
                try:
                    tracker.end_run()
                except Exception:
                    logger.exception("W&B end_run failed for job %s", jid, extra={"tag": "TRAIN"})

    executor = get_training_executor()
    executor.submit(run_training, jid)

    return {
        "status": "started",
        "job_id": job_id,
        "name": request.name or "training",
        "model": request.model,
        "dataset": request.dataset,
        "epochs": request.epochs,
    }


# ── Visual Training ────────────────────────────────────────────────


class VisualTrainRequest(BaseModel):
    dataset: str
    vision_encoder: str = "slonet"
    llm: str = "gpt2"
    connector_hidden_dim: int = 512
    max_seq_length: int = 128
    stage1_epochs: int = 5
    stage2_epochs: int = 10
    stage1_lr: float = 5e-4
    stage2_lr: float = 2e-4
    batch_size: int = 4
    use_lora: bool = False
    lora_rank: int = 8
    freeze_vision: bool = True
    name: str | None = None


@router.post("/training/visual-start")
async def start_visual_training(request: VisualTrainRequest):
    """Start a vision-language model (VLM) fine-tune on a visual dataset.

    Uses the video trainer's VideoCaptionTrainer on an image-caption
    dataset, running stage-1 connector tuning then stage-2 full fine-tune.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    datasets_dir = find_repo_root(Path(__file__).resolve()) / "datasets"
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
            label=f"vlm:{request.dataset}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(op_id)
        job["_cancel_manager_op_id"] = op_id
    except Exception as exc:
        logger.warning("CancelManager registration failed for visual training %s: %s", job_id, exc)

    def run_visual_training(job_id_: str = job_id) -> None:
        try:
            from domains.training.video_trainer import VideoCaptionTrainer

            trainer = VideoCaptionTrainer()
            trainer.build_vocab()

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(job_id)
                if not rec:
                    return
                rec["current_epoch"] = info.get("epoch", 0)
                rec["global_step"] = info.get("step", 0)
                loss = info.get("loss")
                if loss is not None:
                    rec["loss"] = float(loss)
                    rec.setdefault("loss_history", []).append(
                        {"step": info.get("step", 0), "value": float(loss)}
                    )
                rec["progress"] = info.get("progress", rec.get("progress", 0))

            result = trainer.train(
                data_path=data_path_str,
                epochs=request.stage1_epochs + request.stage2_epochs,
                batch_size=request.batch_size,
                learning_rate=request.stage1_lr,
                on_progress=on_progress,
                output_dir=str(output_dir),
            )

            _finish_job(job_id, "completed")
            training_jobs[job_id]["progress"] = 100
            training_jobs[job_id]["result"] = result

            # Save checkpoint
            ckpt_path = output_dir / f"{out_stem}.npz"
            trainer.save_checkpoint(str(ckpt_path))
            training_jobs[job_id]["checkpoint"] = str(ckpt_path)

        except Exception as exc:
            logger.exception("Visual training job %s failed", job_id, extra={"tag": "TRAIN"})
            if job_id in training_jobs:
                _finish_job(job_id, "failed", str(exc))

    executor = get_training_executor()
    executor.submit(run_visual_training, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Visual training started on {request.dataset}",
    }


# ── Distillation ────────────────────────────────────────────────────


@router.post("/training/distill")
async def start_distillation(request: DistillStartRequest):
    """Knowledge distillation: teach a compact SloNet LSTM student from a teacher HF model.

    Loads the teacher from the HF model registry, creates a SloNet LSTM student,
    runs DistillationTrainer.step() in a loop, saves the student as a checkpoint.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    datasets_dir = find_repo_root(Path(__file__).resolve()) / "datasets"
    data_path = datasets_dir / request.dataset
    if not data_path.exists():
        data_path = datasets_dir / f"{request.dataset}.jsonl"
    if not data_path.exists():
        raise_error(f"Dataset not found: {request.dataset}", "E_BAD_REQUEST", status_code=400)
    input_file = data_path / "input.txt" if data_path.is_dir() else data_path
    if data_path.is_dir():
        candidates = [data_path / "input.txt", data_path / "corpus.jsonl", data_path / "train.txt"]
        input_file = next((c for c in candidates if c.exists()), None)
    if not input_file or not Path(input_file).exists():
        raise_error("No training data file (input.txt/corpus.jsonl) in dataset", "E_BAD_REQUEST", status_code=400)
    data_str = Path(input_file).read_text(encoding="utf-8")
    if not data_str.strip():
        raise_error("Training data is empty", "E_BAD_REQUEST", status_code=400)
    out_stem = request.name or f"distill_{job_id}"
    _REPO_ROOT = find_repo_root(Path(__file__).resolve())
    output_dir = _REPO_ROOT / "models" / "auto-training"
    output_dir.mkdir(parents=True, exist_ok=True)

    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"Distill-{job_id}",
        "type": "distill",
        "status": "queued",
        "progress": 0,
        "epochs": request.epochs,
        "dataset": request.dataset,
        "teacher_model": request.teacher_model,
        "config": request.model_dump(),
    }
    training_jobs[job_id] = job
    cancel_event = threading.Event()
    training_jobs[job_id]["_cancel_event"] = cancel_event

    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(request.name or f"distill-{job_id}"),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "distill"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception:
        pass

    def _run_distill(job_id_: str = job_id):
        """Background thread that runs distillation."""
        try:
            training_jobs[job_id_]["status"] = "running"
            import numpy as np
            import random as _random
            _random.seed(42)
            np.random.seed(42)

            # Resolve the teacher: prefer the ModelServer (torch HF) from the
            # model registry; otherwise use the active SloNet provider that the
            # load path publishes into ServerState (pure NumPy). The former
            # ``ctrl._hf_model`` fallback is vestigial — it is never assigned,
            # only deleted on unload — so it cannot supply a teacher.
            from domains.infrastructure.model_registry import get_model_registry
            registry = get_model_registry()
            server = registry.get(request.teacher_model) if registry else None

            slonet_provider = None
            teacher_model = None
            teacher_tokenizer = None
            if server is not None:
                teacher_model = server._model_ref
                teacher_tokenizer = getattr(server, "_tokenizer", None)
            else:
                from domains.infrastructure.server_state import get_server_state
                provider = get_server_state().model.get()
                if (provider is not None
                        and getattr(provider, "model_id", None) == request.teacher_model):
                    slonet_provider = provider
                    teacher_model = getattr(provider, "_get_model", lambda: None)()
                    if teacher_model is not None:
                        teacher_tokenizer = provider.tokenize

            if teacher_model is None:
                _finish_job(job_id, "failed", f"Teacher model '{request.teacher_model}' not loaded")
                return

            # Tokenize training data
            if teacher_tokenizer is not None:
                if slonet_provider is not None:
                    tokens = teacher_tokenizer(data_str[:100000])
                else:
                    tokens = teacher_tokenizer.encode(data_str[:100000])
            else:
                tokens = [ord(c) for c in data_str[:100000]]

            # Build vocab from teacher or data
            import numpy as np
            import random as _random
            _random.seed(42)
            np.random.seed(42)

            vocab = sorted(set(tokens))
            stoi = {c: i for i, c in enumerate(vocab)}
            itos = {i: c for c, i in stoi.items()}
            vocab_size = len(stoi)

            # Create the student — the project's native NumPy transformer.
            # SloughGPTModel accepts numpy input_ids and returns
            # ``(logits, loss)``, matching the DistillationTrainer boundary.
            # (The earlier ``SloNet(vocab_size=...)`` call never matched a
            # real constructor — this route was unreachable before.)
            from domains.models import SloughGPTModel

            student = SloughGPTModel(
                vocab_size=vocab_size,
                n_embed=request.embed_dim,
                n_layer=request.n_layers,
                n_head=request.n_heads or 4,
                block_size=request.block_size,
            )

            # Prepare data for training
            block_size = request.block_size
            token_ids = [stoi.get(t, 0) for t in tokens]
            inputs_list = []
            targets_list = []
            for i in range(0, len(token_ids) - block_size, block_size // 2):
                x = token_ids[i:i + block_size]
                y = token_ids[i + 1:i + block_size + 1]
                if len(x) == block_size and len(y) == block_size:
                    inputs_list.append(x)
                    targets_list.append(y)

            if not inputs_list:
                _finish_job(job_id, "failed", "Not enough data for training")
                return

            inputs_np = np.array(inputs_list, dtype=np.int64)
            targets_np = np.array(targets_list, dtype=np.int64)
            n_samples = len(inputs_np)
            batch_size = min(16, n_samples)

            # Create teacher inputs wrapper
            class _TeacherWrapper:
                """Expose a teacher forward pass to the DistillationTrainer.

                Two teacher backends:
                - SloNet model (slonet=True): pure NumPy forward pass
                  ``forward(input_ids, targets=None) -> (logits, loss)``.
                - torch HF model (registry ModelServer): lazy torch interop.
                """
                def __init__(self, model, tokenizer, slonet=False):
                    self._model = model
                    self._tokenizer = tokenizer
                    self._slonet = slonet
                def parameters(self):
                    return []
                def eval(self):
                    pass
                def __call__(self, x):
                    import numpy as np
                    if isinstance(x, np.ndarray):
                        if self._slonet:
                            logits_t, _ = self._model.forward(x.astype(np.int64), None)
                            out_np = np.asarray(logits_t.data, dtype=np.float64)[..., :vocab_size]
                            return np.squeeze(out_np, 0) if out_np.shape[0] == 1 else out_np
                        raise RuntimeError("Torch teacher models are not supported — use SloNet")
                    return np.zeros((x.shape[0], vocab_size), dtype=np.float32)

            teacher_wrapper = _TeacherWrapper(teacher_model, teacher_tokenizer, slonet=slonet_provider is not None)

            from domains.training.distillation import DistillationTrainer, DistillationConfig
            distill_cfg = DistillationConfig(
                temperature=request.temperature,
                alpha=request.alpha,
                beta=request.beta,
            )
            trainer = DistillationTrainer(teacher_wrapper, student, distill_cfg)

            # Training loop
            epoch_losses = []
            for epoch in range(request.epochs):
                if cancel_event.is_set():
                    _finish_job(job_id, "cancelled")
                    return
                indices = list(range(n_samples))
                _random.shuffle(indices)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    bx = inputs_np[batch_idx]
                    by = targets_np[batch_idx]

                    losses = trainer.step(bx, by)
                    batch_loss = losses.get("total_loss", 0.0)
                    epoch_loss += batch_loss
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                epoch_losses.append(avg_loss)

                training_jobs[job_id]["progress"] = int((epoch + 1) / request.epochs * 100)
                training_jobs[job_id]["current_epoch"] = epoch + 1
                training_jobs[job_id]["train_loss"] = avg_loss

            # Save student checkpoint
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
            ckpt_path = output_dir / f"{safe_stem}_distilled.soul"

            from domains.training.slonet import export_to_sou

            export_to_sou(student, str(ckpt_path), metadata={
                "model_type": "slonet_distill",
                "teacher": request.teacher_model,
                "distill_temperature": request.temperature,
                "epochs": request.epochs,
                "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
                "embed_dim": request.embed_dim,
                "n_layers": request.n_layers,
                "vocab_size": vocab_size,
                "stoi": stoi,
                "itos": itos,
            })

            training_jobs[job_id].update({
                "progress": 100,
                "loss": float(epoch_losses[-1]) if epoch_losses else None,
                "checkpoint": str(ckpt_path),
                "loss_history": [
                    {"step": i, "value": v, "type": "train"}
                    for i, v in enumerate(epoch_losses)
                ],
            })
            _finish_job(job_id, "completed")
            logger.info("Distillation complete: %s loss=%.4f", ckpt_path, epoch_losses[-1] if epoch_losses else 0, extra={"tag": "TRAIN"})

        except Exception as e:
            logger.exception("Distillation job %s failed", job_id, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(e))

    executor = get_training_executor()
    executor.submit(_run_distill, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Distillation started: teacher={request.teacher_model} epochs={request.epochs}",
    }


# ── LoRA Fine-tuning ──────────────────────────────────────────────


@router.post("/training/lora-finetune")
async def start_lora_finetune(request: LoraFinetuneRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    """LoRA fine-tuning on .slnc models using SloNet numpy autograd (no PyTorch).

    Trains low-rank adapters on top of a .slnc model. The adapter is saved
    as a .npz file alongside the base model. Uses HFLoraTrainer from
    domains.training.hf_lora_finetune.
    """
    import uuid

    job_id = f"lora_{uuid.uuid4().hex[:8]}"

    # Validate model path
    model_path = Path(request.model_path)
    if not model_path.is_file():
        repo_root = find_repo_root(Path(__file__).resolve())
        alt_path = repo_root / "models" / request.model_path
        if alt_path.is_file():
            model_path = alt_path
        else:
            raise_error(f"Model not found: {request.model_path}. Provide a .slnc file path.", "E_BAD_REQUEST", status_code=400)
    # Validate dataset
    repo_root = find_repo_root(Path(__file__).resolve())
    datasets_dir = repo_root / "datasets"
    data_dir = datasets_dir / request.dataset
    data_path = None
    if data_dir.is_dir():
        for candidate in ["input.txt", "corpus.jsonl", "train.txt"]:
            p = data_dir / candidate
            if p.is_file():
                data_path = p
                break
    if data_path is None:
        raise_error(f"Dataset not found: {request.dataset}. Use POST /datasets/import/local first.", "E_BAD_REQUEST", status_code=400)
    model_stem = model_path.stem
    dataset_name = request.dataset.strip() if request.dataset else data_path.stem

    # Create job record — matches TrainingJob interface
    job: dict[str, Any] = {
        "id": job_id,
        "name": request.name or f"LoRA-{dataset_name}-r{request.rank}",
        "model": model_stem,
        "dataset": dataset_name,
        "data_path": str(data_path),
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "global_step": 0,
        "total_steps": 0,
        "steps_per_sec": None,
        "eta_s": None,
        "elapsed_s": None,
        "loss": None,
        "train_loss": None,
        "eval_loss": None,
        "loss_history": [],
        "rank": request.rank,
        "alpha": request.alpha,
        "error": None,
        "result": None,
        "checkpoint": None,
    }
    training_jobs[job_id] = job

    # Update global training controller
    controller = get_training_controller()
    controller.start(job_id, request.name or f"LoRA-{dataset_name}")

    # Audit trail
    try:
        from infrastructure.auth import get_audit_logger
        get_audit_logger().log(
            "training.start",
            resource=dataset_name,
            detail="lora",
            extra={"job_id": job_id, "model": model_stem, "rank": request.rank, "epochs": request.epochs},
        )
    except Exception:
        pass

    # Webhook notification
    try:
        import asyncio

        async def notify_async():
            await notify_training_event(
                "training.started",
                {
                    "job_id": job_id,
                    "job_name": request.name or f"LoRA-{dataset_name}",
                    "dataset": dataset_name,
                    "epochs": request.epochs,
                    "method": "lora",
                    "rank": request.rank,
                },
            )

        asyncio.create_task(notify_async())
    except Exception as e:
        logger.debug("LoRA training webhook notification failed: %s", e, extra={"tag": "TRAIN"})

    # Cancel event for stop support
    cancel_event = threading.Event()
    training_jobs[job_id]["_cancel_event"] = cancel_event

    # Register with runtime for consistency
    try:
        from training.runtime import get_training_runtime
        get_training_runtime().register(job_id, training_jobs[job_id], cancel_event, request.model_dump())
    except Exception:
        pass

    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(request.model_dump().get("dataset") or job_id),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "lora"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception:
        pass

    def run_lora_finetune(job_id_: str = job_id):
        try:
            from domains.training.hf_lora_finetune import HFLoraTrainer, HFLoraConfig

            start_time = time.time()

            def on_progress(info: dict[str, Any]) -> None:
                rec = training_jobs.get(job_id)
                if not rec:
                    return
                elapsed = time.time() - start_time
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("step", rec.get("global_step", 0)))
                loss = info.get("loss")
                if loss is not None:
                    rec["train_loss"] = float(loss)
                    rec["loss"] = float(loss)
                    rec.setdefault("loss_history", []).append({
                        "step": rec["global_step"],
                        "value": float(loss),
                        "type": "train",
                    })
                rec["elapsed_s"] = elapsed
                step = rec["global_step"]
                if step > 0 and elapsed > 0:
                    rec["steps_per_sec"] = step / elapsed
                    epochs_left = max(0, (rec.get("epochs", 1) - rec["current_epoch"]) / max(rec["current_epoch"], 1))
                    rec["eta_s"] = elapsed * epochs_left
                # Compute progress from epochs
                total_epochs = rec.get("epochs", 1)
                if total_epochs > 0:
                    rec["progress"] = min(99, int((rec["current_epoch"] / total_epochs) * 100))

            config = HFLoraConfig(
                model_path=str(model_path),
                data_path=str(data_path),
                rank=request.rank,
                alpha=request.alpha,
                dropout=request.dropout,
                target_modules=request.target_modules,
                epochs=request.epochs,
                batch_size=request.batch_size,
                block_size=request.block_size,
                learning_rate=request.learning_rate,
                warmup_steps=request.warmup_steps,
                weight_decay=request.weight_decay,
                grad_clip=request.grad_clip,
                grad_accumulation_steps=request.grad_accumulation_steps,
                log_interval=request.log_interval,
                output_dir=request.output_dir,
                adapter_name=request.adapter_name,
                _cancel_event=cancel_event,
            )

            trainer = HFLoraTrainer(config)
            result = trainer.train(on_progress=on_progress)

            if cancel_event.is_set():
                _finish_job(job_id, "cancelled")
                training_jobs[job_id]["progress"] = 0
                get_training_controller().complete()
                return

            training_jobs[job_id].update({
                "progress": 100,
                "current_epoch": result.epochs_completed or request.epochs,
                "loss": result.final_loss,
                "train_loss": result.final_loss,
                "result": {
                    "adapter_path": result.model_path,
                    "total_steps": result.total_steps,
                    "final_loss": result.final_loss,
                    "epochs_completed": result.epochs_completed,
                },
                "checkpoint": result.model_path,
            })
            _finish_job(job_id, "completed")

            # Sync runtime
            try:
                from training.runtime import get_training_runtime
                get_training_runtime().sync(job_id)
            except Exception:
                pass

            get_training_controller().complete()

            logger.info(
                "LoRA fine-tune complete: %s loss=%.4f",
                result.model_path, result.final_loss or 0,
                extra={"tag": "TRAIN"},
            )

            # Webhook notification
            try:
                asyncio.run(
                    notify_training_event(
                        "training.completed",
                        {
                            "job_id": job_id,
                            "job_name": request.name or f"LoRA-{dataset_name}",
                            "dataset": dataset_name,
                            "final_loss": result.final_loss,
                            "adapter_path": result.model_path,
                        },
                    )
                )
            except Exception:
                pass

        except Exception as exc:
            logger.exception("LoRA fine-tune job %s failed", job_id, extra={"tag": "TRAIN"})
            _finish_job(job_id, "failed", str(exc))
            get_training_controller().complete()

    executor = get_training_executor()
    executor.submit(run_lora_finetune, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"LoRA fine-tune started: rank={request.rank} epochs={request.epochs} dataset={dataset_name}",
    }


class LoadAdapterRequest(BaseModel):
    """Request to load a LoRA adapter into the running model."""
    adapter_path: str = Field(description="Path to .npz adapter file")
    merge: bool = Field(default=False, description="Merge LoRA into base weights for faster inference")


def _resolve_adapter_path(raw: str) -> Path:
    """Resolve adapter path from user input, checking repo-relative paths."""
    p = Path(raw)
    if p.is_file():
        return p
    repo_root = find_repo_root(Path(__file__).resolve())
    for base in [repo_root / "models", repo_root / "data" / "user_adapters"]:
        alt = base / raw
        if alt.is_file():
            return alt
    raise_error(f"Adapter not found: {raw}", "E_BAD_REQUEST", status_code=400)


@router.post("/training/load-adapter")
async def load_adapter(request: LoadAdapterRequest):
    """Load a LoRA adapter into the currently running model for inference.

    Sends the adapter to the worker subprocess which applies LoRA to the model
    in-process. Supports SloNet models via domains.training.lora.
    """
    adapter_path = _resolve_adapter_path(request.adapter_path)

    # Find the ProcessGuard — stored in the models controller (adopted during autoload)
    process_guard = None
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        process_guard = getattr(ctrl, '_process_guard', None)
    except Exception:
        pass

    if process_guard is None or not hasattr(process_guard, 'load_adapter'):
        raise_error("No subprocess worker found. Load adapter via direct model access instead.", "E_BAD_REQUEST", status_code=400)
    try:
        result = process_guard.load_adapter(
            str(adapter_path), merge=request.merge, timeout=120.0
        )
        logger.info("Loaded adapter via worker: %s", adapter_path.name, extra={"tag": "TRAIN"})
        return result
    except Exception as exc:
        logger.exception("Failed to load adapter: %s", exc, extra={"tag": "TRAIN"})
        raise_error(f"Failed to load adapter: {exc}", "E_INFRA_STARTUP", status_code=500)



@router.post("/training/unload-adapter")
async def unload_adapter():
    """Unload the current LoRA adapter and revert to base model weights.

    Sends the unload command to the worker subprocess which reloads the base
    model in-process.
    """
    process_guard = None
    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        process_guard = getattr(ctrl, '_process_guard', None)
    except Exception:
        pass

    if process_guard is None or not hasattr(process_guard, 'unload_adapter'):
        raise_error("No subprocess worker found. Unload adapter via direct model access instead.", "E_BAD_REQUEST", status_code=400)
    try:
        result = process_guard.unload_adapter(timeout=60.0)
        logger.info("Unloaded adapter via worker", extra={"tag": "TRAIN"})
        return result
    except Exception as exc:
        logger.exception("Failed to unload adapter: %s", exc, extra={"tag": "TRAIN"})
        raise_error(f"Failed to unload adapter: {exc}", "E_INFRA_STARTUP", status_code=500)



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
            "epochs": 3,
            "checkpoint_interval": 100,
            "output_checkpoint_stem": out_stem,
            "_cancel_event": cancel_event,
        }

        # Register with CancelManager
        try:
            from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
            _mgr = get_cancel_manager()
            op_id = _mgr.register(
                op_type=OpType.TRAINING,
                label=f"feedback:{out_stem}",
                cancel_fn=lambda: cancel_event.set(),
            )
            _mgr.start(op_id)
            training_jobs[jid]["_cancel_manager_op_id"] = op_id
        except Exception as exc:
            logger.warning("CancelManager registration failed for feedback training %s: %s", jid, exc)

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
                    epochs=3,
                    batch_size=16,
                    lr=1e-4,
                    use_lora=True,
                    lora_rank=8,
                    lora_alpha=16,
                    checkpoint_dir="models",
                    checkpoint_interval=100,
                )

                def on_progress(info: dict) -> None:
                    training_jobs[jid]["progress"] = min(99, int((info.get("progress_percent", 0))))
                    training_jobs[jid]["current_epoch"] = int(info.get("epoch", training_jobs[jid].get("current_epoch", 0)))
                    tl = info.get("train_loss")
                    if tl is not None:
                        training_jobs[jid]["train_loss"] = float(tl)
                        training_jobs[jid].setdefault("loss_history", []).append({"step": info.get("global_step", 0), "value": float(tl), "type": "train"})
                    el = info.get("eval_loss")
                    if el is not None:
                        training_jobs[jid]["eval_loss"] = float(el)
                        training_jobs[jid]["loss"] = float(el)
                        training_jobs[jid].setdefault("loss_history", []).append({"step": info.get("global_step", 0), "value": float(el), "type": "eval"})

                result = trainer.train(on_progress=on_progress)
                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_stem)[:120]
                trainer.save(f"models/{safe_stem}.soul")

                _finish_job(jid, "completed")
                training_jobs[jid]["progress"] = 100
                training_jobs[jid]["checkpoint"] = f"models/{safe_stem}.soul"
                training_jobs[jid]["samples_used"] = count
                get_training_controller().complete()

                # Trigger webhook notification (fire and forget)
                try:
                    import asyncio

                    asyncio.run(
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
                    import asyncio

                    asyncio.run(
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

        executor = get_training_executor()
        executor.submit(run_feedback_training, jid)

        return {
            "status": "started",
            "job_id": jid,
            "samples": count,
            "data_path": str(sft_path),
            "message": "Training started from feedback data",
        }

    except Exception as e:
        logger.exception("Failed to start feedback training", extra={"tag": "TRAIN"})
        raise_error(str(e), "E_INFRA_STARTUP", status_code=500)

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


def _signal_current_job(pause: bool | None = None, cancel: bool = False) -> dict[str, Any]:
    """Signal the controller's current job with cooperative control events.

    Sets/clears the job's ``_pause_event`` and sets ``_cancel_event`` so the
    running ``SloughGPTTrainer`` loop actually pauses or stops. Returns the
    signals that were applied (empty when there is no tracked job).
    """
    controller = get_training_controller()
    jid = getattr(controller, "current_job_id", None)
    if not jid:
        return {}
    job = training_jobs.get(jid)
    if not job:
        return {}
    signaled: dict[str, Any] = {}
    if pause is not None:
        ev = job.get("_pause_event")
        if ev is not None:
            if pause:
                ev.set()
                signaled["pause"] = "requested"
            else:
                ev.clear()
                signaled["resume"] = "requested"
    if cancel:
        ev = job.get("_cancel_event")
        if ev is not None:
            ev.set()
            signaled["cancel"] = "requested"
        try:
            from domains.infrastructure.cancel_manager import get_cancel_manager
            get_cancel_manager().cancel(jid)
        except Exception:
            pass
    return signaled


@router.post("/training/control/pause")
async def control_pause_training():
    """
    Pause current training.

    Signals the running trainer's ``pause_event``; the loop sleeps at the next
    step until ``/training/control/resume`` clears it.
    """
    controller = get_training_controller()
    result = controller.pause()

    # Notify the training job if it's listening
    if result["success"]:
        signaled = _signal_current_job(pause=True)
        logger.info("Training pause requested: %s", signaled or "no tracked job",
            extra={"tag": "TRAIN"})

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
        signaled = _signal_current_job(pause=False)
        logger.info("Training resumed: %s", signaled or "no tracked job",
            extra={"tag": "TRAIN"})

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
        signaled = _signal_current_job(cancel=True)
        logger.info("Training stop requested: %s", signaled or "no tracked job",
            extra={"tag": "TRAIN"})

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
    secret: str | None = None,
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
        raise_error("Invalid events format. Must be JSON array.", "E_BAD_REQUEST", status_code=400)
    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise_error("URL must start with http:// or https://", "E_BAD_REQUEST", status_code=400)
    # Validate events
    invalid_events = [e for e in events_list if e not in TRAINING_EVENTS]
    if invalid_events:
        raise_error(f"Invalid events: {invalid_events}. Available: {TRAINING_EVENTS}", "E_BAD_REQUEST", status_code=400)
    store = get_webhook_store()
    webhook_id = store.register(
        url=url,
        events=events_list,
        secret=secret,
        description=description,
        headers=None,
    )

    webhook = store.get(webhook_id)

    try:
        from infrastructure.auth import get_audit_logger
        get_audit_logger().log(
            "training.webhook.register",
            resource=url,
            extra={"webhook_id": webhook_id, "events": events_list},
        )
    except Exception:
        pass

    return {
        "id": webhook_id,
        "url": url,
        "events": events,
        "secret": webhook.secret if webhook else None,
        "message": "Webhook registered successfully",
    }


@router.get("/training/webhooks/stats")
async def get_webhook_stats():
    """Get webhook statistics."""
    store = get_webhook_store()
    return store.get_stats()


@router.delete("/training/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Unregister a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
    store.unregister(webhook_id)

    try:
        from infrastructure.auth import get_audit_logger
        get_audit_logger().log("training.webhook.delete", resource=webhook_id)
    except Exception:
        pass

    return {"status": "deleted", "webhook_id": webhook_id}


@router.get("/training/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    """Get webhook details (without secret)."""
    store = get_webhook_store()
    webhook = store.get(webhook_id)

    if not webhook:
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
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
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
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


@router.get("/training/webhooks/retry-queue")
async def get_webhook_retry_queue():
    """Get pending webhook retries."""
    store = get_webhook_store()
    return {"retries": store.get_retry_queue()}


@router.get("/training/webhooks/dead-letters")
async def get_webhook_dead_letters(limit: int = 50):
    """Get dead-lettered webhook deliveries."""
    store = get_webhook_store()
    return {"dead_letters": store.get_dead_letters(limit=limit)}


@router.get("/training/builds")
async def list_builds():
    """List all training builds (checkpoints + fine-tuned models + LoRA adapters).

    Combines:
      - ``GET /auto-train/checkpoints`` (SloNet checkpoints + LoRA .soul files)
      - Completed HF fine-tune jobs from ``training_jobs``
      - HF fine-tuned model directories under ``models/hf-finetuned/``
    """
    from routers.auto_train import _load_soul, _load_lora_soul
    _repo_root = find_repo_root(Path(__file__).resolve())
    _checkpoints_dir = _repo_root / "models" / "auto-training"
    _lora_dir = _repo_root / "data" / "user_adapters"
    _hf_finetuned_dir = _repo_root / "models" / "hf-finetuned"

    builds = []

    # 1. Auto-train checkpoints (.soul / .npz)
    seen = set()
    for ext in ("*.soul", "*.npz"):
        for f in sorted(_checkpoints_dir.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name in seen:
                continue
            seen.add(f.name)
            info = _load_soul(f.name)
            if info:
                info["build_type"] = "auto-train"
                builds.append(info)

    # 2. LoRA .soul files
    for npz in sorted(_lora_dir.glob("*.soul"), key=lambda p: p.stat().st_mtime, reverse=True):
        if npz.name in seen:
            continue
        seen.add(npz.name)
        info = _load_lora_soul(npz.name)
        if info:
            info["build_type"] = "lora"
            builds.append(info)

    # 3. Completed HF fine-tune jobs
    for jid, job in training_jobs.items():
        if job.get("status") == "completed":
            model_path = job.get("result", {}).get("model_path", "") if isinstance(job.get("result"), dict) else ""
            builds.append({
                "name": job.get("name") or jid,
                "build_type": "hf-finetune",
                "job_id": jid,
                "model": job.get("model", ""),
                "dataset": job.get("dataset", ""),
                "loss": job.get("loss"),
                "epochs": job.get("epochs"),
                "model_path": model_path,
                "created_at": job.get("started_at", ""),
                "finished_at": job.get("completed_at", ""),
            })

    # 4. HF fine-tuned model directories on disk (for builds not tracked in memory)
    if _hf_finetuned_dir.is_dir():
        for d in sorted(_hf_finetuned_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name not in seen:
                seen.add(d.name)
                config_path = d / "config.json"
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                builds.append({
                    "name": d.name,
                    "build_type": "hf-finetuned-dir",
                    "model_path": str(d),
                    "size_mb": round(size_mb, 1),
                    "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    "model": d.name.split("_")[0].replace("--", "/"),
                    "dataset": d.name.split("_")[1] if "_" in d.name else "",
                })

    return {"builds": builds}


def _finetuned_dir() -> Path:
    """Absolute path to the HF fine-tuned models directory."""
    return find_repo_root(Path(__file__).resolve()) / "models" / "hf-finetuned"


def _write_finetuned_metadata(model_dir: str | Path, model: str, dataset: str,
                              final_loss: float | None = None,
                              epochs: int | None = None) -> None:
    """Persist authoritative model/dataset metadata inside a fine-tuned model dir.

    The directory name is not a reliable source of truth: datasets can contain
    underscores and legacy quick-train runs omit the model prefix entirely.

    Args:
        model_dir: Fine-tuned model directory (``output_dir``)
        model: Base HuggingFace model id (e.g. ``gpt2``)
        dataset: Dataset name used for training (may be empty)
        final_loss: Final training loss if known
        epochs: Number of epochs trained if known

    Side effects:
        - Writes ``metadata.json`` into ``model_dir``
    """
    try:
        meta = {
            "model": model,
            "dataset": dataset,
            "final_loss": final_loss,
            "epochs": epochs,
        }
        (Path(model_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:  # metadata is best-effort, never fail the job
        logger.debug("Failed to write fine-tuned metadata: %s", e)


def _read_finetuned_metadata(model_dir: Path) -> dict[str, Any]:
    """Read persisted metadata from a fine-tuned model dir, or ``{}``.

    Args:
        model_dir: Fine-tuned model directory

    Returns:
        Dict with ``model``/``dataset``/``final_loss``/``epochs`` keys as present
    """
    try:
        return json.loads((model_dir / "metadata.json").read_text())
    except Exception:
        return {}



def _resolve_finetuned(name: str) -> Path:
    """Resolve a fine-tuned model name to its directory, guarding against path traversal."""
    base = _finetuned_dir().resolve()
    target = (base / name).resolve()
    if base not in target.parents:
        raise_error("Invalid fine-tuned model name", "E_BAD_REQUEST", status_code=400)
    if not target.is_dir():
        raise_error(f"Fine-tuned model not found: {name}", "E_NOT_FOUND", status_code=404)
    return target


@router.get("/training/finetuned-models")
async def list_finetuned_models():
    """List HF fine-tuned model directories under ``models/hf-finetuned/``."""
    base = _finetuned_dir()
    models = []
    if base.is_dir():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            meta = _read_finetuned_metadata(d)
            display_name = d.name.split("_")[0].replace("--", "/")
            models.append({
                "name": d.name,
                "model_path": str(d),
                "size_mb": round(size_bytes / (1024 * 1024), 1),
                "size_bytes": size_bytes,
                "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                "model": meta.get("model") or display_name,
                "dataset": meta.get("dataset") or (d.name.split("_")[1] if "_" in d.name else ""),
                # Legacy keys consumed by the shell `finetuned` command table.
                "model_name": d.name,
                "final_loss": meta.get("final_loss"),
                "epochs": meta.get("epochs") or 0,
            })
    return {"models": models}


@router.post("/training/finetuned-models/{name}/load")
async def load_finetuned_model(name: str):
    """Load a fine-tuned model directory into chat (device=cpu).

    Compiles the local fine-tuned model to .slnc on first load, then registers
    it as the ``slonet-native`` provider via ``ModelsController.load_model_path``.
    """
    target = _resolve_finetuned(name)
    from controllers.models import get_models_controller
    result = get_models_controller().load_model_path(str(target), "cpu", identity=name)
    if result.get("status") != "loaded":
        raise_error(result.get("error", "Failed to load fine-tuned model"), "E_INFRA_STARTUP", status_code=500)
    return {"status": "loaded", "name": name, "model_path": str(target),
            "model_id": result.get("model_id")}


@router.delete("/training/finetuned-models/{name}")
async def delete_finetuned_model(name: str, auth_user: dict = Depends(require_auth_if_enabled)):
    """Delete a fine-tuned model directory."""
    target = _resolve_finetuned(name)
    shutil.rmtree(str(target))
    return {"status": "deleted", "name": name}


# ===== JOB RECOVERY =====


@router.get("/recovery/check")
async def check_crashed_jobs(timeout_seconds: int = 300):
    """
    Check for jobs that may have crashed.

    Jobs that are 'running' (or 'recovering') but haven't sent a heartbeat in
    timeout_seconds are considered potentially crashed.
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
    Recover and restart an interrupted/failed job.

    Also accepts a 'recovering' job whose heartbeat went stale (a previous
    recovery run that died without completing). Resumes training from the last
    checkpoint if available.
    """
    store = get_job_store()
    job = store.get(job_id)

    if not job:
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
    if job["status"] not in ("interrupted", "failed") and not (
        job["status"] == "recovering" and store.is_stale_heartbeat(job)
    ):
        raise_error(
            f"Job status is '{job['status']}', only 'interrupted' or 'failed' jobs "
            "(or a 'recovering' job with a stale heartbeat) can be recovered",
            "E_BAD_REQUEST", status_code=400,
        )
    # Get config and checkpoint. The store's checkpoint_dir column is only
    # written on completion, so interrupted jobs have it NULL — the job's
    # stored request config is the authoritative source for the scan directory.
    config = job.get("config", {})
    data_path = job.get("data_path", "")
    checkpoint_path = job.get("checkpoint_path", "")
    checkpoint_dir = job.get("checkpoint_dir") or config.get("checkpoint_dir") or "checkpoints"
    job_name = job.get("name", "recovered_job")

    # Resolve the checkpoint to resume from. A job's recorded path is loaded
    # exactly once, here in the request handler. The job explicitly points at
    # it, so a recorded path that is missing, unsupported, or unreadable fails
    # the recovery request loudly (422) — resuming from a different checkpoint
    # would silently change what is being resumed. Only when the job recorded
    # NO checkpoint path do we fall back to the newest file that actually loads
    # (skipping partial/corrupt checkpoints — a crash mid-write can leave the
    # newest file unreadable). The loaded bundle is handed to train() so no
    # second load happens in the worker thread.
    from domains.training.train_pipeline import CheckpointManager

    manager = CheckpointManager(checkpoint_dir)
    resume_bundle = None
    if checkpoint_path:
        if not CheckpointManager.is_resumable(checkpoint_path):
            raise_error(
                f"Cannot resume from '{checkpoint_path}': checkpoint missing or "
                "unsupported (use a .soul or .npz file)",
                "E_VAL_REQUEST", status_code=422,
            )
        try:
            resume_bundle = CheckpointManager.load_from_path(checkpoint_path)
        except Exception as exc:
            raise_error(f"Cannot resume from '{checkpoint_path}': checkpoint is unreadable ({exc})", "E_VAL_REQUEST", status_code=422)
        if resume_bundle is None:
            raise_error(
                f"Cannot resume from '{checkpoint_path}': checkpoint missing or "
                "unsupported (use a .soul or .npz file)",
                "E_VAL_REQUEST", status_code=422,
            )
    else:
        checkpoint_path, resume_bundle = manager.load_latest_with_path()
        checkpoint_path = checkpoint_path or ""

    # Create recovery job in training_jobs. config is spread first so the
    # explicit control fields (id, status, checkpoint_*) always win over any
    # matching keys persisted in the original request config.
    recovery_job_id = f"recovery_{job_id}"
    recovery_job = {
        **config,
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
    }
    training_jobs[recovery_job_id] = recovery_job

    # Update job store — fresh heartbeat so an actively-recovered row is never
    # mistaken for crashed or still-recoverable while the run is alive.
    store.mark_recovering(job_id)

    # Update controller
    controller = get_training_controller()
    controller.start(recovery_job_id, f"Recovered: {job_name}")

    # Start recovery in background thread
    jid = recovery_job_id
    checkpoint_for_recovery = checkpoint_path
    cancel_event = threading.Event()
    pause_event = threading.Event()
    recovery_job["_cancel_event"] = cancel_event
    recovery_job["_pause_event"] = pause_event

    # Register with CancelManager
    try:
        from domains.infrastructure.cancel_manager import get_cancel_manager, OpType
        _mgr = get_cancel_manager()
        op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=f"recover:{job_name}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(op_id)
        recovery_job["_cancel_manager_op_id"] = op_id
    except Exception as exc:
        logger.warning("CancelManager registration failed for recovery training %s: %s", jid, exc)

    def run_recovery(job_id_: str = jid):
        try:
            from domains.training.train_pipeline import SloughGPTTrainer

            # Reuse the SAME trainer configuration builder as /training/start so
            # the recovered run continues with the original job's hyperparameters
            # (LoRA, dropout, scheduler, device, ...) instead of a fixed subset.
            trainer_config = {
                "data_path": recovery_job.get("data_path", ""),
                **_sloughgpt_trainer_kwds(recovery_job),
            }

            def on_progress(info: dict):
                rec = training_jobs.get(jid)
                if not rec:
                    return
                rec["progress"] = int(info.get("progress_percent", rec.get("progress", 0)))
                rec["current_epoch"] = int(info.get("epoch", rec.get("current_epoch", 0)))
                rec["global_step"] = int(info.get("global_step", 0))
                rec["total_steps"] = int(info.get("total_steps", rec.get("total_steps", 0)))
                rec["steps_per_sec"] = info.get("steps_per_sec", rec.get("steps_per_sec"))
                rec["eta_s"] = info.get("eta_s", rec.get("eta_s"))
                rec["elapsed_s"] = info.get("elapsed_s", rec.get("elapsed_s"))
                aq = info.get("avg_quality")
                if aq is not None:
                    rec["avg_quality"] = float(aq)
                tl = info.get("train_loss")
                if tl is not None:
                    rec["train_loss"] = float(tl)
                    rec.setdefault("loss_history", []).append({"step": int(info.get("global_step", 0)), "value": float(tl), "type": "train"})
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe
                    rec.setdefault("loss_history", []).append({"step": int(info.get("global_step", 0)), "value": fe, "type": "eval"})
                store.update_progress(
                    job_id, rec["progress"], epoch=rec["current_epoch"], step=rec["global_step"]
                )

            trainer = SloughGPTTrainer(**trainer_config)

            # Resume from checkpoint if available (bundle pre-loaded once in the
            # request handler — no disk load happens in this thread)
            trainer.train(
                on_progress=on_progress,
                resume=True,
                resume_checkpoint=resume_bundle,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )

            # Mark as completed. The recovery job lives in-memory only; the
            # persistent original row is the durable record, so terminal state
            # and the produced checkpoint path are written to it.
            if cancel_event.is_set():
                _finish_job(jid, "cancelled")
                training_jobs[jid]["progress"] = 0
                store.update(job_id, status="interrupted")
                controller.complete()
                return

            _finish_job(jid, "completed")
            training_jobs[jid]["progress"] = 100
            recovery_checkpoint = getattr(trainer, "_last_checkpoint_path", None) or checkpoint_for_recovery
            training_jobs[jid]["checkpoint_path"] = recovery_checkpoint
            training_jobs[jid]["checkpoint"] = recovery_checkpoint
            store.mark_completed(job_id, recovery_checkpoint or "")
            controller.complete()

            # Trigger webhook
            try:
                import asyncio

                asyncio.run(
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
            logger.error("Recovery failed: %s", e, extra={"tag": "TRAIN"})
            _finish_job(jid, "failed", str(e))
            store.mark_failed(job_id, str(e))
            controller.fail()

    executor = get_training_executor()
    executor.submit(run_recovery, jid)

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
        raise_error("Job not found", "E_NOT_FOUND", status_code=404)
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
