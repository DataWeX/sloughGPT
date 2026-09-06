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

from domains.shared import find_repo_root
from domains.training.executor import get_training_executor
from fastapi import APIRouter, Depends, Request
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error

from .control import router as control_router
from .controller import get_training_controller
from .execution import router as execution_router
from .helpers import _finish_job, _run_async, _sloughgpt_trainer_kwds
from .job_store import get_job_store
from .jobs import training_jobs
from .jobs_api import router as jobs_router
from .webhooks import (
    notify_training_event,
)

logger = logging.getLogger("slo")

router = APIRouter(tags=["training"])

# Include sub-module routers
router.include_router(execution_router)
router.include_router(jobs_router)
router.include_router(control_router)


def _finetuned_dir() -> Path:
    """Absolute path to the HF fine-tuned models directory."""
    return find_repo_root(Path(__file__).resolve()) / "models" / "hf-finetuned"


def _write_finetuned_metadata(
    model_dir: str | Path,
    model: str,
    dataset: str,
    final_loss: float | None = None,
    epochs: int | None = None,
) -> None:
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


_finetuned_models_cache: tuple[float, dict] | None = None
_FINETUNED_CACHE_TTL = 30.0


@router.get("/training/finetuned-models")
async def list_finetuned_models():
    """List HF fine-tuned model directories under ``models/hf-finetuned/``."""
    global _finetuned_models_cache
    base = _finetuned_dir()
    now = time.monotonic()
    if _finetuned_models_cache and (now - _finetuned_models_cache[0]) < _FINETUNED_CACHE_TTL:
        return _finetuned_models_cache[1]
    models = []
    if base.is_dir():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            meta = _read_finetuned_metadata(d)
            display_name = d.name.split("_")[0].replace("--", "/")
            models.append(
                {
                    "name": d.name,
                    "model_path": str(d),
                    "size_mb": round(size_bytes / (1024 * 1024), 1),
                    "size_bytes": size_bytes,
                    "created_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                    "model": meta.get("model") or display_name,
                    "dataset": meta.get("dataset")
                    or (d.name.split("_")[1] if "_" in d.name else ""),
                    "model_name": d.name,
                    "final_loss": meta.get("final_loss"),
                    "epochs": meta.get("epochs") or 0,
                }
            )
    result = {"models": models}
    _finetuned_models_cache = (now, result)
    return result


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
        raise_error(
            result.get("error", "Failed to load fine-tuned model"),
            "E_INFRA_STARTUP",
            status_code=500,
        )
    return {
        "status": "loaded",
        "name": name,
        "model_path": str(target),
        "model_id": result.get("model_id"),
    }


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
            "E_BAD_REQUEST",
            status_code=400,
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
                "E_VAL_REQUEST",
                status_code=422,
            )
        try:
            resume_bundle = CheckpointManager.load_from_path(checkpoint_path)
        except Exception as exc:
            raise_error(
                f"Cannot resume from '{checkpoint_path}': checkpoint is unreadable ({exc})",
                "E_VAL_REQUEST",
                status_code=422,
            )
        if resume_bundle is None:
            raise_error(
                f"Cannot resume from '{checkpoint_path}': checkpoint missing or "
                "unsupported (use a .soul or .npz file)",
                "E_VAL_REQUEST",
                status_code=422,
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
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

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
                    rec.setdefault("loss_history", []).append(
                        {
                            "step": int(info.get("global_step", 0)),
                            "value": float(tl),
                            "type": "train",
                        }
                    )
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe
                    rec.setdefault("loss_history", []).append(
                        {"step": int(info.get("global_step", 0)), "value": fe, "type": "eval"}
                    )
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
            recovery_checkpoint = (
                getattr(trainer, "_last_checkpoint_path", None) or checkpoint_for_recovery
            )
            training_jobs[jid]["checkpoint_path"] = recovery_checkpoint
            training_jobs[jid]["checkpoint"] = recovery_checkpoint
            store.mark_completed(job_id, recovery_checkpoint or "")
            controller.complete()

            # Trigger webhook
            try:
                _run_async(
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
            except Exception as e:
                logger.warning("Recovery webhook failed for %s: %s", jid, e)

        except FileNotFoundError as e:
            logger.error("Recovery failed (missing file): %s", e, extra={"tag": "TRAIN"})
            _finish_job(jid, "failed", f"Data file not found: {e}")
            store.mark_failed(job_id, f"Data file not found: {e}")
            controller.fail()
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


# ── Unified /training/* → core service + infrastructure wrapping ──────────────
# Business logic: domains.training.service (pure, no HTTP)
# HTTP wrapping: success_response, error classification, audit logging (this file)


@router.get("/training/log")
async def training_log():
    from domains.training.service import get_log
    from schemas.common import success_response

    lines = await get_log()
    return success_response(data={"lines": lines, "total": len(lines)})


@router.post("/training/stop")
async def training_stop():
    """Stop all active training (auto-train, turbo, from-sessions)."""
    from schemas.common import success_response

    from .sse_stream import stop_all_training

    result = stop_all_training()
    return success_response(data=result)


@router.get("/training/checkpoints")
async def training_list_checkpoints():
    from domains.training.service import list_checkpoints
    from schemas.common import success_response

    checkpoints = await list_checkpoints()
    return success_response(data=checkpoints)


@router.delete("/training/checkpoints/{name}")
async def training_delete_checkpoint(name: str):
    from domains.training.service import delete_checkpoint
    from schemas.common import safe_audit_log, success_response

    deleted = await delete_checkpoint(name)
    if deleted:
        safe_audit_log("training.checkpoint.delete", resource=name, detail="deleted")
    return success_response(data={"deleted": deleted, "name": name})


@router.post("/training/checkpoints/{name}/load")
async def training_load_checkpoint(name: str):
    from domains.training.service import load_checkpoint
    from schemas.common import classify_and_raise, success_response

    try:
        result = await load_checkpoint(name)
        return success_response(data=result, message="loaded")
    except Exception as e:
        classify_and_raise(e, source="training.load_checkpoint")


@router.get("/training/checkpoints/{name}/download")
async def training_download_checkpoint(name: str):
    from domains.training.service import download_checkpoint_path
    from fastapi.responses import FileResponse
    from schemas.common import raise_error

    fp = await download_checkpoint_path(name)
    if fp:
        return FileResponse(fp, media_type="application/octet-stream", filename=name)
    raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)


@router.get("/training/checkpoints/{name}/info")
async def training_checkpoint_info(name: str):
    from domains.training.service import checkpoint_info
    from schemas.common import classify_and_raise, success_response

    try:
        info = await checkpoint_info(name)
        return success_response(data=info)
    except Exception as e:
        classify_and_raise(e, source="training.checkpoint_info")


@router.get("/training/metrics/export")
async def training_metrics_export():
    import json as _json

    from domains.training.service import get_all_checkpoint_data
    from fastapi.responses import Response

    checkpoints = await get_all_checkpoint_data()
    export = {
        "exported_at": time.time(),
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }
    content = _json.dumps(export, indent=2, default=str)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=training-metrics.json"},
    )


@router.get("/training/from-sessions/cancel")
async def training_cancel_from_sessions():
    """Cancel session-based training."""
    from schemas.common import success_response

    from .sse_stream import cancel_from_sessions

    result = cancel_from_sessions()
    return success_response(data=result)


@router.get("/training/stream")
async def training_stream(request: Request):
    """SSE training stream."""
    from pathlib import Path

    from .sse_stream import build_training_sse_response

    # Get auto_train state for config
    try:
        import routers.auto_train as at

        config = at.state.config or {}
        if not config:
            from schemas.common import success_response

            return success_response(
                data={
                    "status": "error",
                    "error": "No training state",
                    "code": "E_STATE_IDLE",
                    "http_status": 409,
                }
            )
    except ImportError:
        from schemas.common import success_response

        return success_response(
            data={
                "status": "error",
                "error": "auto_train not available",
                "code": "E_STATE_IDLE",
                "http_status": 409,
            }
        )

    checkpoints_dir = Path(__file__).resolve().parents[2] / "models" / "auto-training"
    return await build_training_sse_response(
        request,
        config,
        task_name="auto-train",
        task_type="training",
        cm_label="auto-train",
        checkpoints_dir=checkpoints_dir,
    )


@router.get("/training/from-sessions-stream")
async def training_from_sessions_stream(request: Request):
    """SSE stream from sessions."""
    from pathlib import Path

    from .sse_stream import build_training_sse_response

    try:
        import routers.auto_train as at

        config = at.state.config or {}
        if not config or config.get("method") != "from-sessions":
            from schemas.common import success_response

            return success_response(
                data={
                    "status": "error",
                    "error": "No from-sessions state",
                    "code": "E_STATE_IDLE",
                    "http_status": 409,
                }
            )
    except ImportError:
        from schemas.common import success_response

        return success_response(
            data={
                "status": "error",
                "error": "auto_train not available",
                "code": "E_STATE_IDLE",
                "http_status": 409,
            }
        )

    checkpoints_dir = Path(__file__).resolve().parents[2] / "models" / "auto-training"
    return await build_training_sse_response(
        request,
        config,
        task_name="auto-train-sessions",
        task_type="training-sessions",
        cm_label="auto-train-sessions",
        runtime_data_source="from-sessions",
        checkpoints_dir=checkpoints_dir,
    )


# Include sub-routers
from .turbo_endpoints import router as _turbo_router
from .webhook_endpoints import router as _webhook_router

router.include_router(_turbo_router)
router.include_router(_webhook_router)
