"""FastAPI routes for char-level training and job orchestration.

Trainer ``*.soul`` checkpoint charset maps: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).

Descriptor-projected (card 20261104_151036): 27 ``ToolSpec`` contracts →
``create_router`` projections instead of hand-laid decorators. Envelope split
byte-parity: the 13 handlers that wrapped in ``success_response`` had their
inner wrap stripped (group A — the projection's wrap emits identical
``{status, data}`` bytes); the 14 bare-payload handlers keep raw wire
(``envelope=False``) — including the ``message="loaded"`` self-wrap, the
``FileResponse``/raw ``Response`` passthroughs and both SSE streams. The two
``Depends(require_auth_if_enabled)`` routes carry
``auth_scope="authenticated"``; the other 25 are public (edge 3: reproduce
the hand-built auth chain). Descriptions are derived from the handler
docstrings with ``inspect.cleandoc`` — the exact expression FastAPI used
on the hand-built routes, so they match by construction; summaries are
pinned explicitly (FastAPI derives them from ``route.name``, which for a
projected route would be the generated endpoint's name, not the handler's).
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import RouteSpec, create_router
from schemas.common import raise_error

from domain.agents import ToolParam, ToolSpec
from domain.shared import find_repo_root

from .control import router as control_router
from .controller import get_training_controller
from .execution import router as execution_router
from .feeds import router as feeds_router
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

# Include training data feed endpoints (live corpus over HTTP)
router.include_router(feeds_router)

# Include the live feed/job observability stream (SSE)
from .live_stream import router as live_stream_router

router.include_router(live_stream_router)


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


async def delete_finetuned_model(name: str, auth_user: dict = Depends(require_auth_if_enabled)):
    """Delete a fine-tuned model directory."""
    target = _resolve_finetuned(name)
    try:
        shutil.rmtree(str(target))
    except OSError as exc:
        raise_error(f"Failed to delete model: {exc}", "E_DELETE_FAILED", status_code=500)
    return {"status": "deleted", "name": name}


# ===== JOB RECOVERY =====


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


async def get_recovery_events():
    """Process-notification snapshot for the dock (contract v1).

    Boot-time events cannot ride the SSE streams — no client is connected
    when they fire — so the frontend's initial notification state is this
    REST projection of durable state (see training/process_events.py).
    Live transitions after load ride the existing per-request SSE streams
    and dedupe against these event_ids.
    """
    from training.process_events import CONTRACT_VERSION, boot_snapshot

    store = get_job_store()
    events = boot_snapshot(store)
    return {
        "contract_version": CONTRACT_VERSION,
        "count": len(events),
        "events": events,
    }


def start_recovery(job_id: str) -> dict:
    """Recover and restart an interrupted/failed job.

    Also accepts a 'recovering' job whose heartbeat went stale (a previous
    recovery run that died without completing). Resumes training from the last
    checkpoint if available.

    This is the ONE recovery path: the HTTP endpoint delegates here, and the
    boot-time auto-resume hook calls it too, so the validation above (fail-
    closed against the job's CURRENT status) guards every dispatch — a stale
    notification action and a boot race both hit the same wall.
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
    # exactly once, here in the request handler (via the engine, card c48aaaad).
    # The job explicitly points at it, so a recorded path that is missing,
    # unsupported, or unreadable fails the recovery request loudly (422) —
    # resuming from a different checkpoint would silently change what is being
    # resumed. Only when the job recorded NO checkpoint path do we fall back to
    # the newest file that actually loads (skipping partial/corrupt checkpoints
    # — a crash mid-write can leave the newest file unreadable). The loaded
    # bundle is handed to train() so no second load happens in the worker
    # thread.
    from domain.training.engine import get_training_engine

    try:
        checkpoint_path, resume_bundle = get_training_engine().resolve_resume_checkpoint(
            checkpoint_path, checkpoint_dir
        )
    except ValueError as exc:
        raise_error(str(exc), "E_VAL_REQUEST", status_code=422)

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
        from domain.infrastructure.cancel_manager import OpType, get_cancel_manager

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
            from domain.training.engine import get_training_engine

            # Reuse the SAME trainer configuration builder as /training/start so
            # the recovered run continues with the original job's hyperparameters
            # (LoRA, dropout, scheduler, device, ...) instead of a fixed subset.
            trainer_config = {
                "data_path": recovery_job.get("data_path", ""),
                **_sloughgpt_trainer_kwds(recovery_job, data_path=recovery_job.get("data_path")),
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

            # Resume from checkpoint if available (bundle pre-loaded once in the
            # request handler — no disk load happens in this thread; the engine
            # builds the trainer, trains, and hands it back — card c48aaaad)
            trainer = get_training_engine().run_recovery_training(
                config=trainer_config,
                on_progress=on_progress,
                resume=resume_bundle,
                events={"cancel_event": cancel_event, "pause_event": pause_event},
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

    get_training_engine().executor_submit(run_recovery, jid)

    return {
        "status": "recovered",
        "original_job_id": job_id,
        "recovery_job_id": recovery_job_id,
        "checkpoint_path": checkpoint_path,
        "message": f"Recovery started. Training restarting from checkpoint: {checkpoint_path or 'beginning'}",
    }


async def recover_job(job_id: str):
    """HTTP projection of start_recovery() — validation and execution are
    shared with the boot-time auto-resume hook, never re-implemented."""
    return start_recovery(job_id)


async def abandon_recovery(job_id: str, auth_user: dict = Depends(require_auth_if_enabled)):
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
# Business logic: the training service layer (domain.training.engine's
# raw-passthrough methods reach the _internal service/state/advisor/outcome/
# monitor modules; this file never imports them directly).
# HTTP wrapping: success_response, error classification, audit logging (this file)


async def training_log():
    from domain.training.engine import get_training_engine

    lines = await get_training_engine().get_log()
    return {"lines": lines, "total": len(lines)}


async def training_stop():
    """Stop all active training (auto-train, turbo, from-sessions)."""
    from .sse_stream import stop_all_training

    result = stop_all_training()
    return result


async def training_list_checkpoints():
    from domain.training.engine import get_training_engine

    checkpoints = await get_training_engine().list_stored_checkpoints()
    return checkpoints


async def training_delete_checkpoint(name: str):
    from schemas.common import safe_audit_log

    from domain.training.engine import get_training_engine

    engine = get_training_engine()
    if not engine.is_valid_ckpt_name(name):
        raise_error("Invalid checkpoint name", "E_BAD_REQUEST", status_code=400)
    deleted = await engine.delete_checkpoint(name)
    if deleted:
        safe_audit_log("training.checkpoint.delete", resource=name, detail="deleted")
    return {"deleted": deleted, "name": name}


async def training_load_checkpoint(name: str):
    from schemas.common import classify_and_raise, success_response

    from domain.training.engine import get_training_engine

    try:
        result = await get_training_engine().load_stored_checkpoint(name)
        return success_response(data=result, message="loaded")
    except Exception as e:
        classify_and_raise(e, source="training.load_checkpoint")


async def training_download_checkpoint(name: str):
    from fastapi.responses import FileResponse
    from schemas.common import raise_error

    from domain.training.engine import get_training_engine

    fp = await get_training_engine().download_checkpoint_path(name)
    if fp:
        return FileResponse(fp, media_type="application/octet-stream", filename=name)
    raise_error("Checkpoint not found", "E_NOT_FOUND", status_code=404)


async def training_checkpoint_info(name: str):
    from schemas.common import classify_and_raise

    from domain.training.engine import get_training_engine

    try:
        info = await get_training_engine().checkpoint_info(name)
        return info
    except Exception as e:
        classify_and_raise(e, source="training.checkpoint_info")


async def training_metrics_export():
    import json as _json

    from fastapi.responses import Response

    from domain.training.engine import get_training_engine

    checkpoints = await get_training_engine().get_all_checkpoint_data()
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


async def training_cancel_from_sessions():
    """Cancel session-based training."""
    from .sse_stream import cancel_from_sessions

    result = cancel_from_sessions()
    return result


async def training_stream(request: Request):
    """SSE training stream."""
    from pathlib import Path

    from .sse_stream import build_training_sse_response

    # Get training state for config
    try:
        from domain.training.engine import get_training_engine

        config = get_training_engine().get_state().config or {}
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


async def training_from_sessions_stream(request: Request):
    """SSE stream from sessions."""
    from pathlib import Path

    from .sse_stream import build_training_sse_response

    try:
        from domain.training.engine import get_training_engine

        config = get_training_engine().get_state().config or {}
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

# Consolidated from routers/ (card 20261101_154058) — the /training row's
# remaining router gap; descriptor-projected, byte-identical paths.
from .cloud_training import router as _cloud_training_router
from .lora_eval import router as _lora_eval_router
from .self_train import router as _self_train_router

router.include_router(_self_train_router)
router.include_router(_lora_eval_router)
router.include_router(_cloud_training_router)


async def get_training_recommendation(
    dataset_path: str = "",
    method: str = "distill",
):
    """Get training configuration recommendations based on dataset characteristics."""
    try:
        from pathlib import Path as _P

        from domain.training.engine import get_training_engine

        engine = get_training_engine()

        dataset_size = 0
        avg_quality = None

        if dataset_path:
            dp = _P(dataset_path).resolve()
            repo_root = find_repo_root(Path(__file__).resolve())
            datasets_dir = (repo_root / "datasets").resolve()
            data_dir = (repo_root / "data").resolve()
            if str(dp).startswith(str(datasets_dir)) or str(dp).startswith(str(data_dir)):
                if dp.exists():
                    if dp.is_file():
                        try:
                            with open(dp, encoding="utf-8", errors="ignore") as f:
                                dataset_size = sum(1 for _ in f)
                        except Exception:
                            dataset_size = 0
                    elif dp.is_dir():
                        data_files = (
                            list(dp.rglob("*.jsonl"))
                            + list(dp.rglob("*.json"))
                            + list(dp.rglob("*.txt"))
                        )
                        dataset_size = len(data_files)

        # Get recommendation
        recommendation = engine.recommend_training_config(
            dataset_size=dataset_size,
            method=method,
            avg_quality=avg_quality,
        )

        # Get tips
        tips = engine.get_training_tips(dataset_size=dataset_size)

        return {
            "dataset_size": dataset_size,
            "recommendation": {
                "learning_rate": recommendation.learning_rate,
                "batch_size": recommendation.batch_size,
                "epochs": recommendation.epochs,
                "warmup_steps": recommendation.warmup_steps,
                "early_stopping_patience": recommendation.early_stopping_patience,
                "reason": recommendation.reason,
                "confidence": recommendation.confidence,
            },
            "tips": tips,
        }

    except Exception as e:
        logger.warning("Failed to generate training recommendation: %s", e)
        return {
            "dataset_size": 0,
            "recommendation": {
                "learning_rate": 3e-4,
                "batch_size": 32,
                "epochs": 20,
                "warmup_steps": 50,
                "early_stopping_patience": 5,
                "reason": "Default recommendation",
                "confidence": 0.5,
            },
            "tips": ["Could not analyze dataset. Using default configuration."],
        }


# ── Training Trends ─────────────────────────────────────────────────────


async def get_training_trends(
    model: str = "",
    dataset: str = "",
    limit: int = 100,
):
    """Get training history trends for the version tracker page.

    Returns quality-over-time data, per-model breakdowns, and summary stats
    from the outcome tracker JSONL file.
    """
    try:
        from domain.training.engine import get_training_engine

        outcomes = get_training_engine().load_outcomes()

        # Filter
        if model:
            outcomes = [o for o in outcomes if o.model == model]
        if dataset:
            outcomes = [o for o in outcomes if o.dataset == dataset]

        # Sort by timestamp
        outcomes.sort(key=lambda o: o.timestamp)

        # Cap to limit
        if len(outcomes) > limit:
            outcomes = outcomes[-limit:]

        if not outcomes:
            return {
                "runs": [],
                "models": [],
                "summary": {
                    "total_runs": 0,
                    "avg_quality": 0,
                    "best_quality": 0,
                    "avg_loss": 0,
                    "trend": "stable",
                },
            }

        # Build run list
        runs = []
        for o in outcomes:
            runs.append(
                {
                    "run_id": o.run_id,
                    "timestamp": o.timestamp,
                    "dataset": o.dataset,
                    "dataset_size": o.dataset_size,
                    "model": o.model,
                    "method": o.method,
                    "epochs": o.epochs,
                    "batch_size": o.batch_size,
                    "learning_rate": o.learning_rate,
                    "final_loss": o.final_loss,
                    "best_loss": o.best_loss,
                    "perplexity": o.perplexity,
                    "converged": o.converged,
                    "early_stopped": o.early_stopped,
                    "quality_score": o.quality_score,
                    "training_time_s": o.training_time_s,
                }
            )

        # Per-model breakdown
        model_map: dict[str, list] = {}
        for r in runs:
            m = r["model"] or "unknown"
            model_map.setdefault(m, []).append(r)

        models = []
        for model_name, model_runs in model_map.items():
            qualities = [r["quality_score"] for r in model_runs]
            losses = [r["final_loss"] for r in model_runs if r["final_loss"] > 0]
            models.append(
                {
                    "model": model_name,
                    "total_runs": len(model_runs),
                    "avg_quality": sum(qualities) / len(qualities) if qualities else 0,
                    "best_quality": max(qualities) if qualities else 0,
                    "avg_loss": sum(losses) / len(losses) if losses else 0,
                    "latest_quality": model_runs[-1]["quality_score"],
                    "improving": len(model_runs) >= 2
                    and model_runs[-1]["quality_score"] > model_runs[0]["quality_score"],
                }
            )

        models.sort(key=lambda m: m["best_quality"], reverse=True)

        # Summary
        all_qualities = [r["quality_score"] for r in runs]
        all_losses = [r["final_loss"] for r in runs if r["final_loss"] > 0]

        # Trend detection: compare last 3 vs first 3
        trend = "stable"
        if len(runs) >= 6:
            first_3 = sum(r["quality_score"] for r in runs[:3]) / 3
            last_3 = sum(r["quality_score"] for r in runs[-3:]) / 3
            delta = last_3 - first_3
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"

        summary = {
            "total_runs": len(runs),
            "avg_quality": sum(all_qualities) / len(all_qualities) if all_qualities else 0,
            "best_quality": max(all_qualities) if all_qualities else 0,
            "avg_loss": sum(all_losses) / len(all_losses) if all_losses else 0,
            "trend": trend,
        }

        return {
            "runs": runs,
            "models": models,
            "summary": summary,
        }

    except Exception as e:
        logger.warning("Failed to load training trends: %s", e)
        return {
            "runs": [],
            "models": [],
            "summary": {
                "total_runs": 0,
                "avg_quality": 0,
                "best_quality": 0,
                "avg_loss": 0,
                "trend": "stable",
            },
        }


async def training_monitor_status():
    """Training monitoring status and alerts.

    Returns current monitoring state, alerts, and metrics history.
    Useful for detecting training issues like loss divergence,
    gradient explosion, or resource exhaustion.
    """
    from domain.training.engine import get_training_engine

    monitor = get_training_engine().get_training_monitor()
    return monitor.get_status()


async def training_monitor_alerts(severity: str | None = None, limit: int = 50):
    """Get training alerts.

    Args:
        severity: Filter by severity (info, warning, error, critical)
        limit: Maximum number of alerts to return
    """
    from domain.training.engine import get_training_engine

    engine = get_training_engine()
    monitor = engine.get_training_monitor()
    severity_filter = engine.alert_severity(severity) if severity else None
    alerts = monitor.get_alerts(severity=severity_filter, limit=limit)
    return {
        "alerts": [a.to_dict() for a in alerts],
        "total": len(alerts),
    }


async def training_monitor_metrics(limit: int = 100):
    """Get training metrics history.

    Args:
        limit: Maximum number of metrics snapshots to return
    """
    from domain.training.engine import get_training_engine

    monitor = get_training_engine().get_training_monitor()
    metrics = monitor.get_metrics_history(limit=limit)
    return {
        "metrics": [m.to_dict() for m in metrics],
        "total": len(metrics),
    }


async def training_monitor_reset():
    """Reset training monitor state."""
    from domain.training.engine import get_training_engine

    monitor = get_training_engine().get_training_monitor()
    monitor.reset()
    return {"message": "Monitor reset"}


async def training_monitor_resources():
    """Check system resources.

    Returns CPU, memory, and GPU usage with alerts if thresholds exceeded.
    """
    from domain.training.engine import get_training_engine

    monitor = get_training_engine().get_training_monitor()
    resources = monitor.check_resources()
    return resources


# ── Descriptor projection: one contract per route (card 20261104_151036) ──
# Registration order mirrors the hand-built decorators exactly — the
# parameterized pairs (/{name}, /{job_id}) come after their static siblings
# where any shadowing is possible. Descriptions derive from the handler
# docstrings (the expression FastAPI itself used); summaries are pinned to
# the captured hand-built values.

SPEC_FINETUNED_LIST = ToolSpec(
    name="training.finetuned.list",
    description=inspect.cleandoc(list_finetuned_models.__doc__ or ""),
    parameters=[],
    execute=list_finetuned_models,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_FINETUNED_LOAD = ToolSpec(
    name="training.finetuned.load",
    description=inspect.cleandoc(load_finetuned_model.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=load_finetuned_model,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_FINETUNED_DELETE = ToolSpec(
    name="training.finetuned.delete",
    description=inspect.cleandoc(delete_finetuned_model.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=delete_finetuned_model,
    auth_scope="authenticated",
    idempotent=False,
    version="1",
)

SPEC_RECOVERY_CHECK = ToolSpec(
    name="training.recovery.check",
    description=inspect.cleandoc(check_crashed_jobs.__doc__ or ""),
    parameters=[ToolParam("timeout_seconds", "integer", "", False)],
    execute=check_crashed_jobs,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_RECOVERY_RECOVERABLE = ToolSpec(
    name="training.recovery.recoverable",
    description=inspect.cleandoc(get_recoverable_jobs.__doc__ or ""),
    parameters=[],
    execute=get_recoverable_jobs,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_RECOVERY_EVENTS = ToolSpec(
    name="training.recovery.events",
    description=inspect.cleandoc(get_recovery_events.__doc__ or ""),
    parameters=[],
    execute=get_recovery_events,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_RECOVERY_RECOVER = ToolSpec(
    name="training.recovery.recover",
    description=inspect.cleandoc(recover_job.__doc__ or ""),
    parameters=[ToolParam("job_id", "string", "", True)],
    execute=recover_job,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_RECOVERY_ABANDON = ToolSpec(
    name="training.recovery.abandon",
    description=inspect.cleandoc(abandon_recovery.__doc__ or ""),
    parameters=[ToolParam("job_id", "string", "", True)],
    execute=abandon_recovery,
    auth_scope="authenticated",
    idempotent=False,
    version="1",
)

SPEC_RECOVERY_STATS = ToolSpec(
    name="training.recovery.stats",
    description=inspect.cleandoc(get_recovery_stats.__doc__ or ""),
    parameters=[],
    execute=get_recovery_stats,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_LOG_GET = ToolSpec(
    name="training.log.get",
    description=inspect.cleandoc(training_log.__doc__ or ""),
    parameters=[],
    execute=training_log,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_STOP_ALL = ToolSpec(
    name="training.stop.all",
    description=inspect.cleandoc(training_stop.__doc__ or ""),
    parameters=[],
    execute=training_stop,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_CHECKPOINTS_LIST = ToolSpec(
    name="training.checkpoints.list",
    description=inspect.cleandoc(training_list_checkpoints.__doc__ or ""),
    parameters=[],
    execute=training_list_checkpoints,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_CHECKPOINTS_DELETE = ToolSpec(
    name="training.checkpoints.delete",
    description=inspect.cleandoc(training_delete_checkpoint.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=training_delete_checkpoint,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_CHECKPOINTS_LOAD = ToolSpec(
    name="training.checkpoints.load",
    description=inspect.cleandoc(training_load_checkpoint.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=training_load_checkpoint,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_CHECKPOINTS_DOWNLOAD = ToolSpec(
    name="training.checkpoints.download",
    description=inspect.cleandoc(training_download_checkpoint.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=training_download_checkpoint,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_CHECKPOINTS_INFO = ToolSpec(
    name="training.checkpoints.info",
    description=inspect.cleandoc(training_checkpoint_info.__doc__ or ""),
    parameters=[ToolParam("name", "string", "", True)],
    execute=training_checkpoint_info,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_METRICS_EXPORT = ToolSpec(
    name="training.metrics.export",
    description=inspect.cleandoc(training_metrics_export.__doc__ or ""),
    parameters=[],
    execute=training_metrics_export,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_FROM_SESSIONS_CANCEL = ToolSpec(
    name="training.from_sessions.cancel",
    description=inspect.cleandoc(training_cancel_from_sessions.__doc__ or ""),
    parameters=[],
    execute=training_cancel_from_sessions,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_STREAM_LIVE = ToolSpec(
    name="training.stream.live",
    description=inspect.cleandoc(training_stream.__doc__ or ""),
    parameters=[],
    execute=training_stream,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_STREAM_FROM_SESSIONS = ToolSpec(
    name="training.stream.from_sessions",
    description=inspect.cleandoc(training_from_sessions_stream.__doc__ or ""),
    parameters=[],
    execute=training_from_sessions_stream,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_RECOMMEND_GET = ToolSpec(
    name="training.recommend.get",
    description=inspect.cleandoc(get_training_recommendation.__doc__ or ""),
    parameters=[
        ToolParam("dataset_path", "string", "", False),
        ToolParam("method", "string", "", False),
    ],
    execute=get_training_recommendation,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_TRENDS_GET = ToolSpec(
    name="training.trends.get",
    description=inspect.cleandoc(get_training_trends.__doc__ or ""),
    parameters=[
        ToolParam("model", "string", "", False),
        ToolParam("dataset", "string", "", False),
        ToolParam("limit", "integer", "", False),
    ],
    execute=get_training_trends,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_MONITOR_STATUS = ToolSpec(
    name="training.monitor.status",
    description=inspect.cleandoc(training_monitor_status.__doc__ or ""),
    parameters=[],
    execute=training_monitor_status,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_MONITOR_ALERTS = ToolSpec(
    name="training.monitor.alerts",
    description=inspect.cleandoc(training_monitor_alerts.__doc__ or ""),
    parameters=[
        ToolParam("severity", "string", "", False),
        ToolParam("limit", "integer", "", False),
    ],
    execute=training_monitor_alerts,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_MONITOR_METRICS = ToolSpec(
    name="training.monitor.metrics",
    description=inspect.cleandoc(training_monitor_metrics.__doc__ or ""),
    parameters=[ToolParam("limit", "integer", "", False)],
    execute=training_monitor_metrics,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_MONITOR_RESET = ToolSpec(
    name="training.monitor.reset",
    description=inspect.cleandoc(training_monitor_reset.__doc__ or ""),
    parameters=[],
    execute=training_monitor_reset,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_MONITOR_RESOURCES = ToolSpec(
    name="training.monitor.resources",
    description=inspect.cleandoc(training_monitor_resources.__doc__ or ""),
    parameters=[],
    execute=training_monitor_resources,
    auth_scope="public",
    idempotent=True,
    version="1",
)

router.include_router(
    create_router(
        SPEC_FINETUNED_LIST,
        RouteSpec(
            path="/training/finetuned-models", envelope=False, summary="List Finetuned Models"
        ),
        list_finetuned_models,
    )
)
router.include_router(
    create_router(
        SPEC_FINETUNED_LOAD,
        RouteSpec(
            path="/training/finetuned-models/{name}/load",
            envelope=False,
            summary="Load Finetuned Model",
        ),
        load_finetuned_model,
    )
)
router.include_router(
    create_router(
        SPEC_FINETUNED_DELETE,
        RouteSpec(
            path="/training/finetuned-models/{name}",
            method="DELETE",
            envelope=False,
            summary="Delete Finetuned Model",
        ),
        delete_finetuned_model,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_CHECK,
        RouteSpec(path="/recovery/check", envelope=False, summary="Check Crashed Jobs"),
        check_crashed_jobs,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_RECOVERABLE,
        RouteSpec(path="/recovery/recoverable", envelope=False, summary="Get Recoverable Jobs"),
        get_recoverable_jobs,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_EVENTS,
        RouteSpec(path="/recovery/events", envelope=False, summary="Get Recovery Events"),
        get_recovery_events,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_RECOVER,
        RouteSpec(path="/recovery/recover/{job_id}", envelope=False, summary="Recover Job"),
        recover_job,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_ABANDON,
        RouteSpec(
            path="/recovery/abandon/{job_id}",
            method="DELETE",
            envelope=False,
            summary="Abandon Recovery",
        ),
        abandon_recovery,
    )
)
router.include_router(
    create_router(
        SPEC_RECOVERY_STATS,
        RouteSpec(path="/recovery/stats", envelope=False, summary="Get Recovery Stats"),
        get_recovery_stats,
    )
)
router.include_router(
    create_router(
        SPEC_LOG_GET,
        RouteSpec(path="/training/log", envelope=True, summary="Training Log"),
        training_log,
    )
)
router.include_router(
    create_router(
        SPEC_STOP_ALL,
        RouteSpec(path="/training/stop", envelope=True, summary="Training Stop"),
        training_stop,
    )
)
router.include_router(
    create_router(
        SPEC_CHECKPOINTS_LIST,
        RouteSpec(path="/training/checkpoints", envelope=True, summary="Training List Checkpoints"),
        training_list_checkpoints,
    )
)
router.include_router(
    create_router(
        SPEC_CHECKPOINTS_DELETE,
        RouteSpec(
            path="/training/checkpoints/{name}",
            method="DELETE",
            envelope=True,
            summary="Training Delete Checkpoint",
        ),
        training_delete_checkpoint,
    )
)
router.include_router(
    create_router(
        SPEC_CHECKPOINTS_LOAD,
        RouteSpec(
            path="/training/checkpoints/{name}/load",
            envelope=False,
            summary="Training Load Checkpoint",
        ),
        training_load_checkpoint,
    )
)
router.include_router(
    create_router(
        SPEC_CHECKPOINTS_DOWNLOAD,
        RouteSpec(
            path="/training/checkpoints/{name}/download",
            envelope=False,
            summary="Training Download Checkpoint",
        ),
        training_download_checkpoint,
    )
)
router.include_router(
    create_router(
        SPEC_CHECKPOINTS_INFO,
        RouteSpec(
            path="/training/checkpoints/{name}/info",
            envelope=True,
            summary="Training Checkpoint Info",
        ),
        training_checkpoint_info,
    )
)
router.include_router(
    create_router(
        SPEC_METRICS_EXPORT,
        RouteSpec(
            path="/training/metrics/export", envelope=False, summary="Training Metrics Export"
        ),
        training_metrics_export,
    )
)
router.include_router(
    create_router(
        SPEC_FROM_SESSIONS_CANCEL,
        RouteSpec(
            path="/training/from-sessions/cancel",
            envelope=True,
            summary="Training Cancel From Sessions",
        ),
        training_cancel_from_sessions,
    )
)
router.include_router(
    create_router(
        SPEC_STREAM_LIVE,
        RouteSpec(path="/training/stream", envelope=False, summary="Training Stream"),
        training_stream,
    )
)
router.include_router(
    create_router(
        SPEC_STREAM_FROM_SESSIONS,
        RouteSpec(
            path="/training/from-sessions-stream",
            envelope=False,
            summary="Training From Sessions Stream",
        ),
        training_from_sessions_stream,
    )
)
router.include_router(
    create_router(
        SPEC_RECOMMEND_GET,
        RouteSpec(path="/training/recommend", envelope=True, summary="Get Training Recommendation"),
        get_training_recommendation,
    )
)
router.include_router(
    create_router(
        SPEC_TRENDS_GET,
        RouteSpec(path="/training/trends", envelope=True, summary="Get Training Trends"),
        get_training_trends,
    )
)
router.include_router(
    create_router(
        SPEC_MONITOR_STATUS,
        RouteSpec(path="/monitor", envelope=True, summary="Training Monitor Status"),
        training_monitor_status,
    )
)
router.include_router(
    create_router(
        SPEC_MONITOR_ALERTS,
        RouteSpec(path="/monitor/alerts", envelope=True, summary="Training Monitor Alerts"),
        training_monitor_alerts,
    )
)
router.include_router(
    create_router(
        SPEC_MONITOR_METRICS,
        RouteSpec(path="/monitor/metrics", envelope=True, summary="Training Monitor Metrics"),
        training_monitor_metrics,
    )
)
router.include_router(
    create_router(
        SPEC_MONITOR_RESET,
        RouteSpec(path="/monitor/reset", envelope=True, summary="Training Monitor Reset"),
        training_monitor_reset,
    )
)
router.include_router(
    create_router(
        SPEC_MONITOR_RESOURCES,
        RouteSpec(path="/monitor/resources", envelope=True, summary="Training Monitor Resources"),
        training_monitor_resources,
    )
)
