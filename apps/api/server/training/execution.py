"""Training execution routes — start, distill, lora, visual, from-feedback, builds.

Each route group is extracted into its own module; this file includes them via
``router.include_router()`` and defines the core ``start_training`` endpoint.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from domains.mobile.notifications import get_notification_service
from domains.training.executor import get_training_executor
from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error

from .controller import get_training_controller
from .helpers import _finish_job, _run_async, _sloughgpt_trainer_kwds
from .jobs import training_jobs
from .resolution import resolve_training_inputs
from .schemas import TrainingRequest
from .webhooks import notify_training_event

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-execution"])

# Include legacy /train endpoints
from .legacy import router as legacy_router

router.include_router(legacy_router)

# Include LoRA routes
from .lora import router as lora_router

router.include_router(lora_router)

# Include distill routes
from .distill import router as distill_router

router.include_router(distill_router)

# Include visual routes
from .visual import router as visual_router

router.include_router(visual_router)

# Include from_feedback routes
from .from_feedback import router as feedback_router

router.include_router(feedback_router)

# Include builds routes
from .builds import router as builds_router

router.include_router(builds_router)


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

    # Pre-flight dataset validation
    from pathlib import Path as _P

    _data_path = _P(data_path_str)
    if not _data_path.exists():
        raise_error(f"Dataset not found: {data_path_str}", "E_BAD_REQUEST", status_code=400)
    if _data_path.is_file():
        _size = _data_path.stat().st_size
        if _size == 0:
            raise_error("Dataset file is empty (0 bytes)", "E_BAD_REQUEST", status_code=400)
        if _size < 100:
            raise_error(f"Dataset file is too small ({_size} bytes). Need at least 100 bytes.", "E_BAD_REQUEST", status_code=400)
    elif _data_path.is_dir():
        _files = list(_data_path.rglob("*"))
        _data_files = [f for f in _files if f.is_file() and f.suffix in (".jsonl", ".json", ".txt", ".csv", ".parquet")]
        if len(_data_files) == 0:
            raise_error(f"No data files found in directory (expected .jsonl, .json, .txt, .csv, or .parquet)", "E_BAD_REQUEST", status_code=400)
        _total_size = sum(f.stat().st_size for f in _data_files)
        if _total_size < 100:
            raise_error(f"Total dataset size too small ({_total_size} bytes across {len(_data_files)} files)", "E_BAD_REQUEST", status_code=400)

    # Pre-flight quality gate: check data quality before wasting compute
    try:
        import asyncio as _aio
        from pathlib import Path as _PQ

        _q_path = _PQ(data_path_str)
        if _q_path.is_file():
            _raw = await _aio.to_thread(lambda: _q_path.read_text(encoding="utf-8", errors="replace")[:200_000])
        elif _q_path.is_dir():
            _candidates = [_q_path / "input.txt", _q_path / "corpus.jsonl", _q_path / "train.txt"]
            _q_file = next((c for c in _candidates if c.exists()), None)
            _raw = await _aio.to_thread(lambda: _q_file.read_text(encoding="utf-8", errors="replace")[:200_000]) if _q_file else ""
        else:
            _raw = ""

        if _raw:
            from domains.training.quality_scorer import compute_data_quality
            _quality = await _aio.to_thread(compute_data_quality, _raw)
            _avg = _quality.get("avg_quality", 0)
            _tox = _quality.get("toxicity_rate", 0)

            if _avg < 1.0:
                raise_error(
                    f"Data quality too low (avg_quality={_avg:.2f}/5.0, minimum=1.0). "
                    "Improve your training data or use a different dataset.",
                    "E_BAD_REQUEST",
                    status_code=400,
                    details=_quality,
                )
            if _tox > 0.5:
                raise_error(
                    f"Data toxicity too high (toxicity_rate={_tox:.2f}, maximum=0.5). "
                    "Remove harmful or inappropriate content from your training data.",
                    "E_BAD_REQUEST",
                    status_code=400,
                    details=_quality,
                )
    except Exception as e:
        from schemas.common import AppError
        if isinstance(e, AppError):
            raise
        logger.warning("Quality gate check failed (proceeding): %s", e)

    # Pre-flight JSONL validation for conversation-format datasets
    try:
        import asyncio as _aio
        from pathlib import Path as _PJ
        from domains.training.train_pipeline import validate_conversation_data

        _j_path = _PJ(data_path_str)
        _j_file = None
        if _j_path.is_file() and _j_path.suffix == ".jsonl":
            _j_file = _j_path
        elif _j_path.is_dir():
            for _c in ["input.jsonl", "corpus.jsonl", "train.jsonl", "conversations.jsonl"]:
                if (_j_path / _c).is_file():
                    _j_file = _j_path / _c
                    break

        if _j_file:
            _result = await _aio.to_thread(validate_conversation_data, str(_j_file))
            if _result["error_count"] > 0 and _result["valid_count"] == 0:
                raise_error(
                    f"Dataset has no valid conversation entries ({_result['error_count']} malformed lines). "
                    f"First error: {_result['errors'][0] if _result['errors'] else 'unknown'}",
                    "E_BAD_REQUEST",
                    status_code=400,
                )
            if _result["error_count"] > 0:
                _total = _result["valid_count"] + _result["error_count"]
                _error_ratio = _result["error_count"] / max(_total, 1)
                logger.warning(
                    "JSONL validation: %d valid, %d errors (%.0f%%) in %s",
                    _result["valid_count"],
                    _result["error_count"],
                    _error_ratio * 100,
                    _j_file,
                )
                if _error_ratio > 0.5:
                    raise_error(
                        f"Dataset has too many malformed lines ({_result['error_count']}/{_total}, "
                        f"{_error_ratio:.0%} error rate). Fix the dataset or lower the error threshold.",
                        "E_BAD_REQUEST",
                        status_code=400,
                    )
    except Exception as e:
        from schemas.common import AppError
        if isinstance(e, AppError):
            raise
        logger.warning("JSONL validation failed (proceeding): %s", e)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
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
        "user_id": auth_user.get("sub", "") if auth_user else "",
        "workspace_id": auth_user.get("workspace_id", "") if auth_user else "",
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
            extra={
                "job_id": job_id,
                "model": request.model,
                "epochs": request.epochs,
                "source_kind": source_kind,
            },
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
    from .runtime import get_training_runtime

    get_training_runtime().register(job_id, training_jobs[job_id], cancel_event, req_snapshot)

    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

        get_cancel_manager().register(
            op_type=OpType.TRAINING,
            label=str(req_snapshot.get("name") or job_id),
            cancel_fn=lambda: cancel_event.set(),
            meta={"job_id": job_id, "method": "slonet"},
            op_id=job_id,
        )
        get_cancel_manager().start(job_id)
    except Exception as e:
        logger.warning("CancelManager registration failed for %s: %s", job_id, e)

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
                    rec.setdefault("loss_history", []).append(
                        {"step": rec.get("global_step", 0), "value": float(tl), "type": "train"}
                    )
                el = info.get("eval_loss")
                if el is not None:
                    fe = float(el)
                    rec["eval_loss"] = fe
                    rec["loss"] = fe
                    rec.setdefault("loss_history", []).append(
                        {"step": rec.get("global_step", 0), "value": fe, "type": "eval"}
                    )
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
                get_training_controller().reset()
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
                _run_async(
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
                _run_async(
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
            except Exception as exc:
                logger.warning("Training failure webhook failed for %s: %s", jid, exc)

            # Push notification to mobile devices
            try:
                get_notification_service().send_notification_sync(
                    title="Training Failed",
                    body=f"Job {training_jobs[jid].get('name', jid)} failed: {str(e)[:100]}",
                    data={"screen": "Training", "job_id": jid},
                    topics=["training"],
                )
            except Exception as exc:
                logger.warning("Training failure push notification failed for %s: %s", jid, exc)
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


# ── Visual Training (delegated to visual.py) ────────────────────
# Note: visual routes are in training/visual.py, included via router below.

# ── Distillation (delegated to distill.py) ───────────────────────
# Note: distill routes are in training/distill.py, included via router below.

# ── LoRA Fine-tuning (delegated to lora.py) ──────────────────────
# Note: lora routes are in training/lora.py, included via router below.

# ── From Feedback (delegated to from_feedback.py) ────────────────
# Note: feedback routes are in training/from_feedback.py, included via router below.

# ── Builds (delegated to builds.py) ─────────────────────────────
# Note: builds routes are in training/builds.py, included via router below.
