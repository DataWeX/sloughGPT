"""Shared helpers for training routes — _finish_job, _sloughgpt_trainer_kwds, _run_async.

Extracted to break circular imports between router.py and execution.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

from .jobs import training_jobs

logger = logging.getLogger("slo")


def _run_async(coro: Coroutine) -> None:
    """Run an async coroutine from a sync (background thread) context.

    If an event loop is already running in the current thread, spawns a
    daemon thread to avoid "cannot call asyncio.run from a running loop".
    Otherwise uses ``asyncio.run()`` directly. Failures are logged and
    swallowed — this is fire-and-forget.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import threading

        def _target():
            try:
                asyncio.run(coro)
            except Exception as exc:
                logger.debug("Fire-and-forget coroutine failed: %s", exc)

        threading.Thread(target=_target, daemon=True).start()
    else:
        try:
            asyncio.run(coro)
        except Exception as exc:
            logger.debug("Fire-and-forget coroutine failed: %s", exc)


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
