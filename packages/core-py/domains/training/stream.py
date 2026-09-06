"""SSE stream business logic — completion handling and cleanup."""

from __future__ import annotations

import logging
from pathlib import Path

from .helpers import log_experiment_metric, log_experiment_param
from .runtime_protocol import update_job

logger = logging.getLogger("slo.training")


def process_training_completion(
    ev: dict,
    task_id: str,
    config: dict,
    checkpoints_dir: Path,
    finish_cm_fn,
) -> None:
    status = "completed" if ev.get("status") == "complete" else "failed"
    error = "" if ev.get("status") == "complete" else str(ev.get("message") or ev.get("data") or "training failed")

    try:
        soul_files = sorted(checkpoints_dir.glob("*.soul"))
        checkpoint = str(soul_files[-1]) if soul_files else None
    except Exception:
        logger.debug("Failed to discover checkpoints in %s", checkpoints_dir, exc_info=True)
        checkpoint = None

    job = update_job(task_id, status=status, error=error, checkpoint=checkpoint)

    experiment_id = config.get("experiment_id")
    if experiment_id and ev.get("status") == "complete" and job is not None:
        final_loss = job.get("train_loss") or job.get("loss")
        if final_loss is not None:
            log_experiment_metric(experiment_id, "final_train_loss", float(final_loss), int(job.get("global_step", 0)))
        log_experiment_param(experiment_id, "epochs", config.get("epochs", 0))
        log_experiment_param(experiment_id, "learning_rate", config.get("learning_rate", 0))

    finish_cm_fn(
        "complete" if ev.get("status") == "complete" else "failed",
        str(ev.get("message") or ev.get("data") or "") if ev.get("status") != "complete" else "",
    )


def cleanup_stream_state(
    task_id: str,
    config: dict,
    state: dict,
    finish_cm_fn,
    status: str = "interrupted",
    error: str = "",
) -> None:
    state["running"] = False
    from .runtime_protocol import get_training_runtime
    job = get_training_runtime().get(task_id)
    if job is not None and job.get("status") not in ("completed", "failed", "cancelled"):
        update_job(task_id, status=status, error=job.get("error") or error or "Training stream ended before completion")
