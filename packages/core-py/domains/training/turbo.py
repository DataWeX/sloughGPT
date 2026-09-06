"""Turbo training — start, worker, status."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .helpers import _finite_payload, log_experiment_metric, resolve_dataset_path
from .runtime_protocol import update_job
from .state import (
    CHECKPOINTS_DIR,
    REPO_ROOT,
    TURBO_DIR,
    _state,
    _turbo_cancel_event,
    _turbo_lock,
    _turbo_pause_event,
    _turbo_state,
)

logger = logging.getLogger("slo.training")


def get_turbo_status() -> dict:
    with _turbo_lock:
        state = dict(_turbo_state)
        if (state.get("status") == "running"
                and state.get("last_heartbeat", 0) > 0
                and (time.time() - state["last_heartbeat"]) > 30):
            state["status"] = "error"
            state["error"] = "Training process lost — no progress for 30 seconds"
            state["paused"] = False
            _turbo_state["status"] = "error"
            _turbo_state["error"] = state["error"]
            _turbo_state["paused"] = False
            _turbo_pause_event.clear()
    return state


def start_turbo_training(config: dict) -> dict:
    with _turbo_lock:
        if _turbo_state.get("status") == "running":
            raise RuntimeError("A turbo training job is already running")

    data_path = config.get("data_path", "")
    dataset_id = config.get("dataset_id")
    if not data_path and dataset_id:
        data_path = resolve_dataset_path(dataset_id)

    if not data_path:
        raise ValueError("No data_path or dataset_id provided")

    config["data_path"] = data_path

    resume = False
    resume_path = ""
    checkpoint_name = config.get("checkpoint_name")
    if checkpoint_name:
        ckpt_soul = CHECKPOINTS_DIR / f"{checkpoint_name}.soul"
        if ckpt_soul.exists():
            resume = True
            resume_path = str(ckpt_soul)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    dp = Path(data_path).resolve()
    allowed_bases = [REPO_ROOT / "datasets", REPO_ROOT / "data"]
    if not dp.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not any(str(dp).startswith(str(b.resolve())) for b in allowed_bases if b.exists()):
        raise ValueError(
            f"Data path must be under datasets/ or data/ directories, got: {data_path}"
        )

    job_id = f"turbo_{int(time.time())}"
    cancel_event = threading.Event()
    pause_event = threading.Event()

    with _turbo_lock:
        _turbo_state.update({
            "status": "running",
            "job_id": job_id,
            "global_step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "loss": None,
            "learning_rate": None,
            "steps_per_sec": None,
            "eta_s": None,
            "elapsed_s": None,
            "avg_quality": None,
            "result": None,
            "error": None,
            "paused": False,
            "last_heartbeat": time.time(),
        })

    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager
        _mgr = get_cancel_manager()
        _cm_op_id = _mgr.register(
            op_type=OpType.TRAINING,
            label=f"auto-turbo:{config.get('method', 'slonet')}",
            cancel_fn=lambda: cancel_event.set(),
        )
        _mgr.start(_cm_op_id)
        _turbo_state["_cm_op_id"] = _cm_op_id
    except Exception as e:
        logger.warning("CancelManager registration failed (turbo training may be unkillable): %s", e)

    output_dir = TURBO_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    method = config.get("method", "slonet")
    experiment_id = config.get("experiment_id")
    runtime_job = {
        "id": job_id,
        "name": config.get("name", None) or "turbo",
        "model": method or "sloughgpt",
        "dataset": config.get("dataset_id") or data_path,
        "data_path": data_path,
        "status": "running",
        "progress": 0.0,
        "epochs": config.get("epochs"),
        "current_epoch": 0,
        "global_step": 0,
        "loss": None,
        "train_loss": None,
        "checkpoint": None,
        "checkpoint_dir": str(output_dir),
        "error": None,
        "experiment_id": experiment_id,
    }
    from domains.training.runtime_protocol import get_training_runtime
    get_training_runtime().register(job_id, runtime_job, cancel_event, config)

    return {
        "job_id": job_id,
        "output_dir": str(output_dir),
        "data_path": data_path,
        "resume": resume,
        "resume_path": resume_path,
        "cancel_event": cancel_event,
        "pause_event": pause_event,
    }


def run_turbo_worker(config: dict) -> None:
    from domains.training.train_pipeline import SloughGPTTrainer

    job_id = _turbo_state.get("job_id", "")
    data_path = config.get("data_path", "")

    def _finish_cm(status: str, error: str = "") -> None:
        op_id = _turbo_state.get("_cm_op_id")
        if op_id:
            try:
                from domains.infrastructure.cancel_manager import get_cancel_manager
                get_cancel_manager().finish(op_id, error=error if status != "completed" else "")
            except Exception as exc:
                logger.debug("CancelManager.finish failed: %s", exc)

    if not data_path:
        dataset_id = config.get("dataset_id")
        if dataset_id:
            data_path = resolve_dataset_path(dataset_id)
        if not data_path:
            update_job(job_id, status="error", error="No data_path or dataset_id provided")
            _finish_cm("error", "No data_path or dataset_id provided")
            return

    output_dir = config.get("output_dir", str(TURBO_DIR))
    resume = config.get("resume", False)
    resume_path = config.get("resume_path", "")
    experiment_id = config.get("experiment_id")

    def on_progress(info: dict[str, Any]) -> None:
        with _turbo_lock:
            _turbo_state["global_step"] = int(info.get("global_step", _turbo_state["global_step"]))
            _turbo_state["total_steps"] = int(info.get("total_steps", _turbo_state["total_steps"]))
            _turbo_state["progress"] = float(info.get("progress_percent", 0))
            _turbo_state["loss"] = info.get("train_loss", _turbo_state["loss"])
            _turbo_state["learning_rate"] = info.get("learning_rate", _turbo_state["learning_rate"])
            _turbo_state["steps_per_sec"] = info.get("steps_per_sec", _turbo_state["steps_per_sec"])
            _turbo_state["eta_s"] = info.get("eta_s", _turbo_state["eta_s"])
            _turbo_state["elapsed_s"] = info.get("elapsed_s", _turbo_state["elapsed_s"])
            _turbo_state["avg_quality"] = info.get("avg_quality", _turbo_state.get("avg_quality"))
            _turbo_state["last_heartbeat"] = time.time()
        update_job(
            job_id,
            progress=float(_turbo_state["progress"]),
            global_step=int(_turbo_state["global_step"]),
            train_loss=_turbo_state["loss"],
            loss=_turbo_state["loss"],
        )
        if experiment_id and _turbo_state["loss"] is not None:
            log_experiment_metric(experiment_id, "train_loss", float(_turbo_state["loss"]), int(_turbo_state["global_step"]))

    try:
        n_layer = config.get("n_layer") or config.get("n_decoder_layers") or 3
        block_size = config.get("block_size") or config.get("max_tgt_len") or 128

        trainer = SloughGPTTrainer(
            data_path=data_path,
            vocab_size=config.get("vocab_size", 500),
            n_embed=config.get("n_embed", 128),
            n_layer=n_layer,
            n_head=config.get("n_head", 4),
            block_size=block_size,
            dropout=config.get("dropout", 0.1),
            batch_size=config.get("batch_size", 4),
            epochs=config.get("epochs", 3),
            lr=config.get("learning_rate", 3e-4),
            checkpoint_dir=output_dir,
        )

        logger.info(
            "Starting SloughGPTTrainer with method=%s data=%s resume=%s",
            config.get("method"), data_path, resume,
        )
        _train_t0 = time.monotonic()
        result = trainer.train(
            on_progress=on_progress,
            cancel_event=_turbo_cancel_event,
            pause_event=_turbo_pause_event,
            resume=resume,
            resume_path=resume_path or None,
        )
        _train_elapsed_ms = (time.monotonic() - _train_t0) * 1000
        logger.info("SloughGPTTrainer result: %s (elapsed=%dms)", result, _train_elapsed_ms)

        if _turbo_cancel_event.is_set():
            _turbo_pause_event.clear()
            with _turbo_lock:
                _turbo_state["status"] = "error"
                _turbo_state["error"] = "Training cancelled"
                _turbo_state["paused"] = False
            update_job(job_id, status="cancelled", error="Training cancelled")
            _finish_cm("cancelled", "Training cancelled")
            return

        if isinstance(result, dict) and result.get("status") == "error":
            _turbo_pause_event.clear()
            with _turbo_lock:
                _turbo_state["status"] = "error"
                _turbo_state["error"] = result.get("message") or "Training failed"
                _turbo_state["paused"] = False
            update_job(job_id, status="failed", error=result.get("message") or "Training failed")
            _finish_cm("failed", result.get("message") or "Training failed")
            return

        _turbo_pause_event.clear()
        with _turbo_lock:
            _turbo_state["status"] = "complete"
            _turbo_state["result"] = _finite_payload(result)
            _turbo_state["progress"] = 100.0
            _turbo_state["paused"] = False
        soul_files = sorted(Path(output_dir).glob("*.soul"))
        update_job(
            job_id,
            status="completed",
            progress=100.0,
            checkpoint=str(soul_files[-1]) if soul_files else None,
        )
        _finish_cm("completed")
    except Exception as e:
        logger.error("SloughGPTTrainer failed: %s", e)
        _turbo_pause_event.clear()
        with _turbo_lock:
            _turbo_state["status"] = "error"
            _turbo_state["error"] = str(e)
            _turbo_state["paused"] = False
        update_job(job_id, status="failed", error=str(e))
        _finish_cm("failed", str(e))
        logger.warning("Turbo training failed: %s", e)
    finally:
        _state.running = False
