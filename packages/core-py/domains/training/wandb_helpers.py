"""Weights & Biases helpers: env flags, flat config for ``wandb.config``, API training runs."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

_WANDB_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _WANDB_TRUTHY


def wandb_training_enabled_from_env() -> bool:
    """Enable W&B for ``POST /training/start`` background jobs when API key or offline mode is usable."""
    return env_flag("SLOUGHGPT_WANDB_TRAINING")


def wandb_server_enabled_from_env() -> bool:
    """Enable long-lived W&B run for HTTP + inference aggregates (``SLOUGHGPT_WANDB_SERVER``)."""
    return env_flag("SLOUGHGPT_WANDB_SERVER")


def default_wandb_project() -> str:
    return os.environ.get("WANDB_PROJECT", "sloughgpt")


def flatten_for_wandb_config(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dataclasses / dicts into dotted keys wandb can display."""
    out: Dict[str, Any] = {}
    if obj is None:
        return out
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if not isinstance(obj, dict):
        key = prefix or "value"
        if isinstance(obj, (int, float, str, bool)):
            out[key] = obj
        else:
            out[key] = str(obj)[:2000]
        return out
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_for_wandb_config(v, p))
        elif isinstance(v, (list, tuple)):
            out[p] = str(v)[:2000]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[p] = v
        elif is_dataclass(v) and not isinstance(v, type):
            out.update(flatten_for_wandb_config(v, p))
        else:
            out[p] = str(v)[:2000]
    return out


def create_training_tracker_for_api_job(
    *,
    job_id: str,
    job_name: str,
    data_path: str,
    hyperparams: Dict[str, Any],
):
    """Start a W&B run for a ``POST /training/start`` job when ``SLOUGHGPT_WANDB_TRAINING`` is set.

    Returns ``None`` if disabled, wandb is missing, or init fails.
    """
    if not wandb_training_enabled_from_env():
        return None
    try:
        from domains.training.tracking import ExperimentTracker, TrackerBackend, TrackingConfig
    except ImportError:
        return None

    safe_name = f"{job_id}_{job_name}"[:128]
    tc = TrackingConfig(
        backend=TrackerBackend.WANDB,
        project=default_wandb_project(),
        entity=os.environ.get("WANDB_ENTITY") or None,
        api_key=os.environ.get("WANDB_API_KEY"),
        run_name=safe_name,
        job_type="train",
        tags=["sloughgpt", "api", "training"],
    )
    try:
        tracker = ExperimentTracker(config=tc)
        tracker.start_run(run_name=safe_name)
        flat = flatten_for_wandb_config(
            {
                "job": {"id": job_id, "name": job_name},
                "data": {"path": data_path},
                "trainer": hyperparams,
            }
        )
        tracker.log_params(flat)
        return tracker
    except Exception:
        import logging

        logging.getLogger("sloughgpt.tracking").warning(
            "W&B training tracker could not be started; continuing without experiment logging",
            exc_info=True,
        )
        return None
