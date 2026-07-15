"""Optional long-lived W&B run for FastAPI: HTTP aggregates + inference counters (flush interval)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

from domains.infrastructure.config import get_config

logger = logging.getLogger("man.wandb.server")

_inference_lock = threading.Lock()
_inference_total = 0
_inference_latency_sum = 0.0
_inference_tokens_sum = 0.0


def record_inference_call(latency_s: float, approx_tokens: float) -> None:
    """Accumulate inference stats for the next W&B flush (no-op if server W&B disabled)."""
    from domains.training.wandb_helpers import wandb_server_enabled_from_env

    if not wandb_server_enabled_from_env():
        return
    global _inference_total, _inference_latency_sum, _inference_tokens_sum
    with _inference_lock:
        _inference_total += 1
        _inference_latency_sum += latency_s
        _inference_tokens_sum += approx_tokens


def _drain_inference_snapshot() -> Dict[str, float]:
    global _inference_total, _inference_latency_sum, _inference_tokens_sum
    with _inference_lock:
        t = _inference_total
        if t == 0:
            return {}
        lat_mean = _inference_latency_sum / t
        tok_mean = _inference_tokens_sum / t
        _inference_total = 0
        _inference_latency_sum = 0.0
        _inference_tokens_sum = 0.0
    return {
        "inference/requests_since_flush": float(t),
        "inference/mean_latency_s": float(lat_mean),
        "inference/mean_approx_tokens": float(tok_mean),
    }


def _wandb_log_payload(payload: Dict[str, Any], step: int) -> None:
    import wandb

    wandb.log(payload, step=step)


def _wandb_init_server_run() -> None:
    import wandb

    from domains.training.wandb_helpers import default_wandb_project

    cfg = get_config().tracking
    wandb.init(
        project=default_wandb_project(),
        entity=cfg.wandb_entity or None,
        job_type="server",
        name=cfg.wandb_server_run_name or f"api-{os.getpid()}",
        tags=["sloughgpt", "fastapi", "server"],
        mode=cfg.wandb_mode,
        api_key=cfg.wandb_api_key or None,
    )


def _wandb_finish_run() -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logger.debug("wandb.finish failed", exc_info=True)


async def start_wandb_server_background(
    http_metrics: Any,
    *,
    extra_metrics: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Optional[asyncio.Task]:
    """Start periodic W&B logging when ``MAN_WANDB_SERVER`` is set. Returns task or ``None``.

    ``extra_metrics`` is called in a worker thread each flush (e.g. psutil host snapshot aligned with ``GET /info``).
    """
    from domains.training.wandb_helpers import wandb_server_enabled_from_env

    if not wandb_server_enabled_from_env():
        return None
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed; set MAN_WANDB_SERVER=0 or pip install wandb", extra={"tag": "INFRA"})
        return None

    interval = float(get_config().tracking.wandb_server_interval)

    async def _loop() -> None:
        await asyncio.to_thread(_wandb_init_server_run)
        step = 0
        try:
            while True:
                await asyncio.sleep(interval)
                http_part: Dict[str, float] = {}
                if hasattr(http_metrics, "wandb_aggregate"):
                    http_part = http_metrics.wandb_aggregate()
                inf_part = _drain_inference_snapshot()
                extra_part: Dict[str, Any] = {}
                if extra_metrics is not None:
                    try:
                        extra_part = await asyncio.to_thread(extra_metrics)
                    except Exception:
                        logger.debug("extra_metrics() failed", exc_info=True)
                payload = {**http_part, **inf_part, **extra_part}
                if payload:
                    await asyncio.to_thread(_wandb_log_payload, payload, step)
                step += 1
        finally:
            await asyncio.to_thread(_wandb_finish_run)

    return asyncio.create_task(_loop())
