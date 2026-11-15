"""Shared helpers for training routes — _finish_job, _sloughgpt_trainer_kwds, _run_async.

Extracted to break circular imports between router.py and execution.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from .jobs import training_jobs

logger = logging.getLogger("slo")


def _run_async(coro: Coroutine) -> None:
    """Run an async coroutine without blocking the caller (fire-and-forget).

    Always dispatches to a daemon thread: calling ``asyncio.run()`` inline
    would hold the worker thread until every webhook delivery finishes,
    which stalls job finalization (observed: record stuck at 96% during
    fan-out). Failures are logged and swallowed.
    """

    def _target() -> None:
        try:
            asyncio.run(coro)
        except Exception as exc:
            logger.debug("Fire-and-forget coroutine failed: %s", exc)

    try:
        from domain.infrastructure._internal.fire_and_forget import get_pool

        get_pool().submit(_target)
    except Exception as exc:
        logger.debug("Fire-and-forget dispatch failed: %s", exc)


def notify_push(title: str, body: str, **kwargs: Any) -> None:
    """Fire-and-forget mobile push notification (never blocks the caller).

    Push delivery is synchronous HTTPS; called inline it holds training
    worker threads past completion. Dispatches via :func:`_run_async`.
    """
    try:
        from domain.mobile import get_notification_service

        service = get_notification_service()

        async def _push() -> None:
            await asyncio.to_thread(
                service.send_notification_sync, title=title, body=body, **kwargs
            )

        _run_async(_push())
    except Exception as exc:
        logger.debug("Push dispatch failed: %s", exc)


def _finish_job(job_id: str, status: str, error: str | None = None) -> None:
    """Set job status and notify CancelManager so operations store stays in sync."""
    job = training_jobs.get(job_id)
    if job is not None:
        training_jobs[job_id] = {**job, "status": status, **({"error": error} if error else {})}
    try:
        from domain.infrastructure.cancel_manager import OpStatus, get_cancel_manager

        mgr = get_cancel_manager()
        op = mgr.get(job_id)
        if op is not None and op.status not in (
            OpStatus.CANCELLED,
            OpStatus.COMPLETED,
            OpStatus.FAILED,
        ):
            mgr.finish(job_id, error=error or "")
    except Exception as exc:
        logger.warning("_finish_job CancelManager.finish failed for %s: %s", job_id, exc)


# BPE tokenizer wiring (algo="bpe"): build (or reuse) a cached TokenTree for
# the job's corpus so tokenized jobs train on BPE ids instead of inflating
# total_steps with char-level ids (card: char-level tokenization inflates
# total_steps). Keyed by corpus content hash — the same data never retrains
# the tokenizer.
_TOKEN_TREE_CACHE_ENV = "SLO_TOKEN_TREE_CACHE"
_TOKENIZER_PARAMS: dict[str, Any] = {
    "vocab_size": 512,
    "min_frequency": 1,
    "lowercase": False,
    "embed_dim": 0,
    "sample_cap": 400_000,
}


def _sample_corpus(text: str, cap: int) -> str:
    """Deterministic even-stride sample of ``text`` capped at ``cap`` chars."""
    if len(text) <= cap:
        return text
    step = -(-len(text) // cap)  # ceil division
    return text[::step]


def _tokenizer_cache_dir() -> Path:
    """Directory for corpus-trained token-tree caches (env-overridable)."""
    env = os.environ.get(_TOKEN_TREE_CACHE_ENV)
    if env:
        return Path(env)
    from domain.shared import find_repo_root

    return find_repo_root(Path(__file__).resolve()) / "data" / "token_trees" / "_cache"


def _corpus_token_tree(data_path: str) -> Any:
    """Train (or load from cache) the TokenTree for a training corpus."""
    from domain.training._internal.token_tree import TokenTree
    from domain.training._internal.train_pipeline import load_corpus_text

    text = load_corpus_text(data_path)
    digest = hashlib.sha256()
    digest.update(json.dumps(_TOKENIZER_PARAMS, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    base = _tokenizer_cache_dir() / f"corpus_{digest.hexdigest()[:32]}"
    try:
        return TokenTree.load(str(base))
    except (FileNotFoundError, OSError, ValueError):
        pass
    tree = TokenTree().train(
        _sample_corpus(text, int(_TOKENIZER_PARAMS["sample_cap"])),
        vocab_size=int(_TOKENIZER_PARAMS["vocab_size"]),
        min_frequency=int(_TOKENIZER_PARAMS["min_frequency"]),
        lowercase=bool(_TOKENIZER_PARAMS["lowercase"]),
        embed_dim=int(_TOKENIZER_PARAMS["embed_dim"]),
    )
    try:
        base.parent.mkdir(parents=True, exist_ok=True)
        tree.save(str(base))
    except Exception as exc:  # cache is best-effort
        logger.warning("Tokenizer cache save failed for %s: %s", base, exc)
    return tree


def _sloughgpt_trainer_kwds(
    req_snapshot: dict[str, Any], data_path: str | None = None
) -> dict[str, Any]:
    """Build ``SloughGPTTrainer`` keyword arguments from a request ``model_dump()``.

    ``data_path`` (never a kwarg — the trainer takes it positionally) lets
    ``algo="bpe"`` attach a corpus-trained TokenTree; without it the request
    degrades to the legacy char-level path.
    """
    device = req_snapshot.get("device")
    kwds = {
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
        "feed_refresh_interval": float(req_snapshot.get("feed_refresh_interval") or 0.0),
        "device": device if device is not None and str(device).strip() != "" else None,
    }
    if str(req_snapshot.get("algo") or "").lower() == "bpe" and data_path:
        kwds["tokenizer"] = _corpus_token_tree(data_path)
    return kwds
