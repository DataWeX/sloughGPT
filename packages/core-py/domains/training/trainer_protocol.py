"""
Minimal training interface (:class:`TrainerProtocol`).

Concrete trainers (e.g. :class:`~domains.training.train_pipeline.SloughGPTTrainer`) implement
``train()`` with this signature structurally — no inheritance required.

Async callers should use :func:`run_trainer_async` (runs ``train()`` in a worker thread).

Training entry points and checkpoint rules: ``docs/policies/CONTRIBUTING.md``.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class TrainerProtocol(Protocol):
    """Sync trainer: run a full fit, return a small result dict (e.g. ``best_eval_loss``, ``global_step``)."""

    def train(
        self,
        resume: bool = False,
        resume_path: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        ...


async def run_trainer_async(
    trainer: TrainerProtocol,
    *,
    resume: bool = False,
    resume_path: Optional[str] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run ``trainer.train`` in a worker thread (safe to ``await`` from asyncio)."""
    fn = functools.partial(
        trainer.train,
        resume=resume,
        resume_path=resume_path,
        on_progress=on_progress,
    )
    return await asyncio.to_thread(fn)


__all__ = ["TrainerProtocol", "run_trainer_async"]
