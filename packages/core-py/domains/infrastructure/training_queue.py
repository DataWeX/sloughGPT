"""
Training Queue — Routes training jobs through the infrastructure TaskQueue.

Replaces raw ``loop.run_in_executor(None, worker)`` spawns with the TaskQueue's
managed worker pool, giving training jobs priority scheduling, pause/resume/cancel,
progress tracking, and EventBus integration.

The handler bridges the task queue's asyncio-based cancel/pause events to the
threading.Event objects expected by SloughGPTTrainer and train_from_sessions.
SSE events are pushed to an asyncio.Queue stored in ``task.metadata["sse_queue"]``
so the HTTP streaming layer can consume them directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional
from domains.shared import find_repo_root

logger = logging.getLogger("slo.training_queue")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())


def _json_safe_payload(o: Any) -> Any:
    """Recursively make a value JSON-serialisable for an SSE event payload.

    Non-finite floats (``inf``/``nan``) are replaced with ``None`` because the
    SSE envelope serialises with ``json.dumps`` (which rejects them), and a
    trainer with no eval run can legitimately produce ``perplexity=inf``.

    Args:
        o: Arbitrary value (dict, list, tuple, dataclass, scalar).

    Returns:
        A JSON-serialisable copy of ``o`` with non-finite floats replaced by ``None``.

    Side effects:
        None — the input is not mutated.
    """
    if isinstance(o, dict):
        return {k: _json_safe_payload(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe_payload(v) for v in o]
    if is_dataclass(o):
        return _json_safe_payload(asdict(o))
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


async def training_handler(task) -> dict:
    """
    TaskQueue handler for SloughGPTTrainer (standard training).

    Expects ``task.payload`` to contain SloughGPTTrainer config keys
    (data_path, n_embed, n_layer, n_head, block_size, dropout, batch_size,
    epochs, learning_rate, checkpoint_dir, early_stopping_patience, resume,
    resume_path).

    Pushes SSE events to ``task.metadata["sse_queue"]`` via the callback in
    ``task.metadata["enqueue"]``.
    """
    from domains.training.train_pipeline import SloughGPTTrainer, TrainerConfig
    from pathlib import Path

    payload = task.payload
    enqueue = task.metadata.get("enqueue")
    if enqueue is None:
        logger.error("training_handler: no enqueue callback in task metadata", extra={
            "task_id": task.id, "task_type": task.task_type,
            "metadata_keys": list(task.metadata.keys()),
        })
        return {"status": "failed", "error": "No enqueue callback"}

    # Bridge task queue asyncio events → threading events for the trainer.
    # Trainer semantics: pause_event.is_set() == paused (train_pipeline.train()
    # sleeps while set). Leave it cleared so training starts immediately.
    cancel_event = threading.Event()
    pause_event = threading.Event()
    pause_event.clear()  # Not paused by default

    async def _bridge_cancel():
        """Propagate task queue cancel → threading cancel_event."""
        while True:
            if task.cancel_event.is_set():
                cancel_event.set()
                return
            try:
                import domains.training.service as _svc
                at_cancel = _svc.get_cancel_event()
                if at_cancel is not None and at_cancel.is_set():
                    cancel_event.set()
                    return
            except Exception as exc:
                logger.debug("Cancel bridge auto_train import failed: %s", exc)
            await asyncio.sleep(0.1)

    async def _bridge_pause():
        """Propagate task queue pause → threading pause_event."""
        while True:
            paused = task.pause_event.is_set()
            try:
                import domains.training.service as _svc
                at_pause = _svc.get_pause_event()
                if at_pause is not None and at_pause.is_set():
                    paused = True
            except Exception as exc:
                logger.warning("Pause bridge auto_train event check failed: %s", exc)
                paused = True
            if paused:
                pause_event.clear()  # Paused
            else:
                pause_event.set()  # Resumed
            if task.cancel_event.is_set():
                return
            await asyncio.sleep(0.1)

    cancel_task = asyncio.create_task(_bridge_cancel())
    pause_task = asyncio.create_task(_bridge_pause())

    data_path = payload.get("data_path", "")
    checkpoint_dir = payload.get("checkpoint_dir", "")
    output_dir = Path(checkpoint_dir) if checkpoint_dir else _REPO_ROOT / "models" / "auto-training"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainerConfig(
        vocab_size=0,
        n_embed=payload.get("n_embed", 128),
        n_layer=payload.get("n_layer", 4),
        n_head=payload.get("n_head", 4),
        block_size=payload.get("block_size", 128),
        dropout=payload.get("dropout", 0.1),
        batch_size=payload.get("batch_size", 16),
        epochs=payload.get("epochs", 20),
        learning_rate=payload.get("learning_rate", 3e-4),
        checkpoint_dir=str(output_dir),
        early_stopping_patience=payload.get("early_stopping_patience", 5),
    )

    def _on_progress(info) -> None:
        if cancel_event.is_set():
            raise InterruptedError("Training cancelled by user")
        loss = info.get("train_loss")
        from domains.api.sse_envelope import sse_event
        enqueue(sse_event(
            "auto-train", "TRAIN", "working",
            data={
                "progress": info.get("progress_percent", 0),
                "loss": loss,
                "eval_loss": info.get("eval_loss"),
                "step": info.get("global_step", 0),
                "global_step": info.get("global_step", 0),
                "total_steps": info.get("total_steps", 0),
                "steps_per_sec": info.get("steps_per_sec", 0),
                "eta_s": info.get("eta_s"),
                "elapsed_s": info.get("elapsed_s"),
                "learning_rate": info.get("learning_rate", 0),
                "done": info.get("done", False),
                "done_reason": info.get("done_reason"),
                "avg_quality": info.get("avg_quality"),
            },
            meta={
                "epoch": info.get("epoch", 0),
                "total_epochs": info.get("epochs", 0),
            },
        ))

    try:
        # Constructor runs prepare_data() which can raise on a bad data_path —
        # keep it inside the try so the error surfaces as a failed result.
        trainer = SloughGPTTrainer(data_path=data_path, config=config)
        # train() is synchronous and long-running — run it in a worker thread
        # so the bridge coroutines keep the event loop alive and can propagate
        # cancel/pause from the task queue while training is in flight.
        result = await asyncio.to_thread(
            trainer.train,
            on_progress=_on_progress,
            cancel_event=cancel_event,
            pause_event=pause_event,
            resume=payload.get("resume", False),
            resume_path=payload.get("resume_path", ""),
        )
        if cancel_event.is_set():
            from domains.api.sse_envelope import sse_complete
            enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
            return {"status": "cancelled"}

        from domains.api.sse_envelope import sse_complete
        enqueue(sse_complete(
            "auto-train",
            data=_json_safe_payload(result),
            message="Training complete",
        ))
        return result
    except InterruptedError:
        from domains.api.sse_envelope import sse_complete
        enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
        return {"status": "cancelled"}
    except Exception as e:
        from domains.api.sse_envelope import sse_error
        enqueue(sse_error("auto-train", "FAILED", str(e)))
        return {"status": "failed", "error": str(e)}
    finally:
        cancel_task.cancel()
        pause_task.cancel()


def _resolve_checkpoint(name: Optional[str], checkpoint_dir: str) -> Optional[str]:
    """Resolve a checkpoint name to a full .soul file path.

    Args:
        name: Checkpoint name (e.g. ``"my-run"``) or already a full path.
        checkpoint_dir: Directory to search in.

    Returns:
        Full path string if found, else ``None``.
    """
    if not name:
        return None
    if Path(name).exists():
        return str(name)
    ckpt_dir = Path(checkpoint_dir)
    for candidate in [
        ckpt_dir / f"{name}.soul",
        ckpt_dir / name,
    ]:
        if candidate.exists():
            return str(candidate)
    logger.warning("Checkpoint not found: %s in %s", name, checkpoint_dir,
        extra={"tag": "TRAIN"})
    return None


async def training_sessions_handler(task) -> dict:
    """
    TaskQueue handler for train_from_sessions (chat-trained method).

    Expects ``task.payload`` to contain ChatTrainConfig keys.
    """
    from domains.training.chat_trainer import ChatTrainConfig, train_from_sessions

    payload = task.payload
    enqueue = task.metadata.get("enqueue")
    if enqueue is None:
        logger.error("training_sessions_handler: no enqueue callback in task metadata", extra={
            "task_id": task.id, "task_type": task.task_type,
            "metadata_keys": list(task.metadata.keys()),
        })
        return {"status": "failed", "error": "No enqueue callback"}

    cancel_event = threading.Event()

    async def _bridge_cancel():
        while True:
            if task.cancel_event.is_set():
                cancel_event.set()
                return
            try:
                import domains.training.service as _svc
                at_cancel = _svc.get_cancel_event()
                if at_cancel is not None and at_cancel.is_set():
                    cancel_event.set()
                    return
            except Exception as exc:
                logger.debug("Cancel bridge auto_train import failed: %s", exc)
            await asyncio.sleep(0.1)

    cancel_task = asyncio.create_task(_bridge_cancel())

    config = ChatTrainConfig(
        n_embed=payload.get("n_embed", 128),
        n_layer=payload.get("n_layer", 4),
        n_head=payload.get("n_head", 4),
        block_size=payload.get("block_size", 128),
        dropout=payload.get("dropout", 0.1),
        epochs=payload.get("epochs", 5),
        lr=payload.get("learning_rate", 3e-4),
        batch_size=payload.get("batch_size", 8),
        min_pair_quality=payload.get("min_pair_quality", 2.0),
        max_pairs=payload.get("max_pairs", 500),
        soul_name=payload.get("soul_name", "chat-trained"),
        checkpoint_dir=payload.get("checkpoint_dir", "models/auto-training"),
        session_ids=payload.get("session_ids"),
        resume_checkpoint=_resolve_checkpoint(
            payload.get("checkpoint_name"),
            payload.get("checkpoint_dir", "models/auto-training"),
        ),
    )

    _step_start = __import__("time").monotonic()

    def _on_step(step: int, loss: float, epoch: int, total_steps: int = 0) -> None:
        if cancel_event.is_set():
            raise InterruptedError("Training cancelled by user")
        from domains.api.sse_envelope import sse_event
        elapsed = __import__("time").monotonic() - _step_start
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        eta_s = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        enqueue(sse_event(
            "auto-train", "TRAIN", "working",
            data={"step": step, "loss": loss, "done": False,
                  "progress_percent": round(progress_pct, 1),
                  "steps_per_sec": round(steps_per_sec, 2),
                  "eta_s": round(eta_s, 1), "elapsed_s": round(elapsed, 1)},
            meta={"epoch": epoch, "total_epochs": config.epochs, "total_steps": total_steps},
        ))

    try:
        from domains.api.sse_envelope import sse_event
        enqueue(sse_event(
            "auto-train", "PAIRS", "working",
            message="Extracting chat pairs from sessions...",
        ))

        model, metadata = await asyncio.to_thread(
            train_from_sessions,
            config=config,
            on_step=_on_step,
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            from domains.api.sse_envelope import sse_complete
            enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
            return {"status": "cancelled"}
        from domains.api.sse_envelope import sse_complete
        enqueue(sse_complete(
            "auto-train",
            data=_json_safe_payload(metadata),
            message="Training complete",
        ))
        return metadata
    except InterruptedError:
        from domains.api.sse_envelope import sse_complete
        enqueue(sse_complete("auto-train", data={"cancelled": True}, message="Training cancelled"))
        return {"status": "cancelled"}
    except Exception as e:
        from domains.api.sse_envelope import sse_error
        enqueue(sse_error("auto-train", "FAILED", str(e)))
        return {"status": "failed", "error": str(e)}
    finally:
        cancel_task.cancel()


def register_training_handlers() -> None:
    """Register training handlers with the global task queue. Call once at startup."""
    from domains.infrastructure.task_queue import get_task_queue
    tq = get_task_queue()
    tq.register_handler("training", training_handler)
    tq.register_handler("training-sessions", training_sessions_handler)
    logger.info("Training handlers registered with task queue", extra={"tag": "INFRA"})


def unregister_training_handlers() -> None:
    """Unregister training handlers. Call at shutdown."""
    from domains.infrastructure.task_queue import get_task_queue
    tq = get_task_queue()
    tq.unregister_handler("training")
    tq.unregister_handler("training-sessions")
