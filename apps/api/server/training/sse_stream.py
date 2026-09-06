"""Shared SSE streaming helper for training endpoints.

Encapsulates the common pattern used by both /training/stream and
/training/from-sessions-stream: TaskQueue enqueue, CancelManager
registration, TrainingRuntime registration, heartbeat, timeout,
disconnect handling, and cleanup.

Core would not know about HTTP. This module is in the API layer
(training/) because it handles FastAPI StreamingResponse and
request lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from starlette.responses import StreamingResponse

logger = logging.getLogger("slo")

# Re-export SSE helpers (available in API layer)
from infrastructure.sse_fallback import sse_error  # noqa: E402


def build_training_sse_response(
    request: Any,
    config: dict,
    *,
    task_name: str,
    task_type: str,
    cm_label: str,
    runtime_model: str = "sloughgpt",
    runtime_data_source: str = "",
    checkpoints_dir: Path,
    validate_fn: Any = None,
) -> StreamingResponse | dict:
    """Build an SSE StreamingResponse for a training stream.

    Args:
        request: FastAPI Request object.
        config: Training configuration dict.
        task_name: Name for the TaskQueue task.
        task_type: Type for the TaskQueue task (e.g. "training", "training-sessions").
        cm_label: Label for CancelManager registration.
        runtime_model: Model name for the runtime job record.
        runtime_data_source: Data source for the runtime job record.
        checkpoints_dir: Path to checkpoints directory.
        validate_fn: Optional callable that raises if config is invalid.

    Returns:
        StreamingResponse on success, or dict with error status.
    """
    if validate_fn:
        try:
            validate_fn(config)
        except ValueError as e:
            return {
                "status": "error",
                "error": str(e),
                "code": "E_STATE_IDLE",
                "http_status": 409,
            }

    cancel_event = threading.Event()

    async def _build() -> StreamingResponse:
        queue: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _enqueue(event_str: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event_str)

        from domains.infrastructure.task_queue import Priority, Task, get_task_queue

        tq = get_task_queue()
        await tq.start()
        task = Task(
            name=task_name,
            task_type=task_type,
            priority=Priority.HIGH,
            payload={**config, "checkpoint_dir": str(checkpoints_dir)},
            timeout=3600,
        )
        task.metadata["sse_queue"] = queue
        task.metadata["enqueue"] = _enqueue

        task_id = await tq.enqueue(task)
        logger.info(
            "Training task enqueued: %s (type=%s)", task_id, task_type, extra={"tag": "TRAIN"}
        )

        # Register with CancelManager
        cm_op_id: str | None = None
        try:
            from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

            _mgr = get_cancel_manager()
            cm_op_id = _mgr.register(
                op_type=OpType.TRAINING,
                label=cm_label,
                cancel_fn=lambda: cancel_event.set(),
            )
            _mgr.start(cm_op_id)
        except Exception as e:
            logger.warning("CancelManager registration failed for %s: %s", cm_label, e)

        # Register with TrainingRuntime
        from training.runtime import get_training_runtime

        runtime_job = {
            "id": task_id,
            "name": task_name,
            "model": runtime_model,
            "dataset": config.get("data_path") or "",
            "data_path": config.get("data_path") or "",
            "data_source": runtime_data_source or task_type,
            "status": "running",
            "progress": 0.0,
            "epochs": config.get("epochs"),
            "current_epoch": 0,
            "global_step": 0,
            "loss": None,
            "train_loss": None,
            "checkpoint": None,
            "checkpoint_dir": str(checkpoints_dir),
            "error": None,
            "experiment_id": config.get("experiment_id"),
        }
        get_training_runtime().register(task_id, runtime_job, None, dict(config))

        # Import service helpers (core layer, zero HTTP deps)
        from domains.training.service import (
            cleanup_stream_state,
            process_training_completion,
        )

        state_dict: dict[str, Any] = {"running": True}

        async def event_generator() -> AsyncGenerator[str, None]:
            def _finish_cm(status: str, error: str = "") -> None:
                if cm_op_id:
                    try:
                        from domains.infrastructure.cancel_manager import get_cancel_manager

                        get_cancel_manager().finish(
                            cm_op_id, error=error if status != "complete" else ""
                        )
                    except Exception as exc:
                        logger.debug("CancelManager.finish failed: %s", exc)

            deadline = time.time() + 3600
            heartbeat_interval = 10.0
            last_yield = time.time()
            try:
                while True:
                    if time.time() > deadline:
                        logger.error("Training SSE timed out after 1 hour", extra={"tag": "TRAIN"})
                        yield sse_error(
                            task_name,
                            "TIMEOUT",
                            "Training SSE stream timed out",
                            code="E_TIMEOUT",
                            http_status=408,
                        )
                        _finish_cm("failed", "SSE timeout")
                        return
                    if await request.is_disconnected():
                        await tq.cancel(task_id)
                        cancel_event.set()
                        state_dict["running"] = False
                        _finish_cm("cancelled", "client disconnected")
                        logger.info(
                            "Client disconnected from %s stream", task_name, extra={"tag": "TRAIN"}
                        )
                        return
                    remaining = heartbeat_interval - (time.time() - last_yield)
                    if remaining <= 0:
                        remaining = heartbeat_interval
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        last_yield = time.time()
                        continue
                    yield event
                    last_yield = time.time()
                    if event.startswith("data: "):
                        try:
                            ev = json.loads(event[6:])
                            if ev.get("status") in ("complete", "error"):
                                process_training_completion(
                                    ev, task_id, config, checkpoints_dir, _finish_cm
                                )
                                break
                        except json.JSONDecodeError:
                            pass

                while not queue.empty():
                    try:
                        extra = queue.get_nowait()
                        yield extra
                    except asyncio.QueueEmpty:
                        break

            except TimeoutError:
                logger.error("Training SSE queue timed out", extra={"tag": "TRAIN"})
                _finish_cm("failed", "SSE queue timeout")
                yield sse_error(
                    task_name,
                    "TIMEOUT",
                    "No training progress for 60 seconds",
                    code="E_TIMEOUT",
                    http_status=408,
                )
            except Exception as e:
                logger.error("Training SSE stream error: %s", e, extra={"tag": "TRAIN"})
                _finish_cm("failed", str(e))
                yield sse_error(
                    task_name, "FAILED", str(e), code="E_INFRA_GENERATION", http_status=500
                )
            finally:
                cleanup_stream_state(task_id, config, state_dict, _finish_cm)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    import asyncio as _asyncio

    return _asyncio.ensure_future(_build())


def stop_all_training() -> dict:
    """Stop all active training (auto-train, turbo, from-sessions).

    This is the canonical stop function for the training module.
    It delegates to CancelManager which tracks all active operations.
    """
    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

        get_cancel_manager().cancel_all(op_type=OpType.TRAINING)
    except Exception as e:
        logger.warning("CancelManager.cancel_all failed: %s", e)

    # Also signal service-layer cancel events
    try:
        from domains.training.service import (
            get_cancel_event,
            get_pgq,
            get_state,
            get_turbo_cancel_event,
            get_turbo_pause_event,
            get_turbo_state,
        )

        ev = get_cancel_event()
        if ev is not None:
            ev.set()
        tev = get_turbo_cancel_event()
        tev.set()
        tpause = get_turbo_pause_event()
        tpause.clear()
        # Try to cancel PGQ job
        pgq = get_pgq()
        if pgq is not None:
            turbo_state = get_turbo_state()
            job_id = turbo_state.get("job_id")
            if job_id:
                try:
                    pgq.cancel_training(job_id)
                except Exception as exc:
                    logger.warning("Failed to cancel PGQ turbo job %s: %s", job_id, exc)
        get_state().running = False
    except Exception:
        pass

    return {"status": "cancelling", "message": "Cancelling all training"}


def cancel_from_sessions() -> dict:
    """Cancel from-sessions training specifically."""
    try:
        from domains.infrastructure.cancel_manager import OpType, get_cancel_manager

        get_cancel_manager().cancel_all(op_type=OpType.TRAINING)
    except Exception as e:
        logger.warning("CancelManager.cancel_all failed: %s", e)

    try:
        from domains.training.service import get_cancel_event, get_state

        ev = get_cancel_event()
        if ev is not None:
            ev.set()
        get_state().running = False
    except Exception:
        pass

    return {"status": "cancelled", "message": "Cancel signal sent"}
