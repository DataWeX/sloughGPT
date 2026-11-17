"""Shared core wiring — one sequence, two consumers.

The startup phases in ``apps/api/server/infrastructure/startup.py`` and the
embeddable ``SloughInstance`` facade both wire the same core: config
validation + ResourceManager warm-up, task queue + handler registration,
agent routines, model registry — and their shutdowns. This module is the
single implementation both call; duplicating the sequence here is exactly
the training-loop debt (one loop, N copies) applied to lifecycle, so it is
forbidden by construction: the host wraps these calls with its progress
reporting (``STARTUP_PHASE``), the embedder reaches them through the facade.

Semantics are preserved verbatim from the host's original phase bodies:
every step is warn-and-continue (non-critical phases log and return, never
raise), and the log strings/levels are byte-identical.

``log`` is injected rather than module-global so the host keeps logging
under its ``slo.startup`` channel unchanged while the facade logs under
``slo.instance`` — one implementation, no forced logger migration.
"""

from __future__ import annotations

import logging
from typing import Any

_DEFAULT_LOG = logging.getLogger("slo.instance")


def wire_config(log: logging.Logger | None = None) -> Any:
    """Validate the config system and warm the ResourceManager.

    Returns the resolved ``AppConfig`` (or ``None`` if validation failed —
    the host ignores this; the facade passes it to the model loader seam).
    """
    log = log if log is not None else _DEFAULT_LOG
    cfg: Any = None
    try:
        from domain.infrastructure._internal.config import get_config

        cfg = get_config()
        _ = cfg.model.name
        log.info("Config system validated", extra={"tag": "START"})
    except Exception as e:
        log.warning("Config system init: %s", e, extra={"tag": "START"})
    # Init ResourceManager — applies BLAS env vars before numpy loads
    try:
        from domain.infrastructure.resource_manager import get_resource_manager

        rm = get_resource_manager()
        rm.apply_blas_env()
        rm.apply_compute_limits()
        rm.apply_environment()
        log.info(
            "ResourceManager initialised: mode=%s %s",
            rm.mode,
            rm.summary(),
            extra={"tag": "START"},
        )
    except Exception as e:
        log.warning("ResourceManager init: %s", e, extra={"tag": "START"})
    return cfg


async def wire_task_queue(log: logging.Logger | None = None) -> Any:
    """Start the background task queue and register its handlers.

    Returns the queue handle even when ``start()`` failed (the host stores
    it so shutdown can still attempt a stop — matching the original
    assignment order inside ``_phase_task_queue``).
    """
    log = log if log is not None else _DEFAULT_LOG
    queue: Any = None
    try:
        from domain.infrastructure.task_queue import get_task_queue

        queue = get_task_queue()
        await queue.start()
        log.info("Task queue initialized and started", extra={"tag": "START"})
    except Exception as e:
        log.warning("Task queue init failed: %s", e, extra={"tag": "START"})
    try:
        from domain.infrastructure.training_queue import register_training_handlers

        register_training_handlers()
        log.info("Training handlers registered with task queue", extra={"tag": "START"})
    except Exception as e:
        log.warning("Training handler registration failed: %s", e, extra={"tag": "START"})
    try:
        from domain.memory import register_memory_handlers

        register_memory_handlers()
        log.info("Memory handlers registered with task queue", extra={"tag": "START"})
    except Exception as e:
        log.warning("Memory handler registration failed: %s", e, extra={"tag": "START"})
    try:
        from domain.memory._internal.maintenance import start_memory_maintenance

        start_memory_maintenance()
        log.info("Memory maintenance scheduler started", extra={"tag": "START"})
    except Exception as e:
        log.warning("Memory maintenance scheduler start failed: %s", e, extra={"tag": "START"})
    return queue


async def unwire_task_queue(queue: Any, log: logging.Logger | None = None) -> None:
    """Stop memory maintenance, then gracefully drain and stop the queue."""
    log = log if log is not None else _DEFAULT_LOG
    if queue is None:
        return
    try:
        from domain.memory._internal.maintenance import stop_memory_maintenance

        await stop_memory_maintenance()
    except Exception as e:
        log.warning("Memory maintenance shutdown: %s", e, extra={"tag": "START"})
    try:
        await queue.stop()
        log.info("Task queue stopped", extra={"tag": "START"})
    except Exception as e:
        log.warning("Task queue shutdown: %s", e, extra={"tag": "START"})


def wire_agent_routines(log: logging.Logger | None = None) -> None:
    """Register the routine handler and start the background ticker."""
    log = log if log is not None else _DEFAULT_LOG
    try:
        from domain.agents import get_routine_runner

        runner = get_routine_runner()
        runner.register_handlers()
        runner.start()
        log.info(
            "Agent routine runner started (interval=%.1fs)",
            runner.tick_interval,
            extra={"tag": "START"},
        )
    except Exception as e:
        log.warning("Agent routine runner start failed: %s", e, extra={"tag": "START"})


def unwire_agent_routines(log: logging.Logger | None = None) -> None:
    """Stop the routine ticker only — in-flight runs belong to the queue."""
    log = log if log is not None else _DEFAULT_LOG
    try:
        from domain.agents import get_routine_runner

        get_routine_runner().stop()
        log.info("Agent routine runner stopped", extra={"tag": "START"})
    except Exception as e:
        log.warning("Agent routine runner shutdown: %s", e, extra={"tag": "START"})


def wire_model_registry(log: logging.Logger | None = None) -> Any:
    """Initialize the model registry; returns the registry or ``None``."""
    log = log if log is not None else _DEFAULT_LOG
    try:
        from domain.infrastructure.model_registry import get_model_registry

        registry = get_model_registry()
        log.info("Phase: model registry initialized", extra={"tag": "START"})
        return registry
    except Exception as e:
        log.warning("Phase: model registry failed: %s", e, extra={"tag": "START"})
        return None


def unwire_registry(log: logging.Logger | None = None) -> None:
    """Reset model registry metrics during shutdown."""
    log = log if log is not None else _DEFAULT_LOG
    try:
        from domain.infrastructure.model_registry import get_model_registry

        get_model_registry().reset_metrics()
    except Exception as e:
        log.debug("Registry reset failed during shutdown: %s", e)


def shutdown_executor(log: logging.Logger | None = None) -> None:
    """Gracefully shut down the TrainingExecutor thread pool."""
    log = log if log is not None else _DEFAULT_LOG
    try:
        from domain.training._internal.executor import _instance

        if _instance is not None:
            _instance.shutdown(wait=True)
            log.info("TrainingExecutor shut down", extra={"tag": "START"})
    except Exception as e:
        log.warning("TrainingExecutor shutdown: %s", e, extra={"tag": "START"})
