"""Training runtime protocol — defines the interface for job persistence.

Core modules depend on this protocol, not on the concrete ``TrainingRuntime``
implementation in the API layer. The API layer provides the implementation at
startup.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class TrainingRuntimeProtocol(Protocol):
    """Interface for job registration, persistence, and lifecycle management."""

    def register(
        self,
        job_id: str,
        job: dict[str, Any],
        cancel_event: Optional[threading.Event] = ...,
        config: Optional[Dict[str, Any]] = ...,
    ) -> None: ...

    def get(self, job_id: str) -> Optional[dict[str, Any]]: ...

    def sync(self, job_id: str) -> None: ...


_runtime: Optional[TrainingRuntimeProtocol] = None


def set_training_runtime(runtime: TrainingRuntimeProtocol) -> None:
    """Register the runtime implementation (called once at startup)."""
    global _runtime
    _runtime = runtime


def get_training_runtime() -> TrainingRuntimeProtocol:
    """Return the registered runtime, or a no-op stub if none registered."""
    if _runtime is not None:
        return _runtime
    return _NoOpRuntime()


def update_job(job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    """Get a job, update fields, and sync. Returns the job or None."""
    job = get_training_runtime().get(job_id)
    if job is None:
        return None
    job.update(fields)
    get_training_runtime().sync(job_id)
    return job


class _NoOpRuntime:
    """Stub runtime that does nothing — used before the real runtime is injected."""

    def __init__(self):
        logger.warning(
            "Using _NoOpRuntime — training jobs will not be persisted. "
            "Ensure TrainingRuntime is injected at startup."
        )

    def register(self, job_id, job, cancel_event=None, config=None):
        logger.warning("NoOpRuntime.register(%s) — job discarded", job_id)

    def get(self, job_id):
        logger.debug("NoOpRuntime.get(%s) — returning None", job_id)
        return None

    def sync(self, job_id):
        logger.debug("NoOpRuntime.sync(%s) — no-op", job_id)
