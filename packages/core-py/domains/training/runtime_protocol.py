"""Training runtime protocol — defines the interface for job persistence.

Core modules depend on this protocol, not on the concrete ``TrainingRuntime``
implementation in the API layer. The API layer provides the implementation at
startup.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol
import threading


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


class _NoOpRuntime:
    """Stub runtime that does nothing — used before the real runtime is injected."""

    def register(self, job_id, job, cancel_event=None, config=None):
        pass

    def get(self, job_id):
        return None

    def sync(self, job_id):
        pass
