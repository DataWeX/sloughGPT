"""Delegating shim — the canonical TaskQueue lives in ``domain`` (card 44229b2a).

History: this module was a standalone fork of
``domain/infrastructure/_internal/task_queue.py`` (extraction 6f6cba8e5,
relocation d7f4e3ef6), kept dependency-free for reuse outside the repo.
Card 44229b2a collapses the duplication: the domain implementation is the
single body and this module re-exports it (shared-copy rule: never fork the
toolchain).

Behaviour changes for former fork callers (canonical wins on every drift point):
- ``TaskStatus`` is now ``enum.StrEnum`` (was ``str, Enum``). Comparisons and
  value lookup are unchanged; ``str()`` and f-strings render the bare value.
- The module logger namespace is now ``slo.task_queue`` (was
  ``infra_lib.task_queue``), and log records carry the canonical
  ``extra={"tag": "INFRA"}``.
- ``num_workers`` defaults to ``None`` and resolves through the resource
  manager (``SLO_TASK_QUEUE_WORKERS`` / cpu-derived) instead of a hard-coded
  2; passing an explicit int behaves as before.
- The task event bus is auto-wired at construction from
  ``domain.infrastructure._internal.event_bus`` (the fork's manual
  ``set_event_bus`` injection helper is gone; no in-repo caller used it).
- ``get_task_queue()`` takes no ``num_workers`` argument; build an explicit
  ``InProcessTaskQueue(num_workers=n)`` and install it with ``set_task_queue()``.

NOTE: this reintroduces an internal dependency (infra-lib -> domain) for this
module, superseding the package's original "zero-internal-dependency" claim —
that contract change is a deliberate landing decision recorded on card 44229b2a.
"""

from __future__ import annotations

from domain.infrastructure._internal.task_queue import (
    InProcessTaskQueue,
    Priority,
    Task,
    TaskQueue,
    TaskStatus,
    WorkerPool,
    get_task_queue,
    set_task_queue,
)

__all__ = [
    "InProcessTaskQueue",
    "Priority",
    "Task",
    "TaskQueue",
    "TaskStatus",
    "WorkerPool",
    "get_task_queue",
    "set_task_queue",
]
