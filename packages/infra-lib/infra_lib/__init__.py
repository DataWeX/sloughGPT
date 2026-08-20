"""infra-lib — Type-agnostic infrastructure primitives.

Zero-internal-dependency modules extracted from sloughGPT for reuse.

Modules:
    cancel_manager — Track and cancel long-running operations
    task_queue     — Async priority queue with pause/resume/cancel/dependencies
    server_state   — Thread-safe atomic state with listener notifications
"""

from .cancel_manager import (
    CancelManager,
    OpStatus,
    OpType,
    Operation,
    get_cancel_manager,
    reset_cancel_manager,
)
from .server_state import AtomicRef, ServerState, get_server_state
from .task_queue import (
    InProcessTaskQueue,
    Priority,
    Task,
    TaskQueue,
    TaskStatus,
    WorkerPool,
)

__all__ = [
    # cancel_manager
    "CancelManager",
    "OpStatus",
    "OpType",
    "Operation",
    "get_cancel_manager",
    "reset_cancel_manager",
    # task_queue
    "InProcessTaskQueue",
    "Priority",
    "Task",
    "TaskQueue",
    "TaskStatus",
    "WorkerPool",
    # server_state
    "AtomicRef",
    "ServerState",
    "get_server_state",
]
