"""infra-lib — Type-agnostic infrastructure primitives.

Zero-internal-dependency modules extracted from sloughGPT for reuse.

Modules:
    cancel_manager — Track and cancel long-running operations
    server_state   — Thread-safe atomic state with listener notifications
"""

from .cancel_manager import (
    CancelManager,
    Operation,
    OpStatus,
    OpType,
    get_cancel_manager,
    reset_cancel_manager,
)
from .server_state import AtomicRef, ServerState, get_server_state

__all__ = [
    # cancel_manager
    "CancelManager",
    "OpStatus",
    "OpType",
    "Operation",
    "get_cancel_manager",
    "reset_cancel_manager",
    # server_state
    "AtomicRef",
    "ServerState",
    "get_server_state",
]
