"""Modular cancellation system for all long-running operations.

Every cancellable operation (training, inference, download, import, etc.)
registers with the singleton CancelManager. Any caller can cancel by
operation ID or by type. Each operation provides its own cancel callback
so the manager stays type-agnostic.

Usage::

    from domains.infrastructure.cancel_manager import get_cancel_manager, OpType

    mgr = get_cancel_manager()

    # Register a training job
    op_id = mgr.register(
        op_type=OpType.TRAINING,
        label="distill-shakespeare",
        cancel_fn=lambda: cancel_event.set(),
    )

    # Cancel it
    mgr.cancel(op_id)

    # Cancel all inference streams
    mgr.cancel_all(op_type=OpType.INFERENCE)

    # List what's running
    for op in mgr.list_active():
        print(op.id, op.label, op.status)
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _emit_op_event(action: str, op: "Operation") -> None:
    """Emit an operation lifecycle event to the global EventBus."""
    try:
        from domains.infrastructure.event_bus import get_event_bus
        get_event_bus().emit_sync(
            "operations",
            {
                "action": action,
                "operation": op.to_dict(),
            },
            source="cancel_manager",
        )
    except Exception:
        pass  # Never let event emission break cancellation logic


# ── Operation types ────────────────────────────────────────────────────

class OpType(enum.Enum):
    """Categories of cancellable operations."""
    TRAINING = "training"
    INFERENCE = "inference"
    DOWNLOAD = "download"
    IMPORT = "import"
    BATCH = "batch"
    OTHER = "other"


class OpStatus(enum.Enum):
    """Lifecycle states of a registered operation."""
    REGISTERED = "registered"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Operation record ──────────────────────────────────────────────────

@dataclass
class Operation:
    """A single cancellable operation tracked by the manager."""
    id: str
    op_type: OpType
    label: str
    status: OpStatus
    cancel_fn: Callable[[], Any]
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.op_type.value,
            "label": self.label,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 2
            ),
            "error": self.error,
            "meta": self.meta,
        }


# ── CancelManager ─────────────────────────────────────────────────────

class CancelManager:
    """Thread-safe registry of cancellable operations.

    Thin layer: stores callbacks, fires them on cancel, tracks status.
    No domain logic — each caller provides its own ``cancel_fn``.
    """

    def __init__(self) -> None:
        self._ops: Dict[str, Operation] = {}
        self._lock = threading.Lock()

    # ── Register ──────────────────────────────────────────────────────

    def register(
        self,
        op_type: OpType,
        label: str,
        cancel_fn: Callable[[], Any],
        meta: Optional[Dict[str, Any]] = None,
        op_id: Optional[str] = None,
    ) -> str:
        """Register a new cancellable operation. Returns its ID."""
        oid = op_id or uuid.uuid4().hex[:12]
        with self._lock:
            self._ops[oid] = Operation(
                id=oid,
                op_type=op_type,
                label=label,
                status=OpStatus.REGISTERED,
                cancel_fn=cancel_fn,
                created_at=time.time(),
                meta=meta or {},
            )
        logger.debug("Registered %s operation: %s (%s)", op_type.value, label, oid)
        return oid

    def start(self, op_id: str) -> None:
        """Mark an operation as actively running."""
        with self._lock:
            op = self._ops.get(op_id)
            if op and op.status == OpStatus.REGISTERED:
                op.status = OpStatus.RUNNING
                op.started_at = time.time()

    def finish(self, op_id: str, error: Optional[str] = None) -> None:
        """Mark an operation as completed or failed."""
        with self._lock:
            op = self._ops.get(op_id)
            if op:
                op.status = OpStatus.FAILED if error else OpStatus.COMPLETED
                op.finished_at = time.time()
                op.error = error

    # ── Cancel ────────────────────────────────────────────────────────

    def cancel(self, op_id: str) -> bool:
        """Cancel a single operation by ID. Returns True if found and cancellable."""
        with self._lock:
            op = self._ops.get(op_id)
            if not op:
                return False
            if op.status not in (OpStatus.REGISTERED, OpStatus.RUNNING):
                return False
            op.status = OpStatus.CANCELLING
        # Fire callback outside lock
        try:
            op.cancel_fn()
        except Exception as exc:
            logger.warning("cancel_fn for %s raised: %s", op_id, exc)
            with self._lock:
                op.status = OpStatus.FAILED
                op.error = str(exc)
                op.finished_at = time.time()
            return False
        with self._lock:
            op.status = OpStatus.CANCELLED
            op.finished_at = time.time()
        logger.info("Cancelled %s operation: %s (%s)", op.op_type.value, op.label, op_id)
        return True

    def cancel_all(self, op_type: Optional[OpType] = None) -> List[str]:
        """Cancel all active operations, optionally filtered by type.

        Returns list of cancelled operation IDs.
        """
        with self._lock:
            targets = [
                oid for oid, op in self._ops.items()
                if op.status in (OpStatus.REGISTERED, OpStatus.RUNNING)
                and (op_type is None or op.op_type == op_type)
            ]
        cancelled = []
        for oid in targets:
            if self.cancel(oid):
                cancelled.append(oid)
        return cancelled

    # ── Query ─────────────────────────────────────────────────────────

    def get(self, op_id: str) -> Optional[Operation]:
        with self._lock:
            return self._ops.get(op_id)

    def list_active(self, op_type: Optional[OpType] = None) -> List[Operation]:
        """Return all non-terminal operations."""
        with self._lock:
            return [
                op for op in self._ops.values()
                if op.status in (OpStatus.REGISTERED, OpStatus.RUNNING, OpStatus.CANCELLING)
                and (op_type is None or op.op_type == op_type)
            ]

    def list_all(self, op_type: Optional[OpType] = None) -> List[Operation]:
        with self._lock:
            return [
                op for op in self._ops.values()
                if op_type is None or op.op_type == op_type
            ]

    def purge(self, max_age_s: float = 3600.0) -> int:
        """Remove finished operations older than max_age_s. Returns count removed."""
        cutoff = time.time() - max_age_s
        with self._lock:
            to_remove = [
                oid for oid, op in self._ops.items()
                if op.status in (OpStatus.COMPLETED, OpStatus.CANCELLED, OpStatus.FAILED)
                and (op.finished_at or 0) < cutoff
            ]
            for oid in to_remove:
                del self._ops[oid]
        return len(to_remove)

    def count(self, op_type: Optional[OpType] = None) -> Dict[str, int]:
        """Count operations by status."""
        with self._lock:
            result: Dict[str, int] = {}
            for op in self._ops.values():
                if op_type and op.op_type != op_type:
                    continue
                key = op.status.value
                result[key] = result.get(key, 0) + 1
            return result


# ── Singleton ──────────────────────────────────────────────────────────

_manager: Optional[CancelManager] = None
_manager_lock = threading.Lock()


def get_cancel_manager() -> CancelManager:
    """Get the global CancelManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = CancelManager()
    return _manager


def reset_cancel_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager
    with _manager_lock:
        _manager = None
