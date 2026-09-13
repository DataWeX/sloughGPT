"""
AI-Native Interrupt System — completion signals, data arrival, gradient updates.

Not hardware interrupts. These are software signals that notify the kernel
when an asynchronous operation completes or needs attention.
"""

from __future__ import annotations

import threading
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Any


class InterruptType(IntEnum):
    """AI-native interrupt types."""
    TIMER = 0         # scheduler tick
    INFERENCE_DONE = 1   # model inference completed
    TRAINING_STEP = 2    # one training step completed
    DATA_READY = 3       # data loaded and ready
    GRADIENT_UPDATE = 4  # gradient computed, ready for optimizer step
    DEVICE_ERROR = 5     # device/driver error
    MEMORY_FULL = 6      # out of tensor memory
    PROCESS_DONE = 7     # process completed
    USER_INPUT = 8       # user sent input
    NETWORK_IO = 9       # network data arrived
    CUSTOM = 10          # user-defined interrupt


@dataclass
class Interrupt:
    """A single interrupt event."""
    vector: InterruptType
    source_pid: int | None = None
    data: Any = None
    priority: int = 0  # lower = higher priority


class InterruptVector:
    """
    Interrupt vector table — maps interrupt types to handlers.

    Each interrupt type can have one handler. Handlers are called
    synchronously when the interrupt fires.
    """

    def __init__(self):
        self._handlers: dict[InterruptType, Callable[[Interrupt], None]] = {}
        self._pending: list[Interrupt] = []
        self._lock = threading.Lock()
        self._masked: set[InterruptType] = set()
        self._history: list[Interrupt] = []
        self._max_history = 1000

    def register(self, vector: InterruptType,
                 handler: Callable[[Interrupt], None]) -> None:
        """Register a handler for an interrupt type."""
        with self._lock:
            self._handlers[vector] = handler

    def unregister(self, vector: InterruptType) -> None:
        """Unregister a handler."""
        with self._lock:
            self._handlers.pop(vector, None)

    def mask(self, vector: InterruptType) -> None:
        """Mask (disable) an interrupt type."""
        with self._lock:
            self._masked.add(vector)

    def unmask(self, vector: InterruptType) -> None:
        """Unmask (enable) an interrupt type."""
        with self._lock:
            self._masked.discard(vector)

    def is_masked(self, vector: InterruptType) -> bool:
        with self._lock:
            return vector in self._masked

    def fire(self, interrupt: Interrupt) -> bool:
        """
        Fire an interrupt. Returns True if handled, False if masked/no handler.
        Always records in history.
        """
        handled = False
        with self._lock:
            if interrupt.vector not in self._masked:
                handler = self._handlers.get(interrupt.vector)
                if handler is not None:
                    handled = True

        # Call handler outside lock to avoid deadlock
        if handled:
            try:
                handler(interrupt)
            except Exception:
                handled = False

        # Record history
        with self._lock:
            self._history.append(interrupt)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        return handled

    def enqueue(self, interrupt: Interrupt) -> None:
        """Enqueue an interrupt for deferred processing."""
        with self._lock:
            self._pending.append(interrupt)
            self._pending.sort(key=lambda i: i.priority)

    def dequeue(self) -> Interrupt | None:
        """Dequeue the highest-priority pending interrupt."""
        with self._lock:
            if self._pending:
                return self._pending.pop(0)
            return None

    def process_pending(self) -> int:
        """Process all pending interrupts. Returns count handled."""
        handled = 0
        while True:
            interrupt = self.dequeue()
            if interrupt is None:
                break
            if self.fire(interrupt):
                handled += 1
        return handled

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def history(self) -> list[Interrupt]:
        with self._lock:
            return list(self._history)

    def stats(self) -> dict:
        with self._lock:
            return {
                "registered_handlers": len(self._handlers),
                "masked_vectors": len(self._masked),
                "pending_interrupts": len(self._pending),
                "total_fired": len(self._history),
                "handlers": [v.name for v in self._handlers],
            }


class InterruptManager:
    """
    Top-level interrupt manager — owns the vector table and provides
    convenience methods for common interrupt patterns.
    """

    def __init__(self):
        self.vector = InterruptVector()

    def on_inference_done(self, handler: Callable[[Interrupt], None]) -> None:
        self.vector.register(InterruptType.INFERENCE_DONE, handler)

    def on_training_step(self, handler: Callable[[Interrupt], None]) -> None:
        self.vector.register(InterruptType.TRAINING_STEP, handler)

    def on_process_done(self, handler: Callable[[Interrupt], None]) -> None:
        self.vector.register(InterruptType.PROCESS_DONE, handler)

    def on_device_error(self, handler: Callable[[Interrupt], None]) -> None:
        self.vector.register(InterruptType.DEVICE_ERROR, handler)

    def on_memory_full(self, handler: Callable[[Interrupt], None]) -> None:
        self.vector.register(InterruptType.MEMORY_FULL, handler)

    def signal_inference_done(self, pid: int, result: Any = None) -> None:
        self.vector.fire(Interrupt(
            vector=InterruptType.INFERENCE_DONE,
            source_pid=pid,
            data=result,
        ))

    def signal_process_done(self, pid: int, result: Any = None) -> None:
        self.vector.fire(Interrupt(
            vector=InterruptType.PROCESS_DONE,
            source_pid=pid,
            data=result,
        ))

    def signal_device_error(self, pid: int, error: str) -> None:
        self.vector.fire(Interrupt(
            vector=InterruptType.DEVICE_ERROR,
            source_pid=pid,
            data=error,
        ))

    def stats(self) -> dict:
        return self.vector.stats()
