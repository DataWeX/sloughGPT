"""
Thread-safe server state management.

Replaces module-level mutable globals in state.py with a Lock-protected
ServerState class. All reads and writes go through the singleton getter,
which ensures atomic access from any thread.
"""

from __future__ import annotations

from threading import Lock, RLock
from typing import Any, Optional, Callable, TypeVar
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AtomicRef:
    """Thread-safe reference to a single value with change listeners."""

    def __init__(self, initial: T, name: str = ""):
        self._value: T = initial
        self._lock = Lock()
        self._name = name
        self._listeners: list[Callable[[T, T], None]] = []
        self._version = 0

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        with self._lock:
            old = self._value
            self._value = value
            self._version += 1
        for listener in self._listeners:
            try:
                listener(old, value)
            except Exception as e:
                logger.warning("AtomicRef[%s] listener failed: %s", self._name, e)

    def swap(self, fn: Callable[[T], T]) -> T:
        """Atomically apply a function to the current value and return the new value."""
        with self._lock:
            old = self._value
            new = fn(old)
            self._value = new
            self._version += 1
        for listener in self._listeners:
            try:
                listener(old, new)
            except Exception as e:
                logger.warning("AtomicRef[%s] listener failed: %s", self._name, e)
        return new

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def on_change(self, listener: Callable[[T, T], None]) -> None:
        self._listeners.append(listener)


class ServerState:
    """Thread-safe singleton for all mutable server state.

    Usage::

        state = get_server_state()
        model = state.model.get()
        state.model.set(new_model)
    """

    def __init__(self) -> None:
        self.model = AtomicRef(None, "model")
        self.tokenizer = AtomicRef(None, "tokenizer")
        self.model_type = AtomicRef(None, "model_type")
        self.checkpoint = AtomicRef(None, "checkpoint")
        self.soul_engine = AtomicRef(None, "soul_engine")
        self.current_soul = AtomicRef(None, "current_soul")
        self.gen_config = AtomicRef(None, "gen_config")
        self.model_request_logger = AtomicRef(None, "model_request_logger")

        # Non-atomic fields (written once at startup, read-only after)
        self.torch_available: bool = False
        self.training_active: bool = False

        # Metrics
        self._started_at: float = time.time()
        self._request_count: int = 0
        self._error_count: int = 0
        self._lock = Lock()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def record_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count


_server_state: Optional[ServerState] = None
_server_state_lock = Lock()


def get_server_state() -> ServerState:
    """Get (or create) the singleton ServerState."""
    global _server_state
    if _server_state is None:
        with _server_state_lock:
            if _server_state is None:
                _server_state = ServerState()
    return _server_state
