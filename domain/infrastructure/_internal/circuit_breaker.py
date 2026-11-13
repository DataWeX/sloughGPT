"""Circuit breaker pattern for fault tolerance."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


class CircuitBreakerState(Enum):
    """States of a circuit breaker: closed (normal), open (failing), half_open (testing)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Simple circuit breaker to stop sending requests to a failing model."""

    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    _state: CircuitBreakerState = CircuitBreakerState.CLOSED
    _failure_count: int = 0
    _last_failure_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)
    _on_state_change: Callable[[CircuitBreakerState, CircuitBreakerState], None] | None = None

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        old_state = self._state
        self._state = new_state
        if self._on_state_change and old_state != new_state:
            self._on_state_change(old_state, new_state)

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if time.time() - self._last_failure_at >= self.recovery_timeout:
                    self._transition_to(CircuitBreakerState.HALF_OPEN)
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._transition_to(CircuitBreakerState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._last_failure_at = time.time()
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._transition_to(CircuitBreakerState.OPEN)
                self._failure_count = self.failure_threshold
            else:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitBreakerState.OPEN)

    def allow_request(self) -> bool:
        return self.state != CircuitBreakerState.OPEN
