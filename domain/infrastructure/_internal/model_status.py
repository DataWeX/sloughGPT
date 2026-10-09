"""Model status enums and metrics collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class ModelStatus(Enum):
    """Lifecycle states of a ModelServer instance."""

    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelMetrics:
    """Metrics collected per-model-session."""

    requests_total: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_timed_out: int = 0
    total_generation_time_ms: float = 0.0
    max_generation_time_ms: float = 0.0
    min_generation_time_ms: float = float("inf")
    tokens_generated_total: int = 0
    last_generation_time_ms: float = 0.0
    last_error: str | None = None
    last_error_at: float | None = None
    consecutive_failures: int = 0
    last_request_time: float = 0.0

    def record_success(self, elapsed_ms: float, tokens: int) -> None:
        self.requests_completed += 1
        self.total_generation_time_ms += elapsed_ms
        self.max_generation_time_ms = max(self.max_generation_time_ms, elapsed_ms)
        self.min_generation_time_ms = min(self.min_generation_time_ms, elapsed_ms)
        self.last_generation_time_ms = elapsed_ms
        self.tokens_generated_total += tokens
        self.consecutive_failures = 0
        self.last_request_time = time.time()

    def record_failure(self, error: str) -> None:
        self.requests_failed += 1
        self.last_error = error
        self.last_error_at = time.time()
        self.consecutive_failures += 1
        self.last_request_time = time.time()

    def record_timeout(self) -> None:
        self.requests_timed_out += 1
        self.consecutive_failures += 1

    def reset(self) -> None:
        """Reset all counters to defaults."""
        self.__init__()

    @property
    def avg_generation_time_ms(self) -> float:
        if self.requests_completed == 0:
            return 0.0
        return self.total_generation_time_ms / self.requests_completed

    @property
    def error_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_failed / self.requests_total

    def snapshot(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "requests_timed_out": self.requests_timed_out,
            "consecutive_failures": self.consecutive_failures,
            "avg_generation_time_ms": round(self.avg_generation_time_ms, 1),
            "max_generation_time_ms": round(self.max_generation_time_ms, 1),
            "min_generation_time_ms": round(self.min_generation_time_ms, 1)
            if self.min_generation_time_ms != float("inf")
            else 0.0,
            "last_generation_time_ms": round(self.last_generation_time_ms, 1),
            "tokens_generated_total": self.tokens_generated_total,
            "last_error": self.last_error,
            "error_rate": round(self.error_rate, 4),
            "last_request_time": self.last_request_time,
        }
