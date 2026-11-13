"""Retry, timing, caching, and rate-limiting helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

__all__ = ["Cache", "RateLimiter", "Timer", "retry"]

logger = logging.getLogger("slo.shared")


def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Retry a function up to *max_attempts* times, sleeping *delay* seconds.

    The final attempt re-raises instead of swallowing the error.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning("Attempt %d failed: %s", attempt + 1, e, extra={"tag": "INFRA"})
                    time.sleep(delay)

        return wrapper

    return decorator


class Timer:
    """Context manager that records elapsed wall-clock seconds."""

    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start


class Cache:
    """Fixed-size in-memory cache with FIFO eviction."""

    def __init__(self, max_size: int = 100):
        self._cache = {}
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class RateLimiter:
    """Allow at most *max_calls* invocations per *period* seconds."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [c for c in self.calls if now - c < self.period]

            if len(self.calls) >= self.max_calls:
                raise Exception(
                    f"Rate limit exceeded. Max {self.max_calls} calls per {self.period}s"
                )

            self.calls.append(now)
            return func(*args, **kwargs)

        return wrapper
