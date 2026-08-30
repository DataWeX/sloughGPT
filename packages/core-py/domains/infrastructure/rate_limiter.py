"""
Pure rate-limiter logic — sliding window counter, no HTTP dependencies.

HTTP middleware lives in apps/api/server/infrastructure/rate_limit_middleware.py.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, List, Tuple


class RateLimiter:
    """Sliding-window rate limiter keyed by client identifier (IP)."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str) -> Tuple[bool, int]:
        """
        Check if ``key`` has exceeded the rate limit.

        Returns ``(allowed, remaining)`` where ``allowed`` is True if the
        request should proceed, and ``remaining`` is the number of requests
        remaining in the current window.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._windows[key]
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)

            if len(timestamps) >= self.max_requests:
                return False, 0

            timestamps.append(now)
            remaining = self.max_requests - len(timestamps)
            return True, remaining

    def reset(self, key: str) -> None:
        """Clear all timestamps for ``key``."""
        with self._lock:
            self._windows.pop(key, None)


# ── Singleton ────────────────────────────────────────────────────────

_limiter: "RateLimiter | None" = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(
    max_requests: int = 60,
    window_seconds: int = 60,
) -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _rate_limiter_lock:
            if _limiter is None:
                _limiter = RateLimiter(max_requests, window_seconds)
    return _limiter


def reset_rate_limiter() -> None:
    """Reset the singleton (for testing)."""
    global _limiter
    with _rate_limiter_lock:
        _limiter = None


# ── Header constants (shared with middleware) ────────────────────────

RATE_LIMIT_HEADER_REMAINING = "X-RateLimit-Remaining"
RATE_LIMIT_HEADER_LIMIT = "X-RateLimit-Limit"
RATE_LIMIT_HEADER_RESET = "X-RateLimit-Reset"
