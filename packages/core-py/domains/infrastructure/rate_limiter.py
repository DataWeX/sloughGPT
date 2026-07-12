"""
Rate limiter middleware for FastAPI.

Uses a sliding window counter per client IP.  Exceeds ``max_requests`` in
``window_seconds`` → 429 Too Many Requests.

Thread-safe via ``threading.Lock``.  No external dependencies (no Redis).
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
            # Prune expired entries
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

_limiter: RateLimiter = None


def get_rate_limiter(
    max_requests: int = 60,
    window_seconds: int = 60,
) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(max_requests, window_seconds)
    return _limiter


# ── FastAPI Middleware ────────────────────────────────────────────────


RATE_LIMIT_HEADER_REMAINING = "X-RateLimit-Remaining"
RATE_LIMIT_HEADER_LIMIT = "X-RateLimit-Limit"
RATE_LIMIT_HEADER_RESET = "X-RateLimit-Reset"


# ── FastAPI Middleware (BaseHTTPMiddleware) ──

from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting via sliding window counter.

    Applies to all routes.  Exceeding ``max_requests`` in ``window_seconds``
    returns 429 with ``Retry-After`` header.
    """

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, window_seconds)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = self.limiter.check(client_ip)

        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
                headers={
                    RATE_LIMIT_HEADER_REMAINING: "0",
                    RATE_LIMIT_HEADER_LIMIT: str(self.max_requests),
                    "Retry-After": str(self.window_seconds),
                },
            )

        response = await call_next(request)
        response.headers[RATE_LIMIT_HEADER_REMAINING] = str(remaining)
        response.headers[RATE_LIMIT_HEADER_LIMIT] = str(self.max_requests)
        return response
