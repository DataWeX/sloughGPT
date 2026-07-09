"""
Rate Limiter — token bucket with per-endpoint, per-user, per-model limits.
Integrates with ErrorTaxonomy (ResourceExhaustedError) and EventBus.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("man.rate_limiter")


@dataclass
class TokenBucket:
    rate: float  # tokens per second
    burst: int   # max accumulated tokens
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    key: str = ""

    def __post_init__(self):
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()

    def refill(self) -> float:
        now = time.monotonic()
        elapsed = now - self.last_refill
        added = elapsed * self.rate
        self.tokens = min(float(self.burst), self.tokens + added)
        self.last_refill = now
        return self.tokens

    def try_consume(self, count: float = 1.0) -> bool:
        self.refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    def wait_time(self, count: float = 1.0) -> float:
        """How many seconds until `count` tokens are available."""
        if self.rate <= 0:
            return float("inf")
        self.refill()
        if self.tokens >= count:
            return 0.0
        deficit = count - self.tokens
        return deficit / self.rate

    @property
    def fill_pct(self) -> float:
        return self.tokens / self.burst * 100


class RateLimiter:
    """Token-bucket rate limiter with multiple dimensions and async support."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._event_bus = None
        self._try_event_bus()

    def _try_event_bus(self):
        try:
            from domains.infrastructure.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except Exception:
            pass

    # ── Bucket management ──

    def add_limit(self, key: str, rate: float, burst: int):
        """Add or update a rate limit for a key (endpoint, user, model)."""
        with self._lock:
            self._buckets[key] = TokenBucket(rate=rate, burst=burst, key=key)

    def remove_limit(self, key: str):
        with self._lock:
            self._buckets.pop(key, None)

    def get_bucket(self, key: str) -> TokenBucket | None:
        with self._lock:
            return self._buckets.get(key)

    def clear(self):
        with self._lock:
            self._buckets.clear()

    # ── Synchronous check ──

    def check(self, key: str, count: float = 1.0) -> bool:
        """Non-blocking check. Returns True if request is allowed."""
        bucket = self.get_bucket(key)
        if bucket is None:
            return True
        allowed = bucket.try_consume(count)
        if not allowed and self._event_bus:
            try:
                self._event_bus.emit_sync("rate_limit.exceeded", {
                    "key": key,
                    "retry_after": bucket.wait_time(count),
                    "bucket_fill": bucket.fill_pct,
                }, source="rate_limiter")
            except Exception:
                pass
        return allowed

    def wait_seconds(self, key: str, count: float = 1.0) -> float:
        """How long until `count` tokens are available."""
        bucket = self.get_bucket(key)
        if bucket is None:
            return 0.0
        return bucket.wait_time(count)

    # ── Stats ──

    def stats(self) -> list[dict[str, Any]]:
        result = []
        with self._lock:
            for key, bucket in self._buckets.items():
                result.append({
                    "key": key,
                    "rate": bucket.rate,
                    "burst": bucket.burst,
                    "tokens": round(bucket.tokens, 1),
                    "fill_pct": round(bucket.fill_pct, 1),
                })
        return result

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)


# ── Async rate limiter (for integration with async endpoints) ──

class AsyncRateLimiter(RateLimiter):
    """Adds async acquire() for awaiting token availability."""

    async def acquire(self, key: str, count: float = 1.0, timeout: float | None = None) -> bool:
        """Wait for a token. Returns True if acquired, False on timeout."""
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            if self.check(key, count):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)


# ── Predefined limits ──

DEFAULT_LIMITS: dict[str, tuple[float, int]] = {
    # key → (rate, burst)
    "endpoint:health": (10.0, 20),       # 10 req/s, burst 20
    "endpoint:chat": (5.0, 20),          # 5 req/s, burst 20
    "endpoint:generate": (3.0, 10),      # 3 req/s, burst 10
    "endpoint:training": (5.0, 20),      # 5 req/s, burst 20
    "endpoint:login": (0.5, 3),          # 1 req/2s, burst 3
    "endpoint:register": (0.2, 2),       # 1 req/5s, burst 2
    "model:inference": (2.0, 8),         # 2 inference/s, burst 8
}

# Env-var overrides: MAN_RATE_LIMIT__<KEY>__RATE / MAN_RATE_LIMIT__<KEY>__BURST
# Example: MAN_RATE_LIMIT__ENDPOINT:CHAT__RATE=10.0 MAN_RATE_LIMIT__ENDPOINT:CHAT__BURST=50
def _apply_rate_limit_env_overrides():
    prefix = "MAN_RATE_LIMIT__"
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        rest = env_key[len(prefix):]
        parts = rest.rsplit("__", 1)
        if len(parts) != 2:
            continue
        limit_key, field = parts[0].replace(":", ":"), parts[1].lower()
        if limit_key not in DEFAULT_LIMITS or field not in ("rate", "burst"):
            continue
        try:
            val = float(raw) if field == "rate" else int(raw)
            rate, burst = DEFAULT_LIMITS[limit_key]
            if field == "rate":
                DEFAULT_LIMITS[limit_key] = (val, burst)
            else:
                DEFAULT_LIMITS[limit_key] = (rate, val)
        except (ValueError, TypeError):
            pass

_apply_rate_limit_env_overrides()


# ── Singleton ──

_default_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter()
        for key, (rate, burst) in DEFAULT_LIMITS.items():
            _default_limiter.add_limit(key, rate, burst)
    return _default_limiter


def set_rate_limiter(limiter: RateLimiter):
    global _default_limiter
    _default_limiter = limiter


# ── FastAPI middleware ──

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware that checks rate limits per endpoint group."""

        def __init__(self, app, limiter: RateLimiter | None = None):
            super().__init__(app)
            self.limiter = limiter or get_rate_limiter()

        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            method = request.method

            endpoint_key = _path_to_endpoint_key(path)
            if endpoint_key:
                user_key = f"user:{request.client.host}" if request.client else None

                # Check endpoint limit
                if not self.limiter.check(f"endpoint:{endpoint_key}"):
                    wait = self.limiter.wait_seconds(f"endpoint:{endpoint_key}")
                    return self._rate_limited_response(request, wait)

                # Check per-user limit
                if user_key and self.limiter.get_bucket(user_key):
                    if not self.limiter.check(user_key):
                        wait = self.limiter.wait_seconds(user_key)
                        return self._rate_limited_response(request, wait)

            return await call_next(request)

        @staticmethod
        def _rate_limited_response(request: Request, wait: float) -> JSONResponse:
            """Build a 429 response with CORS headers so browsers see the error."""
            origin = request.headers.get("origin", "")
            headers = {
                "Retry-After": str(int(wait)),
                "Access-Control-Allow-Origin": origin or "*",
                "Access-Control-Allow-Credentials": "true",
            }
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": round(wait, 1),
                },
                headers=headers,
            )


    def _path_to_endpoint_key(path: str) -> str | None:
        parts = path.strip("/").split("/")
        if not parts:
            return None
        # Map common paths to endpoint group keys
        area = parts[0]
        mapping = {
            "health": "health",
            "chat": "chat",
            "inference": "generate",
            "training": "training",
            "auth": "login",
            "register": "register",
        }
        for prefix, key in mapping.items():
            if area == prefix or area.startswith(prefix):
                return key
        return None

except ImportError:
    class RateLimitMiddleware:  # type: ignore
        """Stub — Starlette not available."""
        def __init__(self, app, limiter=None):
            pass
