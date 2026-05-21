"""
Rate Limit Router - Rate limiting status and configuration
"""
from fastapi import APIRouter, Request
import time
from collections import defaultdict

router = APIRouter(prefix="/rate-limit", tags=["rate-limit"])


class _RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._history: dict = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window = 60.0
        self._history[key] = [t for t in self._history[key] if now - t < window]
        if len(self._history[key]) >= self.requests_per_minute:
            return False
        self._history[key].append(now)
        return True

    def get_wait_time(self, key: str) -> float:
        now = time.time()
        window = 60.0
        self._history[key] = [t for t in self._history[key] if now - t < window]
        if len(self._history[key]) < self.requests_per_minute:
            return 0.0
        return max(0.0, window - (now - self._history[key][0]))


_rate_limiter = _RateLimiter()


@router.get("/status")
async def get_rate_limit_status():
    """Get current rate limit configuration"""
    return {
        "requests_per_minute": _rate_limiter.requests_per_minute,
        "burst_size": _rate_limiter.burst_size,
        "enabled": True,
    }


@router.get("/check")
async def check_rate_limit(request: Request):
    """Check if request would be rate limited"""
    client_ip = request.client.host if request.client else "unknown"
    allowed = _rate_limiter.is_allowed(client_ip)
    return {
        "allowed": allowed,
        "wait_time": 0 if allowed else _rate_limiter.get_wait_time(client_ip),
    }
