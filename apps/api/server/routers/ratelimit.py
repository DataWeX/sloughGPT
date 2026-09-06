"""
Rate Limit Router - Rate limiting status and configuration
"""
from fastapi import APIRouter, Request
import time
from collections import defaultdict

from schemas.common import success_response, classify_and_raise


class _RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._history: dict = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request from key is within rate limits."""
        try:
            now = time.time()
            window = 60.0
            self._history[key] = [t for t in self._history[key] if now - t < window]
            if len(self._history[key]) >= self.requests_per_minute:
                return False
            self._history[key].append(now)
            return True
        except Exception as e:
            classify_and_raise(e, source="ratelimit.is_allowed")

    def get_wait_time(self, key: str) -> float:
        """Get seconds until the key can make another request."""
        try:
            now = time.time()
            window = 60.0
            self._history[key] = [t for t in self._history[key] if now - t < window]
            if len(self._history[key]) < self.requests_per_minute:
                return 0.0
            return max(0.0, window - (now - self._history[key][0]))
        except Exception as e:
            classify_and_raise(e, source="ratelimit.get_wait_time")
class RatelimitRouter:
    """Rate Limit Router - Rate limiting status and configuration."""

    def __init__(self):
        self.router = APIRouter(prefix="/rate-limit", tags=["rate-limit"])
        self._rate_limiter = _RateLimiter()
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/status", endpoint=self.get_rate_limit_status, methods=["GET"])
        self.router.add_api_route(path="/check", endpoint=self.check_rate_limit, methods=["GET"])

    async def get_rate_limit_status(self) -> dict:
        """Get current rate limit configuration"""
        try:
            return success_response(data={
                "requests_per_minute": self._rate_limiter.requests_per_minute,
                "burst_size": self._rate_limiter.burst_size,
                "enabled": True,
            })
        except Exception as e:
            classify_and_raise(e, source="ratelimit.status")

    async def check_rate_limit(self, request: Request) -> dict:
        """Check if request would be rate limited"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            allowed = self._rate_limiter.is_allowed(client_ip)
            return success_response(data={
                "allowed": allowed,
                "wait_time": 0 if allowed else self._rate_limiter.get_wait_time(client_ip),
            })
        except Exception as e:
            classify_and_raise(e, source="ratelimit.check")


router = RatelimitRouter().router
