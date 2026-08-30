"""
FastAPI rate-limit middleware — delegates to core RateLimiter.

Core owns the sliding-window logic. This file owns only HTTP concerns:
BaseHTTPMiddleware, JSONResponse 429, header injection.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from domains.infrastructure.rate_limiter import (
    RateLimiter,
    RATE_LIMIT_HEADER_REMAINING,
    RATE_LIMIT_HEADER_LIMIT,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting via sliding window counter.

    Applies to all routes except health probes (always allowed).
    Localhost requests (127.0.0.1 / ::1) get 10x the limit.
    Exceeding ``max_requests`` in ``window_seconds`` returns 429 with
    ``Retry-After`` header.
    """

    _EXEMPT_PREFIXES = ("/health", "/health/live", "/health/ready", "/health/startup-progress")

    def __init__(self, app, max_requests: int = 300, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, window_seconds)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._local_limiter = RateLimiter(max_requests * 10, window_seconds)

    async def dispatch(self, request, call_next):
        path = request.url.path

        if path.startswith("/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        is_local = client_ip in ("127.0.0.1", "::1", "localhost")

        if is_local:
            allowed, remaining = self._local_limiter.check(client_ip)
        else:
            allowed, remaining = self.limiter.check(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
                headers={
                    RATE_LIMIT_HEADER_REMAINING: "0",
                    RATE_LIMIT_HEADER_LIMIT: str(self.max_requests * (10 if is_local else 1)),
                    "Retry-After": str(self.window_seconds),
                },
            )

        response = await call_next(request)
        response.headers[RATE_LIMIT_HEADER_REMAINING] = str(remaining)
        response.headers[RATE_LIMIT_HEADER_LIMIT] = str(self.max_requests * (10 if is_local else 1))
        return response
