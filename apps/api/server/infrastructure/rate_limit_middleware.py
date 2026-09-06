"""
FastAPI rate-limit middleware — delegates to core RateLimiter.

Core owns the sliding-window logic. This file owns only HTTP concerns:
BaseHTTPMiddleware, JSONResponse 429, header injection.
"""

from domains.infrastructure.rate_limiter import (
    RATE_LIMIT_HEADER_LIMIT,
    RATE_LIMIT_HEADER_REMAINING,
    RateLimiter,
)
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Per-route rate limits: path prefix -> (max_requests, window_seconds)
# Expensive GPU/CPU endpoints get much lower limits.
_ROUTE_LIMITS: dict[str, tuple[int, int]] = {
    "/chat/stream": (10, 60),
    "/chat/voice": (5, 60),
    "/inference/generate/stream": (10, 60),
    "/inference/generate": (20, 60),
    "/inference/embed": (30, 60),
    "/models/load": (2, 120),
    "/models/unload": (2, 120),
    "/training/": (3, 60),
    "/mobile/train": (2, 120),
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting via sliding window counter.

    Applies to all routes except health probes (always allowed).
    Localhost requests (127.0.0.1 / ::1) get 10x the limit.
    Expensive endpoints (chat/stream, inference, model load) get
    separate, stricter limits to prevent GPU OOM.
    Exceeding ``max_requests`` in ``window_seconds`` returns 429 with
    ``Retry-After`` header.
    """

    def __init__(self, app, max_requests: int = 300, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._global_limiter = RateLimiter(max_requests, window_seconds)
        self._local_global_limiter = RateLimiter(max_requests * 10, window_seconds)
        self._route_limiters: dict[str, RateLimiter] = {}
        for prefix, (limit, window) in _ROUTE_LIMITS.items():
            self._route_limiters[prefix] = RateLimiter(limit, window)

    def _match_route(self, path: str) -> tuple[str | None, RateLimiter | None]:
        """Find matching route-specific limiter."""
        for prefix, limiter in self._route_limiters.items():
            if path.startswith(prefix):
                return prefix, limiter
        return None, None

    async def dispatch(self, request, call_next):
        path = request.url.path

        if path.startswith("/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        is_local = client_ip in ("127.0.0.1", "::1", "localhost")

        # Check route-specific limit first (stricter)
        route_prefix, route_limiter = self._match_route(path)
        if route_limiter is not None:
            allowed, remaining = route_limiter.check(f"{client_ip}:{route_prefix}")
            if not allowed:
                route_limit = _ROUTE_LIMITS[route_prefix][0]
                route_window = _ROUTE_LIMITS[route_prefix][1]
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit exceeded for {route_prefix}. Try again later."},
                    headers={
                        RATE_LIMIT_HEADER_REMAINING: "0",
                        RATE_LIMIT_HEADER_LIMIT: str(route_limit),
                        "Retry-After": str(route_window),
                    },
                )

        # Check global limit
        if is_local:
            allowed, remaining = self._local_global_limiter.check(client_ip)
        else:
            allowed, remaining = self._global_limiter.check(client_ip)

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
