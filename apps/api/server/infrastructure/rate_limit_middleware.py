"""
FastAPI rate-limit middleware — delegates to core RateLimiter.

Core owns the sliding-window logic. This file owns only HTTP concerns:
BaseHTTPMiddleware, JSONResponse 429, header injection.

Supports per-workspace rate limiting when auth is enabled.
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

# Per-workspace rate limit multiplier (workspace_id -> multiplier)
# Default is 1x. Workspace admins can be granted higher limits.
_DEFAULT_WORKSPACE_LIMIT = 300  # requests per window


def _extract_workspace_from_token(token: str) -> str:
    """Best-effort extract workspace_id from JWT without full verification.

    This is for rate limiting only — not authentication.
    """
    try:
        import base64
        import json

        # JWT is header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            return ""

        # Decode payload (second part)
        payload = parts[1]
        # Add padding
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding

        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        return data.get("workspace_id", "")
    except Exception:
        return ""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP and per-workspace rate limiting via sliding window counter.

    Applies to all routes except health probes (always allowed).
    Localhost requests (127.0.0.1 / ::1) get 10x the limit.
    Expensive endpoints (chat/stream, inference, model load) get
    separate, stricter limits to prevent GPU OOM.
    When auth is enabled, rate limits are also tracked per workspace.
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
        # Per-workspace limiters: workspace_id -> RateLimiter
        self._workspace_limiters: dict[str, RateLimiter] = {}

    def _match_route(self, path: str) -> tuple[str | None, RateLimiter | None]:
        """Find matching route-specific limiter."""
        for prefix, limiter in self._route_limiters.items():
            if path.startswith(prefix):
                return prefix, limiter
        return None, None

    def _get_workspace_limiter(self, workspace_id: str) -> RateLimiter:
        """Get or create rate limiter for a workspace."""
        if workspace_id not in self._workspace_limiters:
            self._workspace_limiters[workspace_id] = RateLimiter(
                _DEFAULT_WORKSPACE_LIMIT, self.window_seconds
            )
        return self._workspace_limiters[workspace_id]

    def _extract_workspace_id(self, request) -> str:
        """Extract workspace_id from Authorization header JWT."""
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return ""
        token = auth_header[7:]
        return _extract_workspace_from_token(token)

    async def dispatch(self, request, call_next):
        path = request.url.path

        if path.startswith("/health"):
            return await call_next(request)

        # CORS preflight must pass through unmodified — any non-CORS response
        # (e.g. 429) returned before CORSMiddleware adds Access-Control headers
        # causes the browser to fire TypeError: NetworkError.
        if request.method == "OPTIONS":
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

        # Check workspace-specific limit
        workspace_id = self._extract_workspace_id(request)
        if workspace_id:
            ws_limiter = self._get_workspace_limiter(workspace_id)
            allowed, remaining = ws_limiter.check(workspace_id)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Workspace rate limit exceeded. Try again later."},
                    headers={
                        RATE_LIMIT_HEADER_REMAINING: "0",
                        RATE_LIMIT_HEADER_LIMIT: str(_DEFAULT_WORKSPACE_LIMIT),
                        "Retry-After": str(self.window_seconds),
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
