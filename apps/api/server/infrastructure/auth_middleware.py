"""Auth Middleware — middleware-level JWT enforcement.

Instead of relying on per-route ``Depends(require_auth_if_enabled)``, this
middleware checks JWT tokens on every request (except allowlisted paths) and
attaches the decoded user payload to ``request.state.user``.

Routes can then access ``request.state.user`` directly or use the
``get_current_user`` dependency to retrieve it.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("slo.auth.middleware")

# Paths that never require auth
_ALLOWLIST = frozenset(
    {
        "/health",
        "/health/stream",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/metrics",
        "/metrics/prometheus",
    }
)

# Path prefixes that never require auth
_ALLOWLIST_PREFIXES = (
    "/.well-known/",
    "/docs/",
    "/redoc/",
)

# Compiled pattern for auth-exempt paths
_AUTH_EXEMPT_RE = re.compile(
    r"^(/auth/login|/auth/register|/auth/token|/auth/verify|/auth/refresh)"
)


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware.

    When ``SLO_AUTH_REQUIRED=true``, validates the Bearer token on every
    request (except allowlisted paths) and sets ``request.state.user``.

    When disabled, sets ``request.state.user = None`` for all requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if auth is enabled
        auth_required = os.environ.get("SLO_AUTH_REQUIRED", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        if not auth_required:
            request.state.user = None
            return await call_next(request)

        path = request.url.path

        # Allowlist check
        if path in _ALLOWLIST:
            request.state.user = None
            return await call_next(request)

        for prefix in _ALLOWLIST_PREFIXES:
            if path.startswith(prefix):
                request.state.user = None
                return await call_next(request)

        # Auth-exempt paths (login, register, token)
        if _AUTH_EXEMPT_RE.match(path):
            request.state.user = None
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": True,
                    "code": "E_AUTH_MISSING",
                    "message": "Missing or invalid Authorization header",
                },
            )

        token = auth_header[7:]
        try:
            from infrastructure.auth import get_jwt_auth

            payload = get_jwt_auth().verify_token(token)
            request.state.user = payload
        except Exception:
            return JSONResponse(
                status_code=401,
                content={
                    "error": True,
                    "code": "E_AUTH_MISSING",
                    "message": "Invalid or expired token",
                },
            )

        return await call_next(request)


def get_current_user_from_state(request: Request) -> dict | None:
    """FastAPI dependency that returns the user from middleware-set state.

    Usage:
        @router.get("/something")
        async def handler(user: dict | None = Depends(get_current_user_from_state)):
            ...
    """
    return getattr(request.state, "user", None)


def require_user_from_state(request: Request) -> dict:
    """Like ``get_current_user_from_state`` but raises 401 if missing.

    Usage:
        @router.get("/something")
        async def handler(user: dict = Depends(require_user_from_state)):
            ...
    """
    user = getattr(request.state, "user", None)
    if user is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": True,
                "code": "E_AUTH_MISSING",
                "message": "Authentication required",
            },
        )
    return user
