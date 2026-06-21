"""
Middleware modules extracted from main.py.

All middleware re-exported via ``get_configured_middleware()`` for
registration in the FastAPI app.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("man.middleware")


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        if elapsed > 1.0:
            logger.info(
                "SLOW %s %s %d (%.2fs)",
                request.method, request.url.path, response.status_code, elapsed,
            )
        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a correlation ID for log tracing."""

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        corr_id = request.headers.get(self.HEADER) or request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.correlation_id = corr_id
        response = await call_next(request)
        response.headers[self.HEADER] = corr_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, duration, and correlation ID."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        corr_id = getattr(request.state, "correlation_id", "-")
        start = time.monotonic()
        try:
            response = await call_next(request)
            elapsed = time.monotonic() - start
            logger.debug(
                "%s %s %s %d (%.3fs)",
                corr_id, request.method, request.url.path, response.status_code, elapsed,
            )
            return response
        except Exception:
            elapsed = time.monotonic() - start
            logger.exception(
                "%s %s %s UNHANDLED (%.3fs)",
                corr_id, request.method, request.url.path, elapsed,
            )
            raise


def get_configured_middleware() -> list[type[BaseHTTPMiddleware]]:
    """Return middleware classes in registration order.

    Registration order: CorrelationId → Timing → RequestLogging.
    """
    return [
        CorrelationIdMiddleware,
        RequestTimingMiddleware,
        RequestLoggingMiddleware,
    ]


def register_all_middleware(app: FastAPI):
    """Register all middleware on a FastAPI instance."""
    for cls in get_configured_middleware():
        app.add_middleware(cls)
