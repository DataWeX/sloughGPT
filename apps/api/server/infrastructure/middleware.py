"""
Middleware modules extracted from main.py.

All middleware re-exported via ``get_configured_middleware()`` for
registration in the FastAPI app.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("man.middleware")

# Default server-side request timeout (seconds).
# Override via env var MAN_REQUEST_TIMEOUT in main.py.
REQUEST_TIMEOUT_SECONDS = 60.0


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Enforces a server-side per-request timeout.

    If a request handler takes longer than ``timeout`` seconds, the
    middleware aborts it and returns 504 Gateway Timeout.
    """

    def __init__(self, app, timeout: float = REQUEST_TIMEOUT_SECONDS):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout (%0.1fs) on %s %s",
                self.timeout, request.method, request.url.path,
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"error": f"Request timed out after {self.timeout}s"},
            )


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


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records every request to the Prometheus MetricsCollector."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        from domains.infrastructure.metrics import get_metrics_collector
        collector = get_metrics_collector()
        collector.set_active_requests(collector._active_requests + 1)
        start = time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - start
            collector.set_active_requests(max(0, collector._active_requests - 1))
            path = request.url.path
            collector.record_request(path, status_code, elapsed)


def get_configured_middleware(request_timeout: float = REQUEST_TIMEOUT_SECONDS) -> list[tuple[type[BaseHTTPMiddleware], dict]]:
    """Return middleware classes with kwargs in registration order.

    Registration order: RequestTimeout → Metrics → CorrelationId → Timing → RequestLogging.
    """
    return [
        (RequestTimeoutMiddleware, {"timeout": request_timeout}),
        (MetricsMiddleware, {}),
        (CorrelationIdMiddleware, {}),
        (RequestTimingMiddleware, {}),
        (RequestLoggingMiddleware, {}),
    ]


def register_all_middleware(app: FastAPI, request_timeout: float = REQUEST_TIMEOUT_SECONDS):
    """Register all middleware on a FastAPI instance.

    Includes RateLimitMiddleware from the rate limiter module.
    """
    for cls, kwargs in get_configured_middleware(request_timeout):
        app.add_middleware(cls, **kwargs)

    # Wire rate limiter middleware
    try:
        from domains.infrastructure.rate_limiter import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware)
        logger.info("RateLimitMiddleware registered")
    except Exception as exc:
        logger.warning("RateLimitMiddleware skipped: %s", exc)
