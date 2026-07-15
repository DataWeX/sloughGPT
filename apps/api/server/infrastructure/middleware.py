"""
Middleware modules extracted from main.py.

All middleware re-exported via ``get_configured_middleware()`` for
registration in the FastAPI app.

Output format uses type tags for quick visual scanning:
    HH:MM:SS INF [REQ]  man.middleware GET /chat 200  (0.34s)
    HH:MM:SS WRN [SLOW] man.middleware SLOW GET /models 200  (12.4s)
    HH:MM:SS ERR [REQ]  man.middleware POST /chat 500  ← RuntimeError
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("slo.middleware")

# Default server-side request timeout (seconds).
# Override via env var SLO_REQUEST_TIMEOUT in main.py.
REQUEST_TIMEOUT_SECONDS = 60.0

# Paths that are always slow during cold start — suppress SLOW log for these
_COLD_START_PATHS = {"/health", "/models", "/souls", "/chat/sessions", "/training/jobs"}


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
                extra={"tag": "REQ", "context": {"status": 504, "timeout_s": self.timeout}, "error_code": "E_INFRA_TIMEOUT"},
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": f"Request timed out after {self.timeout}s",
                    "code": "E_INFRA_TIMEOUT",
                },
            )


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, status, and duration with type tags."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        if elapsed > 1.0:
            path = request.url.path
            sc = response.status_code
            ctx = {"method": request.method, "path": path, "status": sc, "elapsed": f"{elapsed:.2f}s"}

            # Suppress SLOW log for cold-start paths on first load
            if path in _COLD_START_PATHS and elapsed < 30.0:
                logger.debug("COLD %s %s %d (%.2fs)", request.method, path, sc, elapsed)
            elif sc >= 500:
                logger.error(
                    "SLOW %s %s %d (%.2fs)", request.method, path, sc, elapsed,
                    extra={"tag": "REQ", "context": ctx, "error_code": "E_INFRA_TIMEOUT"},
                )
            elif sc >= 400:
                logger.warning(
                    "SLOW %s %s %d (%.2fs)", request.method, path, sc, elapsed,
                    extra={"tag": "REQ", "context": ctx},
                )
            else:
                logger.info(
                    "SLOW %s %s %d (%.2fs)", request.method, path, sc, elapsed,
                    extra={"tag": "REQ", "context": ctx},
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
            ctx = {
                "corr": corr_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed": f"{elapsed:.3f}s",
            }
            if response.status_code >= 500:
                logger.error(
                    "%s %s %s %d (%.3fs)",
                    corr_id, request.method, request.url.path, response.status_code, elapsed,
                    extra={"tag": "REQ", "context": ctx},
                )
            elif response.status_code >= 400:
                logger.warning(
                    "%s %s %s %d (%.3fs)",
                    corr_id, request.method, request.url.path, response.status_code, elapsed,
                    extra={"tag": "REQ", "context": ctx},
                )
            else:
                logger.debug(
                    "%s %s %s %d (%.3fs)",
                    corr_id, request.method, request.url.path, response.status_code, elapsed,
                    extra={"tag": "REQ", "context": ctx},
                )
            return response
        except Exception:
            elapsed = time.monotonic() - start
            logger.exception(
                "%s %s %s UNHANDLED (%.3fs)",
                corr_id, request.method, request.url.path, elapsed,
                extra={"tag": "REQ", "context": {"corr": corr_id, "status": 500, "elapsed": f"{elapsed:.3f}s"}},
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


class ClientErrorFilterMiddleware(BaseHTTPMiddleware):
    """Filters out client-side errors from browser extensions (crypto wallets, etc.).

    Extension-injected scripts that fail don't indicate server problems.
    Logs them at DEBUG level instead of ERROR to reduce noise.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        origin = request.headers.get("origin", "")
        if "chrome-extension" in origin or "moz-extension" in origin:
            if response.status_code >= 400:
                logger.debug(
                    "Extension error suppressed: %s %s %d",
                    request.method, request.url.path, response.status_code,
                )
        return response


def get_configured_middleware(request_timeout: float = REQUEST_TIMEOUT_SECONDS) -> list[tuple[type[BaseHTTPMiddleware], dict]]:
    """Return middleware classes with kwargs in registration order.

    Registration order: RequestTimeout → Metrics → CorrelationId → Timing → RequestLogging → ClientErrorFilter.
    """
    return [
        (RequestTimeoutMiddleware, {"timeout": request_timeout}),
        (MetricsMiddleware, {}),
        (CorrelationIdMiddleware, {}),
        (RequestTimingMiddleware, {}),
        (RequestLoggingMiddleware, {}),
        (ClientErrorFilterMiddleware, {}),
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
        logger.info("RateLimitMiddleware registered", extra={"tag": "INFRA"})
    except Exception as exc:
        logger.warning("RateLimitMiddleware skipped: %s", exc, extra={"tag": "INFRA"})
