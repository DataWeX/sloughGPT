"""
Middleware modules extracted from main.py.

All middleware re-exported via ``get_configured_middleware()`` for
registration in the FastAPI app.

Unified log format (single line per request):
    HH:MM:SS INF  [REQ]   GET /chat 200 (0.34s) corr=abc1
    HH:MM:SS WRN  [REQ]   400 on POST /multimodal/analyze (19.61s) corr=abc1
    HH:MM:SS ERR  [REQ]   500 on POST /chat (2.10s) corr=abc1
    HH:MM:SS WRN  [SLOW]  GET /models 200 (12.4s) corr=abc1

Type tags for quick scanning:
    [REQ]   — normal request log (level varies by status)
    [SLOW]  — slow request (>1s) log
    [INFRA] — infrastructure events (middleware registration, timeouts)
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

from schemas.common import error_response, set_correlation_id

logger = logging.getLogger("slo.middleware")

# Default server-side request timeout (seconds).
# Override via env var SLO_REQUEST_TIMEOUT in main.py.
REQUEST_TIMEOUT_SECONDS = 60.0

# Slow request threshold (seconds)
SLOW_THRESHOLD_SECONDS = 1.0

# Paths that are always slow during cold start — suppress SLOW log for these
_COLD_START_PATHS = frozenset({"/health", "/health/stream", "/models", "/models/hf", "/souls", "/chat/sessions", "/training/jobs"})


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
            corr_id = request.scope.get("correlation_id", "-")
            logger.warning(
                "request timeout after %0.1fs on %s %s corr=%s",
                self.timeout, request.method, request.url.path, corr_id,
                extra={"tag": "REQ", "context": {"status": 504, "timeout_s": self.timeout}},
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=error_response(
                    f"Request timed out after {self.timeout}s",
                    "E_INFRA_TIMEOUT",
                    correlation_id=corr_id,
                ),
            )


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a correlation ID for log tracing.

    Stores the ID in ``request.scope["correlation_id"]`` (not ``request.state``)
    so that downstream middleware running inside the same ``BaseHTTPMiddleware``
    chain can read it.  ``request.state`` is per-middleware-layer in Starlette
    and does NOT propagate through ``call_next``.
    """

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        corr_id = request.headers.get(self.HEADER) or request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.scope["correlation_id"] = corr_id
        set_correlation_id(corr_id)
        response = await call_next(request)
        response.headers[self.HEADER] = corr_id
        return response


class UnifiedRequestMiddleware(BaseHTTPMiddleware):
    """Logs every request: method, path, status, duration, correlation ID.

    Produces one clean log line per request.  Log level is chosen by status
    code so that errors are visible immediately in stdout:

        5xx  → logger.error   (tag REQ)
        4xx  → logger.warning (tag REQ)
        >1s  → logger.warning (tag SLOW)  — unless path is in _COLD_START_PATHS
        else → logger.debug   (tag REQ)

    On unhandled exceptions the full traceback is logged via logger.exception.

    Error detail extraction is intentionally left to FastAPI exception
    handlers — they already produce structured JSON responses.  This
    middleware avoids reading response bodies because:
      * StreamingResponse has no pre-buffered body attribute.
      * Buffering the body to inspect it would break streaming endpoints.
      * Double-parsing what the handler already logged adds no value.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        corr_id = request.scope.get("correlation_id", "-")
        path = request.url.path
        method = request.method
        start = time.monotonic()

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.monotonic() - start
            elapsed_str = f"{elapsed:.3f}s"
            logger.exception(
                "unhandled exception on %s %s (%s) corr=%s",
                method, path, elapsed_str, corr_id,
                extra={
                    "tag": "REQ",
                    "context": {
                        "corr": corr_id,
                        "method": method,
                        "path": path,
                        "status": 500,
                        "elapsed": elapsed_str,
                    },
                },
            )
            raise

        elapsed = time.monotonic() - start
        sc = response.status_code
        elapsed_str = f"{elapsed:.3f}s"

        # Structured context for log aggregators (Datadog, ELK, etc.)
        ctx = {
            "corr": corr_id,
            "method": method,
            "path": path,
            "status": sc,
            "elapsed": elapsed_str,
        }

        # Choose log level by status code.
        if sc >= 500:
            logger.error(
                "%d on %s %s (%s) corr=%s",
                sc, method, path, elapsed_str, corr_id,
                extra={"tag": "REQ", "context": ctx},
            )
        elif sc >= 400:
            logger.warning(
                "%d on %s %s (%s) corr=%s",
                sc, method, path, elapsed_str, corr_id,
                extra={"tag": "REQ", "context": ctx},
            )
        elif elapsed > SLOW_THRESHOLD_SECONDS:
            if path in _COLD_START_PATHS and elapsed < 60.0:
                logger.debug(
                    "cold-start %s %s %d (%s) corr=%s",
                    method, path, sc, elapsed_str, corr_id,
                    extra={"tag": "REQ", "context": ctx},
                )
            else:
                logger.warning(
                    "%s %s %d (%s) corr=%s",
                    method, path, sc, elapsed_str, corr_id,
                    extra={"tag": "SLOW", "context": ctx},
                )
        else:
            logger.debug(
                "%s %s %d (%s) corr=%s",
                method, path, sc, elapsed_str, corr_id,
                extra={"tag": "REQ", "context": ctx},
            )

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records every request to the Prometheus MetricsCollector."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        from domains.infrastructure.metrics import get_metrics_collector
        collector = get_metrics_collector()
        collector.set_active_requests(collector.get_active_requests() + 1)
        start = time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.monotonic() - start
            collector.set_active_requests(max(0, collector.get_active_requests() - 1))
            path = request.url.path
            collector.record_request(path, status_code, elapsed)


class ClientErrorFilterMiddleware(BaseHTTPMiddleware):
    """Adds a DEBUG note for client-side errors originating from browser extensions.

    Extension-injected scripts (crypto wallets, etc.) that fail don't indicate
    server problems.  The bulk suppression of these errors is handled by the
    ``_ClientExtensionFilter`` logging filter in ``main.py``; this middleware
    only emits a supplementary DEBUG line for extension-origin 4xx/5xx so the
    failure is traceable without polluting the error level.  Because it wraps
    the app outermost, it cannot alter the level of the UnifiedRequest log.
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

    FastAPI/Starlette applies middleware in reverse registration order:
    the LAST ``add_middleware`` call wraps the app outermost and runs
    first on each request.  The list below is therefore the registration
    order, and the inbound request path is the reverse of it.

    Request path (inbound → outbound):
        ClientErrorFilter → CorrelationId → UnifiedRequest → Metrics → RequestTimeout → handler

    CorrelationId MUST run before UnifiedRequest inbound so that
    ``request.scope["correlation_id"]`` is populated when UnifiedRequest logs.
    """
    return [
        (RequestTimeoutMiddleware, {"timeout": request_timeout}),
        (MetricsMiddleware, {}),
        (UnifiedRequestMiddleware, {}),
        (CorrelationIdMiddleware, {}),
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
