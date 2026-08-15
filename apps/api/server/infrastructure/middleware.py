"""
Middleware modules extracted from main.py.

All middleware re-exported via ``get_configured_middleware()`` for
registration in the FastAPI app.

Unified log format (single line per request):
    HH:MM:SS INF [REQ] corr=abc1 GET /chat 200 (0.34s)
    HH:MM:SS WRN [REQ] corr=abc1 POST /multimodal/analyze 400 (19.61s) error="validation failed"
    HH:MM:SS ERR [REQ] corr=abc1 POST /chat 500 (2.10s) error="RuntimeError: ..."
    HH:MM:SS WRN [SLOW] corr=abc1 GET /models 200 (12.4s)

Type tags for quick scanning:
    [REQ]   — normal request log (level varies by status)
    [SLOW]  — slow request (>1s) log
    [INFRA] — infrastructure events (middleware registration, timeouts)
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

# Slow request threshold (seconds)
SLOW_THRESHOLD_SECONDS = 1.0

# Paths that are always slow during cold start — suppress SLOW log for these
_COLD_START_PATHS = {"/health", "/health/stream", "/models", "/models/hf", "/souls", "/chat/sessions", "/training/jobs"}

# Max bytes to capture from response body for error logging
_ERROR_BODY_MAX_BYTES = 512
# Max bytes to capture from request body for error logging
_ERROR_REQ_BODY_MAX_BYTES = 256


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


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a correlation ID for log tracing."""

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        corr_id = request.headers.get(self.HEADER) or request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.correlation_id = corr_id
        response = await call_next(request)
        response.headers[self.HEADER] = corr_id
        return response


class UnifiedRequestMiddleware(BaseHTTPMiddleware):
    """Single middleware that logs every request with timing, correlation, and error details.

    Replaces the old RequestTimingMiddleware + RequestLoggingMiddleware pair.
    Produces one clean log line per request:

        HH:MM:SS INF [REQ] abc1 GET /chat 200 (0.34s)
        HH:MM:SS WRN [REQ] abc1 POST /multimodal/analyze 400 (19.61s) error="validation failed"
        HH:MM:SS ERR [REQ] abc1 POST /chat 500 (2.10s) error="RuntimeError: oops"

    On 4xx/5xx, captures the response body (truncated) as the error field.
    On POST/PUT/PATCH errors, also captures the first N bytes of request body.
    On unhandled exceptions, logs the full traceback.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        corr_id = getattr(request.state, "correlation_id", "-")
        path = request.url.path
        method = request.method
        start = time.monotonic()

        # Capture request body for error debugging (must read before handler consumes it)
        req_body_preview = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                raw_body = await request.body()
                if raw_body:
                    req_body_preview = raw_body[:_ERROR_REQ_BODY_MAX_BYTES].decode("utf-8", errors="replace")
            except Exception:
                pass

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.monotonic() - start
            ctx = {
                "corr": corr_id,
                "method": method,
                "path": path,
                "status": 500,
                "elapsed": f"{elapsed:.3f}s",
            }
            if req_body_preview:
                ctx["req_body"] = req_body_preview
            logger.exception(
                "%s %s %s UNHANDLED (%.3fs)",
                corr_id, method, path, elapsed,
                extra={"tag": "REQ", "context": ctx},
            )
            raise

        elapsed = time.monotonic() - start
        sc = response.status_code
        elapsed_str = f"{elapsed:.3f}s"

        # --- Build error body for 4xx/5xx ---
        error_msg = None
        if sc >= 400:
            error_msg = self._extract_error_body(response)

        # --- Context dict for structured logging ---
        ctx = {
            "corr": corr_id,
            "method": method,
            "path": path,
            "status": sc,
            "elapsed": elapsed_str,
        }
        if error_msg:
            ctx["error"] = error_msg
        if req_body_preview and sc >= 400:
            ctx["req_body"] = req_body_preview

        # --- Log level by status code ---
        if sc >= 500:
            parts = [f"{corr_id} {method} {path} {sc} ({elapsed_str})"]
            if error_msg:
                parts.append(f'error="{error_msg}"')
            logger.error(
                " ".join(parts),
                extra={"tag": "REQ", "context": ctx},
            )
        elif sc >= 400:
            parts = [f"{corr_id} {method} {path} {sc} ({elapsed_str})"]
            if error_msg:
                parts.append(f'error="{error_msg}"')
            logger.warning(
                " ".join(parts),
                extra={"tag": "REQ", "context": ctx},
            )
        elif elapsed > SLOW_THRESHOLD_SECONDS:
            # Slow but successful — separate [SLOW] tag for grep-ability
            if path in _COLD_START_PATHS and elapsed < 60.0:
                logger.debug(
                    "%s %s %s %d (%s) cold-start",
                    corr_id, method, path, sc, elapsed_str,
                    extra={"tag": "REQ", "context": ctx},
                )
            else:
                logger.warning(
                    "%s %s %s %d (%s)",
                    corr_id, method, path, sc, elapsed_str,
                    extra={"tag": "SLOW", "context": ctx},
                )
        else:
            logger.debug(
                "%s %s %s %d (%s)",
                corr_id, method, path, sc, elapsed_str,
                extra={"tag": "REQ", "context": ctx},
            )

        return response

    @staticmethod
    def _extract_error_body(response: Response) -> str | None:
        """Safely read and truncate the response body for error logging.

        For StreamingResponse the body is not seekable — returns None.
        For JSONResponse / HTMLResponse / etc. reads up to _ERROR_BODY_MAX_BYTES.
        """
        body = getattr(response, "body", None)
        if body is None:
            return None
        try:
            raw = body if isinstance(body, bytes) else body.encode("utf-8", errors="replace")
            text = raw[:_ERROR_BODY_MAX_BYTES].decode("utf-8", errors="replace")
            if len(raw) > _ERROR_BODY_MAX_BYTES:
                text += "..."

            # Try to parse JSON and extract "detail" or "error" field
            try:
                parsed = json.loads(raw[:_ERROR_BODY_MAX_BYTES])
                if isinstance(parsed, dict):
                    detail = parsed.get("detail") or parsed.get("error") or parsed.get("message")
                    if detail:
                        return str(detail)[:256]
            except (json.JSONDecodeError, ValueError):
                pass

            return text if text else None
        except Exception:
            return None


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

    Registration order (FastAPI applies in reverse — last in list runs first on request):
        ClientErrorFilter → UnifiedRequest → CorrelationId → Metrics → RequestTimeout
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
