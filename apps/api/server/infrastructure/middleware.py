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
    [REQ]   - normal request log (level varies by status)
    [SLOW]  - slow request (>1s) log
    [INFRA] - infrastructure events (middleware registration, timeouts)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from domains.infrastructure.correlation import set_correlation_id
from domains.logging.config import set_request_id
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from schemas.common import error_response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("slo.middleware")

# Default server-side request timeout (seconds).
# Override via env var SLO_REQUEST_TIMEOUT in main.py.
REQUEST_TIMEOUT_SECONDS = 60.0

# Slow request threshold (seconds)
SLOW_THRESHOLD_SECONDS = 1.0

# Paths that are always slow during cold start - suppress SLOW log for these
_COLD_START_PATHS = frozenset(
    {
        "/health",
        "/health/stream",
        "/models",
        "/models/hf",
        "/souls",
        "/chat/sessions",
        "/training/jobs",
        "/system/stream",
    }
)

# Inference endpoints that require a loaded model
_INFERENCE_PATHS = frozenset(
    {"/chat", "/chat/stream", "/inference/generate", "/inference/generate/stream"}
)


def _model_ready() -> bool:
    """True when a model is actually materialized and ready for inference.

    Checks three sources because the lazy-guard autoload path stores the
    provider in the core ``ServerState`` singleton but leaves
    ``state.__dict__["model"]`` as ``None`` (module ``__setattr__`` is a
    no-op in CPython — writes go to ``__dict__`` directly).

    1. ``state.model`` — set by eager-load paths.
    2. ``state.provider._model`` — set when eager load materializes weights.
    3. Core ``ServerState.model.get()`` — set by the lazy-guard path.
    """
    try:
        import state as server_state

        if server_state.model is not None:
            return True
        provider = server_state.provider
        if provider is not None and getattr(provider, "_model", None) is not None:
            return True
        # Lazy-guard path: provider lives in the core ServerState singleton
        # but state.__dict__["model"] stays None.
        from domains.infrastructure.server_state import get_server_state

        core_model = get_server_state().model.get()
        if core_model is not None:
            return True
    except Exception:
        logger.debug("Model loaded check failed", exc_info=True)
    return False


def _get_startup_phase() -> dict:
    """Return the current startup phase info."""
    try:
        from startup_progress import STARTUP_PHASE

        return STARTUP_PHASE
    except Exception:
        logger.debug("Startup phase lookup failed", exc_info=True)
        return {"phase": "unknown", "step": 0, "total": 9, "message": "Starting..."}


class ReadinessGateMiddleware(BaseHTTPMiddleware):
    """Blocks inference requests until the model is loaded.

    Returns 503 with Retry-After header for /chat and /inference/*
    endpoints when the model is not ready. Includes startup phase
    info in the response body so the frontend can show progress.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path not in _INFERENCE_PATHS:
            return await call_next(request)

        # OPTIONS requests are CORS preflight — they never hit inference
        # logic and MUST succeed or the browser cannot make the real request.
        if request.method == "OPTIONS":
            return await call_next(request)

        if _model_ready():
            return await call_next(request)

        phase = _get_startup_phase()
        phase_name = phase.get("phase", "unknown")
        step = phase.get("step", 0)
        total = phase.get("total", 9)
        msg = phase.get("message", "Starting...")

        if phase_name in ("initializing", "unknown"):
            retry_after = 30
        elif phase_name == "ready":
            retry_after = 2
        else:
            remaining_steps = max(1, total - step)
            retry_after = min(30, remaining_steps * 10)

        corr_id = request.scope.get("correlation_id", "-")
        logger.warning(
            "readiness_gate: %s %s blocked (phase=%s step=%d/%d) retry_after=%ds corr=%s",
            request.method,
            path,
            phase_name,
            step,
            total,
            retry_after,
            corr_id,
            extra={
                "tag": "INFRA",
                "http": {"method": request.method, "path": path, "status": 503, "corr": corr_id},
            },
        )

        return JSONResponse(
            status_code=503,
            content=error_response(
                f"Model still loading — {msg} (step {step}/{total})",
                "E_MODEL_LOADING",
                details={"phase": phase_name, "step": step, "total": total, "message": msg},
            ),
            headers={"Retry-After": str(retry_after)},
        )


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Enforces a server-side per-request timeout.

    If a request handler takes longer than ``timeout`` seconds, the
    middleware aborts it and returns 504 Gateway Timeout.
    """

    def __init__(self, app, timeout: float = REQUEST_TIMEOUT_SECONDS):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            elapsed_str = f"{self.timeout:.3f}s"
            corr_id = request.scope.get("correlation_id", "-")
            logger.warning(
                "504 on %s %s (%s) corr=%s",
                request.method,
                request.url.path,
                elapsed_str,
                corr_id,
                extra={
                    "op": "http.request",
                    "ok": False,
                    "err": {
                        "code": "E_INFRA_TIMEOUT",
                        "msg": f"request timed out after {self.timeout}s",
                    },
                    "dur_ms": int(self.timeout * 1000),
                    "http": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": 504,
                        "corr": corr_id,
                    },
                },
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=error_response(
                    f"Request timed out after {self.timeout}s",
                    "E_INFRA_TIMEOUT",
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

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        corr_id = (
            request.headers.get(self.HEADER)
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())[:8]
        )
        request.scope["correlation_id"] = corr_id
        set_correlation_id(corr_id)
        set_request_id(corr_id)  # also set for logging contextvars
        response = await call_next(request)
        response.headers[self.HEADER] = corr_id
        return response


class UnifiedRequestMiddleware(BaseHTTPMiddleware):
    """Logs every request: method, path, status, duration, correlation ID.

    Produces one clean log line per request.  Log level is chosen by status
    code so that errors are visible immediately in stdout:

        5xx  -> logger.error   (tag REQ)
        4xx  -> logger.warning (tag REQ)
        >1s  -> logger.warning (tag SLOW)  - unless path is in _COLD_START_PATHS
        else -> logger.debug   (tag REQ)

    On unhandled exceptions the full traceback is logged via logger.exception.

    Error detail extraction is intentionally left to FastAPI exception
    handlers - they already produce structured JSON responses.  This
    middleware avoids reading response bodies because:
      * StreamingResponse has no pre-buffered body attribute.
      * Buffering the body to inspect it would break streaming endpoints.
      * Double-parsing what the handler already logged adds no value.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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
                method,
                path,
                elapsed_str,
                corr_id,
                extra={
                    "op": "http.request",
                    "ok": False,
                    "err": {"code": "E_UNHANDLED", "msg": "unhandled exception"},
                    "dur_ms": int(elapsed * 1000),
                    "http": {
                        "method": method,
                        "path": path,
                        "status": 500,
                        "corr": corr_id,
                    },
                },
            )
            raise

        elapsed = time.monotonic() - start
        sc = response.status_code
        elapsed_str = f"{elapsed:.3f}s"

        # Choose log level by status code.
        if sc >= 500:
            logger.error(
                "%d on %s %s (%s) corr=%s",
                sc,
                method,
                path,
                elapsed_str,
                corr_id,
                extra={
                    "op": "http.request",
                    "ok": False,
                    "dur_ms": int(elapsed * 1000),
                    "http": {"method": method, "path": path, "status": sc, "corr": corr_id},
                },
            )
        elif sc >= 400:
            logger.warning(
                "%d on %s %s (%s) corr=%s",
                sc,
                method,
                path,
                elapsed_str,
                corr_id,
                extra={
                    "op": "http.request",
                    "ok": False,
                    "dur_ms": int(elapsed * 1000),
                    "http": {"method": method, "path": path, "status": sc, "corr": corr_id},
                },
            )
        elif elapsed > SLOW_THRESHOLD_SECONDS:
            if path in _COLD_START_PATHS and elapsed < 60.0:
                logger.debug(
                    "cold-start %s %s %d (%s) corr=%s",
                    method,
                    path,
                    sc,
                    elapsed_str,
                    corr_id,
                    extra={
                        "op": "http.request",
                        "dur_ms": int(elapsed * 1000),
                        "http": {"method": method, "path": path, "status": sc, "corr": corr_id},
                    },
                )
            else:
                logger.warning(
                    "%s %s %d (%s) corr=%s",
                    method,
                    path,
                    sc,
                    elapsed_str,
                    corr_id,
                    extra={
                        "op": "http.request",
                        "dur_ms": int(elapsed * 1000),
                        "http": {"method": method, "path": path, "status": sc, "corr": corr_id},
                    },
                )
        else:
            logger.info(
                "%s %s %d (%s) corr=%s",
                method,
                path,
                sc,
                elapsed_str,
                corr_id,
                extra={
                    "op": "http.request",
                    "dur_ms": int(elapsed * 1000),
                    "http": {"method": method, "path": path, "status": sc, "corr": corr_id},
                },
            )

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records every request to the Prometheus MetricsCollector."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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


class PayloadLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request/response payloads at DEBUG level for debugging.

    Only active when root logger level is DEBUG. Truncates large bodies.
    Never buffers StreamingResponse (checks response class).
    """

    MAX_BODY_LOG = 2048  # chars

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not logger.isEnabledFor(logging.DEBUG):
            return await call_next(request)

        corr_id = request.scope.get("correlation_id", "-")
        path = request.url.path
        method = request.method

        # Read request body (non-streaming only)
        req_body = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                raw = await request.body()
                if raw:
                    req_body = raw[: self.MAX_BODY_LOG]
                    if len(raw) > self.MAX_BODY_LOG:
                        req_body += f"... ({len(raw)} bytes total)"
            except Exception:
                req_body = "<read error>"

        logger.debug(
            ">>> %s %s corr=%s body=%s",
            method,
            path,
            corr_id,
            req_body,
            extra={
                "op": "http.request",
                "http": {"method": method, "path": path, "corr": corr_id, "phase": "request"},
            },
        )

        response = await call_next(request)

        # Log response body for non-streaming responses only
        resp_body = None
        # Check if it's a StreamingResponse - skip body logging
        from starlette.responses import StreamingResponse

        if not isinstance(response, StreamingResponse):
            try:
                resp_body_raw = response.body if hasattr(response, "body") else None
                if resp_body_raw:
                    resp_body = resp_body_raw[: self.MAX_BODY_LOG]
                    if len(resp_body_raw) > self.MAX_BODY_LOG:
                        resp_body += f"... ({len(resp_body_raw)} bytes total)"
            except Exception:
                resp_body = "<read error>"

        logger.debug(
            "<<< %s %s %d corr=%s body=%s",
            method,
            path,
            response.status_code,
            corr_id,
            resp_body,
            extra={
                "op": "http.request",
                "http": {
                    "method": method,
                    "path": path,
                    "status": response.status_code,
                    "corr": corr_id,
                    "phase": "response",
                },
            },
        )

        return response


class ClientErrorFilterMiddleware(BaseHTTPMiddleware):
    """Adds a DEBUG note for client-side errors originating from browser extensions.

    Extension-injected scripts (crypto wallets, etc.) that fail don't indicate
    server problems.  The bulk suppression of these errors is handled by the
    ``_ClientExtensionFilter`` logging filter in ``main.py``; this middleware
    only emits a supplementary DEBUG line for extension-origin 4xx/5xx so the
    failure is traceable without polluting the error level.  Because it wraps
    the app outermost, it cannot alter the level of the UnifiedRequest log.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        origin = request.headers.get("origin", "")
        if "chrome-extension" in origin or "moz-extension" in origin:
            if response.status_code >= 400:
                logger.debug(
                    "Extension error suppressed: %s %s %d",
                    request.method,
                    request.url.path,
                    response.status_code,
                    extra={"op": "http.request"},
                )
        return response


def get_configured_middleware(
    request_timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> list[tuple[type[BaseHTTPMiddleware], dict]]:
    """Return middleware classes with kwargs in registration order.

    FastAPI/Starlette applies middleware in reverse registration order:
    the LAST ``add_middleware`` call wraps the app outermost and runs
    first on each request.  The list below is therefore the registration
    order, and the inbound request path is the reverse of it.

    Request path (inbound -> outbound):
        ClientErrorFilter -> CorrelationId -> ReadinessGate -> UnifiedRequest -> Metrics -> RequestTimeout -> handler

    CorrelationId MUST run before ReadinessGate so that the gate can
    include the correlation ID in its logs.  ReadinessGate MUST run
    before UnifiedRequest so blocked requests are logged by the gate.
    """
    return [
        (RequestTimeoutMiddleware, {"timeout": request_timeout}),
        (MetricsMiddleware, {}),
        (PayloadLoggingMiddleware, {}),
        (UnifiedRequestMiddleware, {}),
        (ReadinessGateMiddleware, {}),
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
        from infrastructure.rate_limit_middleware import RateLimitMiddleware

        app.add_middleware(RateLimitMiddleware)
        logger.info("RateLimitMiddleware registered", extra={"op": "infra.startup"})
    except Exception as exc:
        logger.warning("RateLimitMiddleware skipped: %s", exc, extra={"op": "infra.startup"})
