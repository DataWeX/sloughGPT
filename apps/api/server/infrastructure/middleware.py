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

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from schemas.common import error_response
from starlette.middleware.base import BaseHTTPMiddleware

from domain.infrastructure.correlation import set_correlation_id
from domain.logging._internal.config import set_request_id

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

# Prefixes that lazy-train a component on their first request. Suppressed only
# during the warm-up window after process start so a genuinely slow route
# still surfaces once the server is warm.
_COLD_START_PREFIXES = frozenset({"/token-tree/"})
_WARMUP_SECONDS = 120.0
_SERVER_START = time.monotonic()

# Inference endpoints that require a loaded model
_INFERENCE_PATHS = frozenset(
    {
        "/chat",
        "/chat/stream",
        "/inference/generate",
        "/inference/generate/stream",
        "/infer",
        "/infer/stream",
    }
)


def _model_ready() -> bool:
    """True when a model is actually materialized and ready for inference.

    Checks multiple sources because different load paths store the model
    in different locations:
    1. ``state.model`` — set by eager-load paths.
    2. ``state.provider._model`` — set when eager load materializes weights.
    3. Core ``ServerState.model.get()`` — set by the lazy-guard path.
    4. ``state.provider`` is not None — lazy-guard provider delegates to subprocess.
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
        from domain.infrastructure.server_state import get_server_state

        core_model = get_server_state().model.get()
        if core_model is not None:
            return True
        # Lazy-guard: provider is set but model is in subprocess — can still serve
        if provider is not None:
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

        # Use the same model-status logic as inference routers
        from routers.inference import _get_model_status

        ms = _get_model_status()
        if ms["ready"]:
            return await call_next(request)

        corr_id = request.scope.get("correlation_id", "-")
        logger.warning(
            "readiness_gate: %s %s blocked (code=%s) corr=%s",
            request.method,
            path,
            ms["code"],
            corr_id,
            extra={
                "tag": "INFRA",
                "http": {"method": request.method, "path": path, "status": 503, "corr": corr_id},
            },
        )

        return JSONResponse(
            status_code=503,
            content=error_response(
                ms["reason"],
                ms["code"],
            ),
            headers={"Retry-After": "2"},
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
        except TimeoutError:
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
        raw_id = (
            request.headers.get(self.HEADER)
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())[:8]
        )
        # Sanitize: strip control chars, newlines, truncate to 64 chars
        corr_id = (
            "".join(c for c in raw_id if c.isalnum() or c in "-_.")[:64] or str(uuid.uuid4())[:8]
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
        4xx  -> logger.warning (tag REQ)  - DEBUG if an exception handler
                                            already logged it (single-owner)
        >1s  -> logger.warning (tag SLOW)  - unless path is in _COLD_START_PATHS
        else -> logger.debug   (tag REQ)

    On unhandled exceptions only a concise debug line is logged here.
    The single ERROR line + full file traceback come from the FastAPI
    exception handlers (which own error responses).

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
        except Exception as exc:
            elapsed = time.monotonic() - start
            elapsed_str = f"{elapsed:.3f}s"
            try:
                from domain.logging._internal.errors import format_concise

                detail = format_concise(exc, source=f"{method} {path}")
            except Exception:
                detail = f"{type(exc).__name__}: {exc}"
            # Debug only: the exception handler logs the single ERROR
            # line + the file log keeps the full traceback. Logging
            # here at error level would triple-log every failure.
            logger.debug(
                "unhandled exception on %s %s (%s) corr=%s: %s",
                method,
                path,
                elapsed_str,
                corr_id,
                detail,
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
            # Extension-origin failures (wallet injections, devtools scripts)
            # are not our bugs: add a DEBUG note so the WARN stays traceable
            # without demanding attention.  This is the production home of the
            # note ClientErrorFilterMiddleware used to emit as a separate
            # outermost layer (it is no longer registered).
            origin = request.headers.get("origin", "")
            if "chrome-extension" in origin or "moz-extension" in origin:
                logger.debug(
                    "Extension error suppressed: %s %s %d",
                    method,
                    path,
                    sc,
                    extra={"op": "http.request"},
                )
            # Single-owner logging: when an exception handler already logged
            # this failure (detail + code), keep only its WARN and demote our
            # line to DEBUG so timing stays traceable without a duplicate WARN.
            # The flag lives in request.scope — request.state does not
            # propagate through call_next across BaseHTTPMiddleware layers.
            if request.scope.get("slo.error_logged"):
                logger.debug(
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
                return response
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
            is_cold_exact = path in _COLD_START_PATHS
            is_cold_prefix = any(path.startswith(p) for p in _COLD_START_PREFIXES)
            in_warmup = (time.monotonic() - _SERVER_START) < _WARMUP_SECONDS
            if (is_cold_exact and elapsed < 60.0) or (is_cold_prefix and in_warmup):
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
            logger.debug(
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
        from domain.infrastructure.metrics import get_metrics_collector

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


class SerializationGuardMiddleware(BaseHTTPMiddleware):
    """Catches response serialization/validation errors and returns structured JSON.

    FastAPI wraps Pydantic errors in ``ResponseValidationError`` for validation
    and may let ``PydanticSerializationError`` escape during JSON encoding.
    Both result in a bare 500 with no body.  This middleware intercepts both
    and returns a structured JSON error with diagnostic info.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            from fastapi.exceptions import ResponseValidationError

            is_validation = isinstance(exc, ResponseValidationError)
            is_serialization = type(exc).__name__ in (
                "PydanticSerializationError",
                "SerializationError",
            ) or (isinstance(exc, (ValueError, TypeError)) and "serialize" in str(exc).lower())

            if not (is_validation or is_serialization):
                raise

            import re
            import traceback

            cid = request.scope.get("correlation_id", "-")
            error_msg = str(exc)

            # Extract the unknown type: <class 'method'> or <class 'SomeType'>
            unknown_type = None
            match = re.search(r"<class\s+['\"]?(\w+)['\"]?>", error_msg)
            if match:
                unknown_type = match.group(1)

            # Identify response_model from the route
            route = request.scope.get("route")
            response_model_name = None
            if route is not None:
                rm = getattr(route, "response_model", None)
                if rm is not None:
                    response_model_name = getattr(rm, "__name__", str(rm))

            # Extract failing fields from Pydantic error list
            failing_fields = []
            if is_validation:
                for err in exc.errors():
                    loc = err.get("loc", ())
                    if len(loc) >= 2:
                        failing_fields.append(str(loc[-1]))

            if unknown_type == "method":
                detail = "A method was returned instead of its value. Check route handlers for missing ()"
            elif unknown_type:
                detail = f"Unexpected type {unknown_type} in response"
            elif failing_fields:
                detail = f"Response field(s) {', '.join(failing_fields)} failed validation"
            else:
                detail = "Response serialization failed"

            logger.error(
                "Response serialization error on %s %s [%s] response_model=%s unknown_type=%s fields=%s",
                request.method,
                request.url.path,
                cid,
                response_model_name or "none",
                unknown_type or "none",
                failing_fields or "none",
                extra={
                    "tag": "REQ",
                    "context": {
                        "corr": cid,
                        "status": 500,
                        "error_type": type(exc).__name__,
                        "response_model": response_model_name,
                        "unknown_type": unknown_type,
                        "failing_fields": failing_fields,
                    },
                },
            )
            logger.debug(
                "Response serialization traceback:\n%s",
                traceback.format_exc(),
                extra={"tag": "REQ", "context": {"corr": cid}},
            )

            from schemas.common import error_response as _err

            return JSONResponse(
                status_code=500,
                content=_err(
                    "The server encountered an error processing your request. Please try again.",
                    "E_SERIALIZATION",
                    details={"detail": detail, "response_model": response_model_name}
                    if logger.isEnabledFor(logging.DEBUG)
                    else None,
                ),
            )


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

    NOTE: Merged into UnifiedRequestMiddleware in production. Standalone
    class kept for backward compatibility.
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
        (UnifiedRequestMiddleware, {}),
        (SerializationGuardMiddleware, {}),
        (ReadinessGateMiddleware, {}),
        (CorrelationIdMiddleware, {}),
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
