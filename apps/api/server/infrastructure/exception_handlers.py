"""
Exception handlers extracted from main.py.

Provides FastAPI exception handlers for:
- AppError (unified error taxonomy from domains.infrastructure.errors)
- SloughGPTDomainError (legacy domain errors, now extends AppError)
- ValidationError (Pydantic)
- RequestValidationError (FastAPI)
- HTTPException (FastAPI)
- BaseException / unhandled errors (catch-all, classified via classify_exception)

All handlers emit error events on the EventBus ONCE (in the AppError handler)
and use structured error codes.

Register via ``register_all_handlers(app)``.

All error responses use the unified shape from ``schemas.common.error_response()``:
    {"error": "...", "code": "...", "details": {...}, "correlation_id": "..."}

``correlation_id`` is resolved automatically from request context (set by
``CorrelationIdMiddleware``) — handlers do not need to thread it manually.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from schemas.common import error_response, get_correlation_id

logger = logging.getLogger("slo.exception_handlers")


def _corr_id(request: Request) -> str:
    """Return correlation ID: prefer contextvar, fall back to scope."""
    return get_correlation_id() or request.scope.get("correlation_id", "-")


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch domain-layer exceptions (SloughGPTDomainError subclasses).

    Since SloughGPTDomainError now extends AppError, the AppError handler
    is the primary handler. This is kept as a safety net — it reads the
    actual error attributes instead of hardcoding them.
    """
    cid = _corr_id(request)
    code = getattr(exc, "code", "E_DOMAIN")
    http_status = getattr(exc, "http_status", status.HTTP_400_BAD_REQUEST)
    user_message = getattr(exc, "user_message", str(exc) or "Domain error")

    logger.warning(
        "%s on %s %s",
        str(exc),
        request.method,
        request.url.path,
        extra={"tag": "REQ", "context": {"corr": cid, "code": code, "status": http_status}},
    )
    return JSONResponse(
        status_code=http_status,
        content=error_response(user_message, code),
    )


async def _validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Catch Pydantic validation errors."""
    errors = exc.errors()
    cid = _corr_id(request)
    logger.warning(
        "Validation failed on %s",
        request.url.path,
        extra={"tag": "REQ", "context": {"corr": cid, "fields": len(errors), "status": 422}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error_response(
            "Validation failed",
            "E_VAL_FIELD",
            details={"errors": errors},
        ),
    )


async def _request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Catch FastAPI request validation errors."""
    errors = exc.errors()
    # Pydantic v2 may include bytes in error detail (raw request body) — convert for JSON safety
    _safe = json.loads(json.dumps(errors, default=str))
    cid = _corr_id(request)
    logger.warning(
        "Request validation failed on %s %s",
        request.method,
        request.url.path,
        extra={"tag": "REQ", "context": {"corr": cid, "errors": len(_safe), "status": 422}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error_response(
            "Request validation failed",
            "E_VAL_REQUEST",
            details={"errors": _safe},
        ),
    )


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Re-raise HTTPExceptions as JSON."""
    h = exc  # type: HTTPException
    cid = _corr_id(request)

    # Map status code to error code
    code_map = {
        401: "E_AUTH_MISSING",
        403: "E_AUTH_FORBIDDEN",
        404: "E_NOT_FOUND",
        408: "E_INFRA_TIMEOUT",
        429: "E_INFRA_RATE_LIMIT",
        503: "E_INFRA_REGISTRY",
    }
    error_code = code_map.get(h.status_code, "E_DOMAIN")

    log_fn = logger.error if h.status_code >= 500 else logger.warning
    log_fn(
        "HTTP %d on %s %s",
        h.status_code,
        request.method,
        request.url.path,
        extra={
            "tag": "REQ",
            "context": {"corr": cid, "detail": str(h.detail)[:120], "status": h.status_code},
        },
    )
    return JSONResponse(
        status_code=h.status_code,
        content=error_response(str(h.detail), error_code),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all — classify raw exceptions into AppError, emit event, return structured response.

    This is the SINGLE point where EventBus events are emitted for unhandled errors.
    classify_and_raise() in routers does NOT emit — it only classifies and raises.
    """
    cid = _corr_id(request)

    # Classify into the error taxonomy
    try:
        from domains.infrastructure.errors import classify_exception, emit_error_event

        classified = classify_exception(exc)
        classified.source = f"{request.method} {request.url.path}"
        emit_error_event(classified, source=classified.source)
    except ImportError:
        classified = None

    logger.exception(
        "Unhandled error on %s %s [%s]",
        request.method,
        request.url.path,
        cid,
        extra={"tag": "REQ", "context": {"corr": cid, "status": 500}},
    )

    if classified is not None:
        return JSONResponse(
            status_code=classified.http_status,
            content=error_response(
                classified.user_message,
                classified.code,
                details=classified.details if logger.isEnabledFor(logging.DEBUG) else None,
            ),
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response("Internal server error", "E_UNHANDLED"),
    )


def register_app_error_handler(app: FastAPI):
    """Register only the AppError handler — minimal handler for test clients.

    Unlike ``register_all_handlers``, this does NOT override FastAPI's default
    handlers for ``RequestValidationError`` or ``HTTPException``, so existing
    test assertions against ``resp.json()["detail"]`` remain valid.
    """
    try:
        from domains.infrastructure.errors import AppError

        async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            # SINGLE EventBus emission point for all AppErrors
            try:
                from domains.infrastructure.errors import emit_error_event

                emit_error_event(exc, source=f"{request.method} {request.url.path}")
            except Exception as e:
                logger.debug("Error event emission failed: %s", e)
            return JSONResponse(
                status_code=exc.http_status,
                content=error_response(
                    exc.user_message,
                    exc.code,
                    details=exc.details if logger.isEnabledFor(logging.DEBUG) else None,
                ),
            )

        app.add_exception_handler(AppError, _app_error_handler)
    except ImportError:
        pass


def register_all_handlers(app: FastAPI):
    """Register all exception handlers on a FastAPI instance.

    Handler priority (FastAPI matches first registered):
      1. AppError — structured taxonomy (SINGLE EventBus emission point)
      2. SloughGPTDomainError — legacy, now extends AppError (kept for safety)
      3. ValidationError — Pydantic
      4. RequestValidationError — FastAPI
      5. HTTPException — FastAPI
      6. Exception — catch-all, classified via classify_exception
    """
    # AppError — the PRIMARY handler. Emits EventBus event ONCE.
    try:
        from domains.infrastructure.errors import AppError

        async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            cid = _corr_id(request)
            # SINGLE EventBus emission — classify_and_raise() does NOT emit
            try:
                from domains.infrastructure.errors import emit_error_event

                emit_error_event(exc, source=f"{request.method} {request.url.path}")
            except Exception as e:
                logger.debug("Error event emission failed: %s", e)
            log_fn = logger.error if exc.http_status >= 500 else logger.warning
            log_fn(
                "%s [%s] on %s %s",
                exc.code,
                exc.message,
                request.method,
                request.url.path,
                extra={
                    "tag": "REQ",
                    "context": {"corr": cid, "code": exc.code, "status": exc.http_status},
                },
            )
            return JSONResponse(
                status_code=exc.http_status,
                content=error_response(
                    exc.user_message,
                    exc.code,
                    details=exc.details if logger.isEnabledFor(logging.DEBUG) else None,
                ),
            )

        app.add_exception_handler(AppError, _app_error_handler)
    except ImportError:
        pass

    # Domain errors (legacy hierarchy — now extends AppError, kept for safety)
    try:
        from domains.errors import SloughGPTDomainError

        app.add_exception_handler(SloughGPTDomainError, _domain_error_handler)
    except ImportError:
        pass

    app.add_exception_handler(ValidationError, _validation_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)

    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.add_exception_handler(Exception, _unhandled_error_handler)
