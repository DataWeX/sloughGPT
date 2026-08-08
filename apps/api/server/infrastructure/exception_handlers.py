"""
Exception handlers extracted from main.py.

Provides FastAPI exception handlers for:
- AppError (structured error taxonomy from domains.infrastructure.errors)
- SloughGPTDomainError (domain-layer errors)
- ValidationError (Pydantic)
- HTTPException (FastAPI)
- BaseException / unhandled errors (catch-all, classified via classify_exception)

All handlers emit error events on the EventBus and use structured error codes.
Register via ``register_all_handlers(app)``.
"""

from __future__ import annotations

import json
import logging
import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger("slo.exception_handlers")


def _error_response(
    status_code: int,
    message: str,
    error_code: str,
    details: dict | None = None,
    correlation_id: str | None = None,
) -> dict:
    """Build a structured error response body."""
    body = {
        "error": message,
        "code": error_code,
    }
    if details:
        body["details"] = details
    if correlation_id:
        body["correlation_id"] = correlation_id
    return body


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch domain-layer exceptions (SloughGPTDomainError subclasses)."""
    msg = str(exc) or "Domain error"
    corr_id = getattr(request.state, "correlation_id", "-")
    logger.warning(
        "%s on %s %s",
        msg, request.method, request.url.path,
        extra={"context": {"corr": corr_id, "status": status.HTTP_400_BAD_REQUEST}},
    )
    # Attach tag via extra — BridgeHandler picks it up
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=_error_response(
            status.HTTP_400_BAD_REQUEST, msg, "E_DOMAIN",
            correlation_id=corr_id,
        ),
    )


async def _validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Catch Pydantic validation errors."""
    errors = exc.errors()
    corr_id = getattr(request.state, "correlation_id", "-")
    logger.warning(
        "Validation failed on %s", request.url.path,
        extra={"context": {"corr": corr_id, "fields": len(errors), "status": 422}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Validation failed", "E_VAL_FIELD",
            details={"errors": errors},
            correlation_id=corr_id,
        ),
    )


async def _request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Catch FastAPI request validation errors."""
    errors = exc.errors()
    # Pydantic v2 may include bytes in error detail (raw request body) — convert for JSON safety
    _safe = json.loads(json.dumps(errors, default=str))
    corr_id = getattr(request.state, "correlation_id", "-")
    logger.warning(
        "Request validation failed on %s %s", request.method, request.url.path,
        extra={"context": {"corr": corr_id, "errors": len(_safe), "status": 422}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Request validation failed", "E_VAL_REQUEST",
            details={"errors": _safe},
            correlation_id=corr_id,
        ),
    )


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Re-raise HTTPExceptions as JSON."""
    from fastapi import HTTPException as _HTTPException
    h = exc  # type: _HTTPException
    corr_id = getattr(request.state, "correlation_id", "-")

    # Map status code to error code
    code_map = {
        401: "E_AUTH_MISSING",
        403: "E_AUTH_FORBIDDEN",
        404: "E_NOT_FOUND",
        408: "E_INFRA_TIMEOUT",
        429: "E_INFRA_TIMEOUT",
        503: "E_INFRA_REGISTRY",
    }
    error_code = code_map.get(h.status_code, "E_DOMAIN")

    log_fn = logger.warning if h.status_code >= 500 else logger.info
    log_fn(
        "HTTP %d on %s %s", h.status_code, request.method, request.url.path,
        extra={"context": {"corr": corr_id, "detail": str(h.detail)[:120], "status": h.status_code}},
    )
    return JSONResponse(
        status_code=h.status_code,
        content=_error_response(
            h.status_code, str(h.detail), error_code,
            correlation_id=corr_id,
        ),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all — classify raw exceptions into AppError, emit event, return structured response."""
    corr_id = getattr(request.state, "correlation_id", "-")

    # Classify into the error taxonomy
    try:
        from domains.infrastructure.errors import classify_exception, emit_error_event
        classified = classify_exception(exc)
        emit_error_event(classified, source=f"{request.method} {request.url.path}")
    except ImportError:
        classified = None

    logger.exception(
        "Unhandled error on %s %s [%s]", request.method, request.url.path, corr_id,
        extra={"context": {"corr": corr_id, "status": 500}},
    )

    if classified is not None:
        return JSONResponse(
            status_code=classified.http_status,
            content=_error_response(
                classified.http_status,
                classified.user_message,
                classified.code,
                details=classified.details if logger.isEnabledFor(logging.DEBUG) else None,
                correlation_id=corr_id,
            ),
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error", "E_INFRA_STARTUP",
            correlation_id=corr_id,
        ),
    )


def register_all_handlers(app: FastAPI):
    """Register all exception handlers on a FastAPI instance."""
    # AppError — structured taxonomy with codes, user messages, EventBus events
    try:
        from domains.infrastructure.errors import AppError

        async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            corr_id = getattr(request.state, "correlation_id", "-")
            try:
                from domains.infrastructure.errors import emit_error_event
                emit_error_event(exc, source=f"{request.method} {request.url.path}")
            except Exception:
                pass
            log_fn = logger.warning if exc.http_status >= 500 else logger.info
            log_fn(
                "%s [%s] on %s %s", exc.code, exc.message, request.method, request.url.path,
                extra={"context": {"corr": corr_id, "code": exc.code, "status": exc.http_status}},
            )
            return JSONResponse(
                status_code=exc.http_status,
                content=_error_response(
                    exc.http_status,
                    exc.user_message,
                    exc.code,
                    details=exc.details if logger.isEnabledFor(logging.DEBUG) else None,
                    correlation_id=corr_id,
                ),
            )

        app.add_exception_handler(AppError, _app_error_handler)
    except ImportError:
        pass

    # Domain errors (legacy hierarchy)
    try:
        from domains.core.base import SloughGPTDomainError
        app.add_exception_handler(SloughGPTDomainError, _domain_error_handler)
    except ImportError:
        pass

    app.add_exception_handler(ValidationError, _validation_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)

    from fastapi import HTTPException
    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.add_exception_handler(Exception, _unhandled_error_handler)
