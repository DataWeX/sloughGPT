"""
Exception handlers extracted from main.py.

Provides FastAPI exception handlers for:
- AppError (unified error taxonomy from domain.infrastructure._internal.errors)
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
    recoverable = getattr(exc, "recoverable", False)

    log_fn = logger.error if http_status >= 500 else logger.warning
    log_fn(
        "Domain error [%s] on %s %s: %s",
        code,
        request.method,
        request.url.path,
        str(exc)[:200],
        extra={
            "tag": "REQ",
            "context": {
                "corr": cid,
                "code": code,
                "status": http_status,
                "recoverable": recoverable,
            },
        },
    )
    return JSONResponse(
        status_code=http_status,
        content=error_response(user_message, code),
    )


async def _validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Catch Pydantic validation errors."""
    errors = exc.errors()
    cid = _corr_id(request)

    # Extract field names for logging
    field_names = []
    for error in errors:
        loc = error.get("loc", [])
        if len(loc) > 1:
            field_names.append(str(loc[-1]))

    logger.warning(
        "Validation failed on %s: %d errors in fields [%s]",
        request.url.path,
        len(errors),
        ", ".join(field_names[:5]),
        extra={
            "tag": "REQ",
            "context": {
                "corr": cid,
                "fields": len(errors),
                "field_names": field_names[:5],
                "status": 422,
            },
        },
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error_response(
            "Validation failed",
            "E_VAL_FIELD",
            details={"errors": errors},
        ),
    )


async def _pydantic_serialization_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch Pydantic serialization errors with diagnostic logging.

    Logs the endpoint, HTTP method, response model (if available), and the
    unknown type that caused the failure so the offending handler can be
    located quickly.
    """
    import re
    import traceback

    cid = _corr_id(request)
    error_msg = str(exc)
    detail = ""

    # Extract the type name from the error message
    unknown_type = None
    if "Unable to serialize unknown type" in error_msg:
        match = re.search(r"<class\s+'?(\w+)'?>", error_msg)
        if match:
            unknown_type = match.group(1)
            if unknown_type == "method":
                detail = "A method was returned instead of its value. Check route handlers for missing ()"
            else:
                detail = f"Unexpected type {unknown_type} in response"

    if not detail:
        detail = "Response serialization failed"

    # Try to identify which route / response_model was involved
    route = request.scope.get("route")
    response_model_name = None
    if route is not None:
        response_model_name = getattr(route, "response_model", None)
        if response_model_name is not None:
            response_model_name = getattr(response_model_name, "__name__", str(response_model_name))

    logger.error(
        "Serialization error on %s %s [%s] response_model=%s unknown_type=%s: %s",
        request.method,
        request.url.path,
        cid,
        response_model_name or "none",
        unknown_type or "unknown",
        error_msg[:300],
        extra={
            "tag": "REQ",
            "context": {
                "corr": cid,
                "status": 500,
                "error_type": "PydanticSerializationError",
                "response_model": response_model_name,
                "unknown_type": unknown_type,
                "detail": detail,
            },
        },
    )
    # Full traceback at DEBUG for root-cause analysis
    logger.debug(
        "Serialization error traceback:\n%s",
        traceback.format_exc(),
        extra={"tag": "REQ", "context": {"corr": cid}},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(
            "The server encountered an error processing your request. Please try again.",
            "E_SERIALIZATION",
            details={"detail": detail, "response_model": response_model_name}
            if logger.isEnabledFor(logging.DEBUG)
            else None,
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

    # Map status code to error code and user-friendly message
    code_map = {
        400: ("E_BAD_REQUEST", "The request is invalid. Please check your input."),
        401: ("E_AUTH_MISSING", "Authentication required. Please log in."),
        403: ("E_AUTH_FORBIDDEN", "You do not have permission to access this resource."),
        404: ("E_NOT_FOUND", "The requested resource was not found."),
        405: ("E_BAD_REQUEST", "This method is not allowed for this endpoint."),
        408: ("E_INFRA_TIMEOUT", "The request timed out. Please try again."),
        409: ("E_CONFLICT", "The resource is in a conflicting state. Please try again."),
        413: ("E_BAD_REQUEST", "The request is too large."),
        415: ("E_BAD_REQUEST", "The request format is not supported."),
        422: ("E_VAL_FIELD", "The request data is invalid."),
        429: ("E_INFRA_RATE_LIMIT", "Too many requests. Please slow down."),
        500: ("E_INTERNAL", "An internal error occurred. Please try again."),
        502: ("E_INFRA_REGISTRY", "A service is unavailable. Please try again later."),
        503: (
            "E_INFRA_REGISTRY",
            "The service is temporarily unavailable. Please try again later.",
        ),
        504: ("E_INFRA_TIMEOUT", "The gateway timed out. Please try again."),
    }

    error_code, user_msg = code_map.get(h.status_code, ("E_DOMAIN", str(h.detail)))

    # If the HTTPException has a detail that's more specific, use it
    if h.detail and isinstance(h.detail, str) and len(h.detail) > 0:
        # But only if it's user-friendly (not a raw error message)
        if not any(
            word in h.detail.lower() for word in ["exception", "error", "traceback", "file"]
        ):
            user_msg = h.detail

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
        content=error_response(user_msg, error_code),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all — classify raw exceptions into AppError, emit event, return structured response.

    This is the SINGLE point where EventBus events are emitted for unhandled errors.
    classify_and_raise() in routers does NOT emit — it only classifies and raises.
    """
    cid = _corr_id(request)

    # Classify into the error taxonomy
    try:
        from domain.infrastructure._internal.errors import classify_exception, emit_error_event

        classified = classify_exception(exc)
        classified.source = f"{request.method} {request.url.path}"
        emit_error_event(classified, source=classified.source)
    except ImportError:
        classified = None

    # Determine user-friendly message based on exception type
    user_msg = "An unexpected error occurred. Please try again."
    if classified is not None:
        user_msg = classified.user_message
    else:
        # Fallback: provide more specific messages for common exception types
        exc_type = type(exc).__name__
        if "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            user_msg = "A required service is unavailable. Please restart the server."
        elif "AttributeError" in exc_type:
            user_msg = "An internal error occurred. Please try again."
        elif "TypeError" in exc_type:
            user_msg = "An internal error occurred. Please try again."
        elif "KeyError" in exc_type:
            user_msg = "The requested data was not found."
        elif "ConnectionRefused" in str(exc) or "ConnectionError" in str(exc):
            user_msg = "A service is unavailable. Please try again later."
        elif "Timeout" in str(exc):
            user_msg = "The request timed out. Please try again."

    source = f"{request.method} {request.url.path}"
    try:
        from domain.logging._internal.errors import log_error

        log_error(
            logger,
            exc,
            source=source,
            code=getattr(classified, "code", "E_UNHANDLED")
            if classified is not None
            else "E_UNHANDLED",
            corr=cid,
            status=getattr(classified, "http_status", 500) if classified is not None else 500,
        )
    except Exception:
        logger.error(
            "Unhandled error on %s [%s]: %s: %s",
            source,
            cid,
            type(exc).__name__,
            str(exc)[:300],
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
        content=error_response(user_msg, "E_UNHANDLED"),
    )


def register_app_error_handler(app: FastAPI):
    """Register only the AppError handler — minimal handler for test clients.

    Unlike ``register_all_handlers``, this does NOT override FastAPI's default
    handlers for ``RequestValidationError`` or ``HTTPException``, so existing
    test assertions against ``resp.json()["detail"]`` remain valid.
    """
    try:
        from domain.infrastructure._internal.errors import AppError

        async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            # SINGLE EventBus emission point for all AppErrors
            try:
                from domain.infrastructure._internal.errors import emit_error_event

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
        logger.debug("AppError not available — structured error handler disabled")


def register_all_handlers(app: FastAPI):
    """Register all exception handlers on a FastAPI instance.

    Handler priority (FastAPI matches first registered):
      1. AppError — structured taxonomy (SINGLE EventBus emission point)
      2. SloughGPTDomainError — legacy, now extends AppError (kept for safety)
      3. ValidationError — Pydantic
      4. RequestValidationError — FastAPI
      5. PydanticSerializationError — response serialization failures
      6. HTTPException — FastAPI
      7. Exception — catch-all, classified via classify_exception
    """
    # AppError — the PRIMARY handler. Emits EventBus event ONCE.
    try:
        from domain.infrastructure._internal.errors import AppError

        async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
            cid = _corr_id(request)
            # SINGLE EventBus emission — classify_and_raise() does NOT emit
            try:
                from domain.infrastructure._internal.errors import emit_error_event

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
        logger.debug("AppError not available — structured error handling disabled")

    # Domain errors (legacy hierarchy — now extends AppError, kept for safety)
    try:
        from domain.errors import SloughGPTDomainError

        app.add_exception_handler(SloughGPTDomainError, _domain_error_handler)
    except ImportError:
        logger.debug("SloughGPTDomainError not available — legacy handler skipped")

    app.add_exception_handler(ValidationError, _validation_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_error_handler)

    # Pydantic serialization errors (e.g., returning a method instead of its value)
    try:
        from pydantic_core import PydanticSerializationError

        app.add_exception_handler(PydanticSerializationError, _pydantic_serialization_error_handler)
    except ImportError:
        logger.debug("PydanticSerializationError not available — serialization handler skipped")

    app.add_exception_handler(HTTPException, _http_exception_handler)

    app.add_exception_handler(Exception, _unhandled_error_handler)
