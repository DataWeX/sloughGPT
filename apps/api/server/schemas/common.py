"""
Standard API response models.

Every endpoint returns responses wrapped in StandardResponse for consistency.
Frontend unwraps ``data`` to get the actual payload.

Error responses use the unified shape from ``error_response()``::

    {"error": "...", "code": "...", "details": {...}, "correlation_id": "..."}

``correlation_id`` is resolved automatically from request context when not
passed explicitly — callers never need to thread it manually.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

_audit_logger = logging.getLogger("audit")

T = TypeVar("T")

# Per-request correlation ID, set by CorrelationIdMiddleware.
# error_response() reads this automatically when correlation_id is not passed.
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_correlation_id", default=None,
)


def set_correlation_id(cid: str | None) -> None:
    """Store the current request's correlation ID in context."""
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    """Return the current request's correlation ID, or ``None``."""
    return _correlation_id.get()


class StandardResponse(BaseModel, Generic[T]):
    """Unified response envelope for all API endpoints.

    Attributes:
        status: "success" or "error".
        data: The actual payload (list, dict, scalar, etc.).
        message: Human-readable message (optional).
        meta: Extra metadata like pagination, counts, etc. (optional).
    """

    status: str = "success"
    data: T = Field(default_factory=dict)
    message: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


def success_response(
    data: Any = None,
    message: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict:
    """Build a success StandardResponse as a plain dict (FastAPI serializes it).

    Args:
        data: The actual payload.
        message: Optional human-readable message.
        meta: Optional metadata.

    Returns:
        dict matching StandardResponse shape.
    """
    resp: dict[str, Any] = {"status": "success", "data": data}
    if message:
        resp["message"] = message
    if meta:
        resp["meta"] = meta
    return resp


def error_response(
    message: str,
    code: str = "E_DOMAIN",
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> dict:
    """Build a structured error response body.

    This is the single source of truth for error response shape.
    Exception handlers in ``infrastructure.exception_handlers`` import
    and use this function directly.

    ``correlation_id`` is resolved automatically from request context
    (set by ``CorrelationIdMiddleware``) when not passed explicitly.

    Shape::

        {"error": "...", "code": "...", "details": {...}, "correlation_id": "..."}

    Args:
        message: Human-readable error description.
        code: Machine-readable error code (e.g. ``E_NOT_FOUND``).
        details: Optional extra context (validation errors, etc.).
        correlation_id: Request correlation ID.  Falls back to the
            value set by ``CorrelationIdMiddleware`` when omitted.

    Returns:
        dict with unified error response shape.
    """
    cid = correlation_id or get_correlation_id()
    body: dict[str, Any] = {"error": message, "code": code}
    if details:
        body["details"] = details
    if cid:
        body["correlation_id"] = cid
    return body


def wrap_controller_result(
    result: dict,
    error_code: str = "E_DOMAIN",
) -> dict:
    """Wrap a controller result dict into the correct response shape.

    Controllers return ``{"status": "error", "error": "..."}`` on failure
    and ``{"status": "loaded", ...}`` on success.  This helper detects the
    error case and returns an ``error_response()`` instead of wrapping it
    in ``success_response()``.

    Args:
        result: Dict returned by a controller method.
        error_code: Error code to use when the result is an error.

    Returns:
        ``error_response()`` if result has ``status == "error"``,
        otherwise ``success_response(data=result)``.
    """
    if isinstance(result, dict) and result.get("status") == "error":
        msg = result.get("error") or result.get("message") or "Operation failed"
        return error_response(msg, error_code, details=result)
    return success_response(data=result)


def raise_error(
    message: str,
    code: str = "E_DOMAIN",
    *,
    status_code: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Raise an AppError that the global exception handler converts to JSON.

    Maps router error codes to the AppError subclass hierarchy so the
    existing exception_handlers.py catches and formats the response
    with the correct HTTP status.

    Args:
        message: Human-readable error description.
        code: Machine-readable error code (e.g. ``E_NOT_FOUND``).
        status_code: Override HTTP status (optional — derived from code if omitted).
        details: Optional extra context.

    Raises:
        AppError (or subclass) — never returns.
    """
    # Lazy import to avoid circular dependency at module load time
    from domains.infrastructure.errors import (
        AppError,
        NotFoundError,
        ValidationError as AppValidationError,
        AuthError,
        ResourceExhaustedError,
        ConfigError,
    )

    # Map router error codes → AppError subclasses + HTTP status
    _code_map: dict[str, tuple[type[AppError], int]] = {
        "E_NOT_FOUND":        (NotFoundError, 404),
        "E_VAL_REQUEST":      (AppValidationError, 422),
        "E_VAL_FIELD":        (AppValidationError, 422),
        "E_BAD_REQUEST":      (AppValidationError, 400),
        "E_AUTH_MISSING":     (AuthError, 401),
        "E_AUTH_FORBIDDEN":   (AuthError, 403),
        "E_INFRA_BUSY":       (ResourceExhaustedError, 409),
        "E_INFRA_RATE_LIMIT": (ResourceExhaustedError, 429),
        "E_INFRA_TIMEOUT":    (ResourceExhaustedError, 408),
        "E_INFRA_STARTUP":    (ConfigError, 503),
        "E_INFRA_REGISTRY":   (ConfigError, 503),
        "E_DOMAIN":           (AppError, status_code or 400),
    }

    exc_cls, default_status = _code_map.get(code, (AppError, status_code or 400))
    http_status = status_code or default_status

    raise exc_cls(
        message=message,
        code=code,
        user_message=message,
        http_status=http_status,
        details=details or {},
    )


def safe_audit_log(
    action: str,
    resource: str = "",
    detail: str = "",
    user: str = "anonymous",
    extra: Optional[dict] = None,
    **kwargs: Any,
) -> None:
    """Log an audit event without crashing on failure.

    Falls back to standard ``logging`` if the audit logger is unavailable,
    so audit events are never silently lost.

    Args:
        action: The action being logged (e.g. 'knowledge.add').
        resource: The resource being acted upon.
        detail: Human-readable detail string.
        user: User identifier (default: "anonymous").
        extra: Optional extra dict to forward to the audit logger.
        **kwargs: Additional fields merged into ``extra``.
    """
    try:
        from infrastructure.auth import get_audit_logger
        merged = {**(extra or {}), **kwargs} if kwargs else extra
        get_audit_logger().log(action, user=user, resource=resource, detail=detail, extra=merged)
    except Exception:
        _audit_logger.info(
            "audit:%s resource=%s detail=%s %s",
            action, resource, detail,
            " ".join(f"{k}={v}" for k, v in (extra or kwargs).items()) if (extra or kwargs) else "",
        )


def classify_and_raise(e: Exception, source: str = "router") -> None:
    """Classify an exception, emit an error event, and raise HTTPException.

    Replaces the repeated ``classify_exception + emit_error_event + raise_error``
    blocks across router files.

    Args:
        e: The caught exception.
        source: Identifier for the error source.

    Raises:
        HTTPException: Always, with classified error details.
    """
    from fastapi import HTTPException as _HTTPException
    try:
        from domains.infrastructure.errors import classify_exception, emit_error_event
        err = classify_exception(e)
        emit_error_event(err, source=source)
        raise _HTTPException(status_code=err.http_status, detail=err.user_message)
    except _HTTPException:
        raise
    except Exception:
        _audit_logger.warning(
            "classify_and_raise fallback: source=%s error=%s",
            source, e, exc_info=True,
        )
        raise _HTTPException(status_code=500, detail=str(e))
