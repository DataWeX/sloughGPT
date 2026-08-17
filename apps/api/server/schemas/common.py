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
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

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


def safe_audit_log(
    action: str,
    resource: str = "",
    detail: str = "",
    **kwargs: Any,
) -> None:
    """Log an audit event without crashing on failure.

    Args:
        action: The action being logged (e.g. 'knowledge.add').
        resource: The resource being acted upon.
        detail: Human-readable detail string.
        **kwargs: Extra fields forwarded to the audit logger.
    """
    try:
        from infrastructure.auth import get_audit_logger
        get_audit_logger().log(action, resource=resource, detail=detail, **kwargs)
    except Exception:
        pass
