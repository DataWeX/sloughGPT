"""
Standard API response models.

Every endpoint returns responses wrapped in StandardResponse for consistency.
Frontend unwraps `data` to get the actual payload.
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


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

    Shape::

        {"error": "...", "code": "...", "details": {...}, "correlation_id": "..."}

    Args:
        message: Human-readable error description.
        code: Machine-readable error code (e.g. ``E_NOT_FOUND``).
        details: Optional extra context (validation errors, etc.).
        correlation_id: Optional request correlation ID.

    Returns:
        dict with unified error response shape.
    """
    body: dict[str, Any] = {"error": message, "code": code}
    if details:
        body["details"] = details
    if correlation_id:
        body["correlation_id"] = correlation_id
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
