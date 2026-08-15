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

    Matches the shape produced by exception handlers in
    ``infrastructure.exception_handlers._error_response()``:

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
