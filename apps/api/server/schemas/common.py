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
    data: Any = None,
    meta: dict[str, Any] | None = None,
    status_code: int = 500,
) -> dict:
    """Build an error StandardResponse as a plain dict.

    Args:
        message: Error description.
        data: Optional error details.
        meta: Optional metadata.

    Returns:
        dict matching StandardResponse shape.
    """
    resp: dict[str, Any] = {"status": "error", "message": message, "data": data}
    if meta:
        resp["meta"] = meta
    return resp
