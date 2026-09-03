"""
Correlation ID context variable — shared across core and API layers.

Core uses this to include request correlation IDs in log records.
API layer sets it per-request via middleware.
"""

from __future__ import annotations

import contextvars

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_correlation_id", default=None,
)


def set_correlation_id(cid: str | None) -> None:
    """Set the current request's correlation ID."""
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    """Return the current request's correlation ID, or ``None``."""
    return _correlation_id.get()
