"""Backward-compatibility shim — imports from the new ``domain.api`` package."""

from domain.api import (
    SSEEnvelope,
    StreamPhase,
    StreamStatus,
    sse_complete,
    sse_error,
    sse_event,
    sse_token,
)

__all__ = [
    "StreamPhase",
    "StreamStatus",
    "SSEEnvelope",
    "sse_event",
    "sse_error",
    "sse_complete",
    "sse_token",
]
