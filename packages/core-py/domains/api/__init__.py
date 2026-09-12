"""Backward-compatibility shim — imports from the new ``domain.api`` package."""

from domain.api import (
    StreamPhase,
    StreamStatus,
    SSEEnvelope,
    sse_event,
    sse_error,
    sse_complete,
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
