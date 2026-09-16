"""api — SSE envelope and streaming utilities.

Public API:
    StreamPhase, StreamStatus, SSEEnvelope, sse_event, sse_error, sse_complete, sse_token
"""

from domain.api._internal.sse_envelope import (
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
