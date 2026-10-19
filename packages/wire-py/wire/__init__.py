"""wire — our own minimal HTTP/1.1 wire layer (card ecea49c8).

Sync, stdlib-only. Phase 1 delivers framing + a keep-alive TCP accept
loop; the ASGI host adapter (phase 2) builds on top of this seam.
"""

from __future__ import annotations

from .framing import (
    MAX_HEADER_LINE,
    MAX_HEADERS,
    MAX_REQUEST_LINE,
    HeaderTooLarge,
    MalformedRequest,
    Request,
    WireError,
    build_response,
    read_request,
    should_keep_alive,
)
from .server import Handler, Response, WireServer

__version__ = "0.1.0"

__all__ = [
    "MAX_HEADERS",
    "MAX_HEADER_LINE",
    "MAX_REQUEST_LINE",
    "HeaderTooLarge",
    "Handler",
    "MalformedRequest",
    "Request",
    "Response",
    "WireError",
    "WireServer",
    "build_response",
    "read_request",
    "should_keep_alive",
]
