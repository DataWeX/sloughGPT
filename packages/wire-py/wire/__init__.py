"""wire — our own minimal HTTP/1.1 wire layer (card ecea49c8).

Sync, stdlib-only. Phase 1 delivers framing + a keep-alive TCP accept
loop; the ASGI host adapter (phase 2) builds on top of this seam; phase 3
adds the lifespan handshake (:class:`Lifespan`) around the accept loop and
RFC6455 WebSocket upgrades (``wire.websocket`` codec + ``wire.asgi_host``
session) — the flip gate for the transport default.
"""

from __future__ import annotations

from .asgi_host import Lifespan, LifespanError
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
    "Lifespan",
    "LifespanError",
    "MalformedRequest",
    "Request",
    "Response",
    "WireError",
    "WireServer",
    "build_response",
    "read_request",
    "should_keep_alive",
]
