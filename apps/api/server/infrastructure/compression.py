"""Selective GZip compression middleware.

Starlette's built-in GZipMiddleware buffers all responses, which kills SSE
streaming. This middleware skips compression for:
- SSE streams (``text/event-stream``)
- Responses already compressed (``Content-Encoding`` set)
- Small responses (< 500 bytes)
- Binary content types (images, video, etc.)

Everything else gets gzip compression for reduced bandwidth.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Content types that should NOT be compressed
_SKIP_CONTENT_TYPES = frozenset(
    {
        "text/event-stream",
        "image/",
        "video/",
        "audio/",
        "application/octet-stream",
        "application/zip",
        "application/gzip",
        "application/pdf",
    }
)

# Minimum response size to compress (bytes)
_MIN_SIZE = 500


class SelectiveGZipMiddleware(BaseHTTPMiddleware):
    """Compress non-streaming HTTP responses with gzip."""

    def __init__(self, app: ASGIApp, minimum_size: int = _MIN_SIZE):
        super().__init__(app)
        self.minimum_size = minimum_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Skip if client doesn't support gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return response

        # Skip streaming responses
        content_type = response.headers.get("content-type", "")
        if any(ct in content_type for ct in _SKIP_CONTENT_TYPES):
            return response

        # Skip if already compressed
        if response.headers.get("content-encoding"):
            return response

        # Read the full response body
        body = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                body += chunk.encode("utf-8")
            else:
                body += chunk

        # Skip small responses
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        # Compress
        compressed = gzip.compress(body, compresslevel=6)

        # Only use compressed version if it's actually smaller
        if len(compressed) >= len(body):
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        # Build new response with compression headers
        headers = dict(response.headers)
        headers["content-encoding"] = "gzip"
        headers["content-length"] = str(len(compressed))
        headers["vary"] = "Accept-Encoding"

        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=content_type,
        )
