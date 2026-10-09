"""Selective GZip compression middleware.

Starlette's built-in GZipMiddleware buffers all responses, which kills SSE
streaming. This middleware skips compression for:
- SSE streams (``text/event-stream``)
- Responses already compressed (``Content-Encoding`` set)
- Small responses (< 500 bytes)
- Huge responses (> 8 MB): buffering 2x a large file in RAM is an OOM risk,
  and the gateway compresses the wire anyway
- Binary / pre-compressed content types (images, video, xz, zstd, ...)

Everything else gets gzip compression for reduced bandwidth. Compression runs
at the async seam (``asyncio.to_thread``) so a large body never stalls the
event loop. This middleware is the direct-hit fallback: behind the gateway
(which negotiates ``identity``) it stays idle.
"""

from __future__ import annotations

import asyncio
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
        # Already-compressed containers: gzip wastes CPU for zero gain (the
        # size guard catches growth, but not the CPU burn).
        "application/x-bzip2",
        "application/x-xz",
        "application/zstd",
    }
)

# Minimum response size to compress (bytes)
_MIN_SIZE = 500
# Maximum response size to compress (bytes) — above this, serve identity.
_MAX_SIZE = 8 * 1024 * 1024


def _accepts_gzip(header: str) -> bool:
    """True when Accept-Encoding offers gzip with q > 0.

    A substring check lies: ``gzip;q=0`` is an explicit refusal of gzip,
    not an offer.
    """
    for part in header.split(","):
        tokens = [t.strip() for t in part.split(";")]
        if tokens[0].lower() != "gzip":
            continue
        for param in tokens[1:]:
            if param.lower().startswith("q="):
                try:
                    return float(param[2:]) > 0
                except ValueError:
                    return True
        return True  # present without an explicit q => acceptable
    return False


class SelectiveGZipMiddleware(BaseHTTPMiddleware):
    """Compress non-streaming HTTP responses with gzip."""

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = _MIN_SIZE,
        maximum_size: int = _MAX_SIZE,
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Vary on Accept-Encoding for EVERY response this middleware governs,
        # not just the compressed one — identity and compressed variants must
        # be cached separately.
        vary = response.headers.get("vary", "")
        if "accept-encoding" not in vary.lower():
            response.headers["vary"] = f"{vary}, Accept-Encoding".strip(", ")

        # Skip if client doesn't support gzip (q=0 is a refusal)
        accept_encoding = request.headers.get("accept-encoding", "")
        if not _accepts_gzip(accept_encoding):
            return response

        # Skip streaming responses
        content_type = response.headers.get("content-type", "")
        if any(ct in content_type for ct in _SKIP_CONTENT_TYPES):
            return response

        # Skip if already compressed
        if response.headers.get("content-encoding"):
            return response

        # Buffer the full response body. Collect-then-join: `body += chunk`
        # in this loop is O(n^2) across chunks.
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                chunks.append(chunk.encode("utf-8"))
            else:
                chunks.append(chunk)
        body = b"".join(chunks)

        # Skip small and oversized responses
        if len(body) < self.minimum_size or len(body) > self.maximum_size:
            return self._identity_response(response, body, content_type)

        # Compress at the seam: never run gzip synchronously on the event loop
        compressed = await asyncio.to_thread(gzip.compress, body, 6)

        # Only use compressed version if it's actually smaller
        if len(compressed) >= len(body):
            return self._identity_response(response, body, content_type)

        # Build new response with compression headers
        headers = dict(response.headers)
        headers["content-encoding"] = "gzip"
        headers["content-length"] = str(len(compressed))
        # Fully buffered: a stale chunked framing header would be a lie
        headers.pop("transfer-encoding", None)

        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=content_type,
        )

    @staticmethod
    def _identity_response(response: Response, body: bytes, content_type: str) -> Response:
        """Rebuild an uncompressed response with correct framing headers."""
        headers = dict(response.headers)
        # Let Starlette recompute the length for the buffered body; a stale
        # content-length or transfer-encoding would misframe it.
        headers.pop("content-length", None)
        headers.pop("transfer-encoding", None)
        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=content_type,
        )
