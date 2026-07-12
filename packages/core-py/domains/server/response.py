"""Minimal HTTP response objects."""

import json
from typing import Any, AsyncIterator, Dict, Optional, Union


class Response:
    """Base HTTP response."""

    def __init__(
        self,
        body: Union[str, bytes] = "",
        status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        content_type: str = "text/plain",
    ):
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status
        self.headers = headers or {}
        self.headers["content-type"] = content_type
        self.headers["content-length"] = str(len(self.body))

    def to_bytes(self) -> bytes:
        """Serialize to raw HTTP response bytes."""
        lines = [f"HTTP/1.1 {self.status} {self._status_text()}"]
        for k, v in self.headers.items():
            lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("")
        header_bytes = "\r\n".join(lines).encode()
        return header_bytes + self.body

    def _status_text(self) -> str:
        return {
            200: "OK",
            201: "Created",
            204: "No Content",
            301: "Moved Permanently",
            304: "Not Modified",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
        }.get(self.status, "Unknown")


class JSONResponse(Response):
    """JSON HTTP response."""

    def __init__(
        self,
        content: Any,
        status: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ):
        body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        super().__init__(body, status=status, headers=headers, content_type="application/json")


class StreamingResponse:
    """SSE streaming response — sent chunk-by-chunk."""

    def __init__(
        self,
        iterator: AsyncIterator[str],
        status: int = 200,
        content_type: str = "text/event-stream",
    ):
        self.iterator = iterator
        self.status = status
        self.content_type = content_type

    async def write(self, writer) -> None:
        """Write streaming response to transport writer."""
        # Send headers
        header = (
            f"HTTP/1.1 {self.status} OK\r\n"
            f"content-type: {self.content_type}\r\n"
            f"cache-control: no-cache\r\n"
            f"connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode())
        # Drain is handled by the protocol

        # Send chunks
        async for chunk in self.iterator:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            writer.write(chunk)

        # Flush final
        writer.write(b"\r\n")
