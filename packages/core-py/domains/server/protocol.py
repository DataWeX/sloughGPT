"""
Minimal HTTP/1.1 server — pure asyncio, zero dependencies.

Parses HTTP requests, dispatches to App, sends responses.
"""

import asyncio
import logging
import re
import signal
import sys
from typing import Optional

from domains.server.app import App
from domains.server.request import Request
from domains.server.response import StreamingResponse

logger = logging.getLogger("man.server")

# ── HTTP Parser ──────────────────────────────────────────────────────────────


def parse_request(data: bytes) -> Optional[Request]:
    """Parse raw HTTP bytes into a Request. Returns None on incomplete data."""
    try:
        header_end = data.find(b"\r\n\r\n")
        if header_end == -1:
            return None

        header_bytes = data[:header_end]
        body = data[header_end + 4:]

        lines = header_bytes.decode().split("\r\n")
        if not lines:
            return None

        # Request line: METHOD PATH HTTP/1.1
        parts = lines[0].split(" ", 2)
        if len(parts) < 2:
            return None
        method, raw_path = parts[0].upper(), parts[1]

        # Split path and query string
        if "?" in raw_path:
            path, qs = raw_path.split("?", 1)
            query = dict(re.findall(r"([^&=]+)=([^&]*)", qs))
        else:
            path, query = raw_path, {}

        # Headers
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        # Handle chunked encoding
        if headers.get("transfer-encoding") == "chunked":
            body = _decode_chunked(body)

        return Request(method=method, path=path, headers=headers, body=body, query=query)
    except Exception:
        return None


def _decode_chunked(data: bytes) -> bytes:
    """Decode chunked transfer encoding."""
    result = bytearray()
    pos = 0
    while pos < len(data):
        # Find chunk size line
        nl = data.find(b"\r\n", pos)
        if nl == -1:
            break
        size_str = data[pos:nl].decode().strip()
        if not size_str:
            pos = nl + 2
            continue
        chunk_size = int(size_str, 16)
        if chunk_size == 0:
            break
        pos = nl + 2
        result.extend(data[pos : pos + chunk_size])
        pos += chunk_size + 2  # skip trailing \r\n
    return bytes(result)


# ── Connection Handler ───────────────────────────────────────────────────────


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    app: App,
) -> None:
    """Handle a single HTTP connection."""
    addr = writer.get_extra_info("peername")
    try:
        data = await asyncio.wait_for(reader.read(65536), timeout=30)
        if not data:
            return

        request = parse_request(data)
        if request is None:
            writer.write(b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\n\r\n")
            await writer.drain()
            return

        logger.info("%s %s", request.method, request.path)

        response = await app.handle(request)

        if isinstance(response, StreamingResponse):
            await response.write(writer)
        else:
            writer.write(response.to_bytes())

        await writer.drain()
    except asyncio.TimeoutError:
        writer.write(b"HTTP/1.1 408 Request Timeout\r\ncontent-length: 0\r\n\r\n")
        await writer.drain()
    except ConnectionResetError:
        pass
    except Exception as e:
        logger.exception("Connection error: %s", e)
        try:
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\n\r\n")
            await writer.drain()
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ── Server Entry Point ──────────────────────────────────────────────────────


async def _serve(app: App, host: str, port: int) -> None:
    """Run the server."""
    # Run startup hooks
    await app.run_startup()

    server = await asyncio.start_server(
        lambda r, w: _handle_connection(r, w, app),
        host,
        port,
    )

    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    logger.info("Listening on %s", addrs)

    # Handle signals (skip in non-main threads)
    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass  # Windows or non-main thread

    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        await app.run_shutdown()
        server.close()
        await server.wait_closed()
        logger.info("Stopped.")


def run(app: App, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the server (blocking)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting %s v%s on %s:%d", app.title, app.version, host, port)
    try:
        asyncio.run(_serve(app, host, port))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
