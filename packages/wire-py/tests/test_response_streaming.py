"""Streaming RESPONSE bodies with a declared Content-Length (card ecea49c8).

Middleware like Starlette's GZip computes the full Content-Length up front,
declares it on ``http.response.start``, then streams the body across frames
with ``more_body=True`` (captured from the main app's SelectiveGZip:
start[content-length, content-encoding] / body more=True / empty body
more=False). uvicorn honours this; the wire host used to hard-raise
``streaming body with app Content-Length`` and 500 every compressed route.

The fix buffers the frames and decides at the terminal frame:
- total == declared  -> single Content-Length frame (fast-path shape);
- total != declared  -> the app lied: strip its Content-Length, emit
  wire-owned ``Transfer-Encoding: chunked`` — never desyncs, never 500s.

Buffering also means a mid-stream app crash writes NOTHING to the socket,
so the cycle still turns into a clean 500 (better than uvicorn, which may
have already streamed a partial 200).
"""

from __future__ import annotations

import socket

from wire.server import WireServer

# ── helpers ────────────────────────────────────────────────────────────


def _connect(server) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=5.0)


def _read_head(sock: socket.socket, buf: bytearray) -> tuple[int, dict[bytes, bytes]]:
    while True:
        end = buf.find(b"\r\n\r\n")
        if end >= 0:
            head = bytes(buf[:end])
            del buf[: end + 4]
            status = int(head.split(b"\r\n")[0].split()[1])
            headers: dict[bytes, bytes] = {}
            for line in head.split(b"\r\n")[1:]:
                name, _, value = line.partition(b":")
                headers[name.strip().lower()] = value.strip()
            return status, headers
        data = sock.recv(4096)
        assert data, "connection closed before a response head"
        buf += data


def _read_cl_response(sock: socket.socket, buf: bytearray) -> tuple[int, dict, bytes]:
    """One full Content-Length response; leftovers stay in buf."""
    status, headers = _read_head(sock, buf)
    length = int(headers[b"content-length"])
    while len(buf) < length:
        data = sock.recv(4096)
        assert data, "connection closed before a complete body"
        buf += data
    body = bytes(buf[:length])
    del buf[:length]
    return status, headers, body


def _read_chunked_response(sock: socket.socket, buf: bytearray) -> tuple[int, dict, bytes]:
    """One full chunked response (size lines hex, CRLF-framed, 0-terminated)."""
    status, headers = _read_head(sock, buf)
    assert headers.get(b"transfer-encoding") == b"chunked"
    assert b"content-length" not in headers  # the lie was stripped
    body = bytearray()
    while True:
        while b"\r\n" not in buf:
            data = sock.recv(4096)
            assert data, "connection closed before a chunk size"
            buf += data
        line, _, _ = bytes(buf).partition(b"\r\n")
        size = int(line.split(b";")[0], 16)
        del buf[: len(line) + 2]
        if size == 0:  # terminal chunk: consume the trailing CRLF
            while len(buf) < 2:
                data = sock.recv(4096)
                assert data, "connection closed before the chunk terminator"
                buf += data
            del buf[:2]
            return status, headers, bytes(body)
        while len(buf) < size + 2:  # chunk data + CRLF
            data = sock.recv(4096)
            assert data, "connection closed mid-chunk"
            buf += data
        body += buf[:size]
        del buf[: size + 2]


class _GetApp:
    """GET-only tiny request so keep-alive probes can't trip on upload bytes."""

    def __init__(self) -> None:
        self.requests = 0

    async def _handle(self, scope, receive, send) -> None:
        self.requests += 1
        await self.stream(send)

    async def stream(self, send) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def __call__(self, scope, receive, send) -> None:
        while True:  # drain any request body so keep-alive stays in sync
            message = await receive()
            if not message.get("more_body", False):
                break
        await self._handle(scope, receive, send)


class GzipStyleApp(_GetApp):
    """The captured SelectiveGZip frame shape: declared CL, streamed frames."""

    def __init__(self, chunks: list[bytes], declared: int) -> None:
        super().__init__()
        self.chunks = chunks
        self.declared = declared

    async def stream(self, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-length", str(self.declared).encode()),
                    (b"content-encoding", b"gzip"),
                ],
            }
        )
        for chunk in self.chunks[:-1]:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": self.chunks[-1], "more_body": False})


class CrashingStreamApp(GzipStyleApp):
    """Dies after the first streamed frame — head must never reach the wire."""

    async def stream(self, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(self.declared).encode())],
            }
        )
        await send({"type": "http.response.body", "body": self.chunks[0], "more_body": True})
        raise RuntimeError("app died mid-stream")


# ── declared-Content-Length streaming ──────────────────────────────────


def test_gzip_shape_streamed_cl_serves_200_and_keeps_alive():
    # the exact captured frame sequence: CL declared, body more=True,
    # empty terminal frame — was RuntimeError 500 on every gzip route
    payload = b'{"openapi":"3.1.0","paths":{' + b'"x":1,' * 1200 + b"}}"
    app = GzipStyleApp(chunks=[payload, b""], declared=len(payload))
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            buf = bytearray()
            sock.sendall(b"GET /openapi.json HTTP/1.1\r\nHost: t\r\n\r\n")
            status, headers, body = _read_cl_response(sock, buf)
            assert status == 200
            assert headers[b"content-length"] == str(len(payload)).encode()
            assert headers[b"content-encoding"] == b"gzip"
            assert body == payload  # byte-exact, nothing truncated
            # keep-alive survives: the declared length was truthful
            sock.sendall(b"GET /again HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, body = _read_cl_response(sock, buf)
            assert status == 200
            assert body == payload
            assert app.requests == 2  # second response really reused the conn
    finally:
        server.stop()


def test_multi_frame_streamed_cl_accumulates_to_one_length_frame():
    # the stream arrives in several frames (not just body + empty terminal)
    parts = [b"a" * 1000, b"b" * 2000, b"c" * 3000]
    payload = b"".join(parts)
    app = GzipStyleApp(chunks=parts, declared=len(payload))
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            buf = bytearray()
            sock.sendall(b"GET /multi HTTP/1.1\r\nHost: t\r\n\r\n")
            status, headers, body = _read_cl_response(sock, buf)
            assert status == 200
            assert headers[b"content-length"] == str(len(payload)).encode()
            assert body == payload
    finally:
        server.stop()


def test_declared_cl_mismatch_falls_back_to_chunked_not_desync():
    # the app declares a WRONG length: never 500, never trust the lie —
    # strip Content-Length, emit wire-owned chunked, body byte-exact.
    # streamed as body-more=True + empty terminal (the GZip shape)
    payload = b"z" * 200
    app = GzipStyleApp(chunks=[payload, b""], declared=9999)
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            buf = bytearray()
            sock.sendall(b"GET /liar HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, body = _read_chunked_response(sock, buf)
            assert status == 200
            assert body == payload  # truthful bytes despite the lying head
            # framing is exact — the next request on this connection parses
            sock.sendall(b"GET /again HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, body = _read_chunked_response(sock, buf)
            assert status == 200
            assert body == payload
    finally:
        server.stop()


def test_empty_first_frame_with_declared_cl_still_serves():
    # an empty more_body=True frame before the payload must not confuse
    # the buffer (contributes zero bytes, total still matches)
    payload = b"q" * 64
    app = GzipStyleApp(chunks=[b"", payload], declared=len(payload))
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            buf = bytearray()
            sock.sendall(b"GET /empty-first HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, body = _read_cl_response(sock, buf)
            assert status == 200
            assert body == payload
    finally:
        server.stop()


def test_crash_mid_buffer_sends_clean_500_and_closes():
    # nothing was written while buffering — the crash can still become a
    # clean 500 (uvicorn may have already streamed a partial 200)
    app = CrashingStreamApp(chunks=[b"x" * 100], declared=100)
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            buf = bytearray()
            sock.sendall(b"GET /boom HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, _ = _read_cl_response(sock, buf)
            assert status == 500
            sock.settimeout(2.0)
            assert sock.recv(1) == b""  # keep_alive=False on the error path
    finally:
        server.stop()
