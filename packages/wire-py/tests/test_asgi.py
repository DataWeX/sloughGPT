"""ASGI host adapter — drive an ASGI app from the wire (card ecea49c8, phase 2).

The adapter is the ONLY place asyncio is allowed to live in this package:
`WireServer` stays sync and hands one parsed request over; the app runs in
an event loop created and torn down inside the adapter, and response bytes
stream straight back through the sync socket write. That is the
sync-core/async-seam split the card mandates.

SSE works because response frames are written as they arrive — no
buffering of the whole body. Hermetic apps only (no starlette/fastapi in
these tests): the package must stay stdlib-only, and a hand-written ASGI
app pins the protocol contract itself.
"""

from __future__ import annotations

import socket

import pytest
from wire.server import WireServer

# ── helpers ───────────────────────────────────────────────────────────


def _chunk_decode(body: bytes) -> bytes:
    out = bytearray()
    pos = 0
    while True:
        line_end = body.find(b"\r\n", pos)
        assert line_end != -1, f"no chunk size line at {pos}: {body!r}"
        size = int(body[pos:line_end].split(b";", 1)[0], 16)
        pos = line_end + 2
        if size == 0:
            assert body[pos : pos + 2] == b"\r\n", "missing terminal CRLF"
            return bytes(out)
        out += body[pos : pos + size]
        assert body[pos + size : pos + size + 2] == b"\r\n"
        pos += size + 2


def _recv_response(sock: socket.socket, timeout: float = 3.0) -> tuple[int, dict, bytes]:
    """Read exactly one response (CL- or chunked-framed) off ``sock``."""
    sock.settimeout(timeout)
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = sock.recv(65536)
        assert data, f"connection closed before headers: {buf!r}"
        buf += data
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, _, value = line.partition(b":")
        headers[key.strip().decode("latin-1").lower()] = value.strip().decode("latin-1")

    if headers.get("transfer-encoding") == "chunked":
        # pull until the terminal chunk is visible, then decode
        body = rest
        while not body.endswith(b"0\r\n\r\n"):
            data = sock.recv(65536)
            assert data, "connection closed mid-chunk"
            body += data
        return status, headers, _chunk_decode(body)

    length = int(headers.get("content-length", "0"))
    body = rest
    while len(body) < length:
        data = sock.recv(65536)
        assert data, "connection closed mid-body"
        body += data
    return status, headers, body[:length]


def _recv_closed(sock: socket.socket) -> bool:
    sock.settimeout(3.0)
    try:
        return sock.recv(1) == b""
    except (TimeoutError, OSError):
        return False


# ── hermetic ASGI apps ────────────────────────────────────────────────


class EchoApp:
    """Records every scope; echoes method+path+body back."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise RuntimeError("disconnect before body")
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        payload = scope["method"].encode() + b" " + scope["raw_path"] + b" " + body
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class SseApp:
    """Streams three SSE frames with no content-length — must go chunked."""

    async def __call__(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for i in range(3):
            await send(
                {
                    "type": "http.response.body",
                    "body": f"data: ping {i}\n\n".encode(),
                    "more_body": i < 2,
                }
            )


class CloseApp:
    """ASGI-level Connection: close must end the keep-alive loop."""

    async def __call__(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [(b"connection", b"close")],
            }
        )
        await send({"type": "http.response.body", "body": b""})


class ExplodingApp:
    """Crash before the response starts — the client must see 500 + close."""

    async def __call__(self, scope, receive, send):
        raise RuntimeError("asgi boom")


def _connect(server: WireServer) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=3.0)


def _serve(app):
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0)
    server.start()
    return server


@pytest.fixture()
def echo_server():
    server = _serve(EchoApp())
    try:
        yield server
    finally:
        server.stop()


# ── tests ─────────────────────────────────────────────────────────────


def test_scope_is_built_from_the_request(echo_server: WireServer):
    app = echo_server.asgi_app  # type: ignore[attr-defined]
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /v1/chat?stream=1 HTTP/1.1\r\nHost: wire.test\r\n\r\n")
        status, headers, body = _recv_response(sock)
        assert status == 200
        assert body == b"GET /v1/chat "

        # phase 3: the app sees a lifespan scope first (auto-mode attempt —
        # EchoApp is lifespan-unaware); the http scope under test is the
        # one built from the request
        scope = next(s for s in app.scopes if s["type"] == "http")
        assert scope["asgi"]["version"] == "3.0"
        assert scope["method"] == "GET"
        assert scope["path"] == "/v1/chat"
        assert scope["raw_path"] == b"/v1/chat"
        assert scope["query_string"] == b"stream=1"
        assert scope["scheme"] == "http"
        assert scope["http_version"] == "1.1"
        assert (b"host", b"wire.test") in scope["headers"]
        assert all(k == k.lower() for k, _ in scope["headers"])  # ASGI lowercases
        assert isinstance(scope["client"], tuple)
        assert isinstance(scope["server"], tuple)
    finally:
        sock.close()


def test_post_body_flows_through_receive(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        body = b'{"prompt": "hi"}'
        sock.sendall(
            b"POST /gen HTTP/1.1\r\nHost: t\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        status, _, got = _recv_response(sock)
        assert status == 200
        assert got == b"POST /gen " + body
    finally:
        sock.close()


def test_keep_alive_persists_across_asgi_requests(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /a HTTP/1.1\r\nHost: t\r\n\r\n")
        _, _, first = _recv_response(sock)
        sock.sendall(b"GET /b HTTP/1.1\r\nHost: t\r\n\r\n")
        _, _, second = _recv_response(sock)
        assert first == b"GET /a "
        assert second == b"GET /b "
    finally:
        sock.close()


def test_streaming_response_without_length_is_chunked_and_intact():
    """The SSE contract: frames stream out, decoder sees every byte."""
    server = _serve(SseApp())
    try:
        sock = _connect(server)
        try:
            sock.sendall(b"GET /events HTTP/1.1\r\nHost: t\r\n\r\n")
            status, headers, body = _recv_response(sock)
            assert status == 200
            assert headers["content-type"] == "text/event-stream"
            assert headers.get("transfer-encoding") == "chunked"
            assert "content-length" not in headers
            assert body == b"data: ping 0\n\ndata: ping 1\n\ndata: ping 2\n\n"
            # connection survives the stream: second request still served
            sock.sendall(b"GET /again HTTP/1.1\r\nHost: t\r\n\r\n")
            status, _, _ = _recv_response(sock)
            assert status == 200
        finally:
            sock.close()
    finally:
        server.stop()


def test_asgi_connection_close_header_ends_keep_alive():
    server = _serve(CloseApp())
    try:
        sock = _connect(server)
        try:
            sock.sendall(b"GET /x HTTP/1.1\r\nHost: t\r\n\r\n")
            status, headers, _ = _recv_response(sock)
            assert status == 204
            assert headers.get("connection") == "close"
            assert _recv_closed(sock)
        finally:
            sock.close()
    finally:
        server.stop()


def test_asgi_exception_before_start_is_500_and_close():
    server = _serve(ExplodingApp())
    try:
        sock = _connect(server)
        try:
            sock.sendall(b"GET /x HTTP/1.1\r\nHost: t\r\n\r\n")
            status, headers, _ = _recv_response(sock)
            assert status == 500
            assert headers.get("connection") == "close"
            assert _recv_closed(sock)
        finally:
            sock.close()
    finally:
        server.stop()


def test_malformed_request_still_gets_wire_400_in_asgi_mode():
    server = _serve(EchoApp())
    try:
        sock = _connect(server)
        try:
            sock.sendall(b"NOPE\r\n\r\n")
            status, _, _ = _recv_response(sock)
            assert status == 400
        finally:
            sock.close()
    finally:
        server.stop()
