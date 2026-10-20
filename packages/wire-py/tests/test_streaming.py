"""Streaming request bodies — the last uvicorn-parity gap (card c24543a4).

Bodies over ``body_buffer_threshold`` reach the app incrementally through
ASGI ``receive()`` (``http.request`` + ``more_body``) instead of being
buffered whole; ``max_body`` becomes a RUNNING total (413 via
``BodyTooLarge``). Small bodies keep the single-frame path — pinned by
regression here, so the hot path (chat JSON) never moves.

Reads happen off the event loop on a single-worker reader executor (the
same seam as the WS frame reader): a cancelled ``receive`` shares the
in-flight future instead of stranding a second reader on the socket.

Unconsumed upload bytes force the connection closed at cycle end — bytes
still in flight would desync the next request (uvicorn closes too).
"""

from __future__ import annotations

import asyncio
import socket
import time

from wire.server import WireServer

# ── helpers ────────────────────────────────────────────────────────────


def _connect(server) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=5.0)


def _read_response(sock: socket.socket, buf: bytearray) -> tuple[int, bytes]:
    """One full Content-Length response; pipelined leftovers stay in buf."""
    while True:
        end = buf.find(b"\r\n\r\n")
        if end >= 0:
            head = bytes(buf[:end])
            status = int(head.split(b"\r\n")[0].split()[1])
            length = 0
            for line in head.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
            if len(buf) >= end + 4 + length:
                body = bytes(buf[end + 4 : end + 4 + length])
                del buf[: end + 4 + length]
                return status, body
        data = sock.recv(4096)
        assert data, "connection closed before a complete response"
        buf += data


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


class DrainApp:
    """Raw ASGI: drains receive() counting frames + bytes, echoes the count.

    Counters reset per request — one instance serves keep-alive siblings.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.received = bytearray()

    async def __call__(self, scope, receive, send) -> None:
        self.frames = 0
        self.received = bytearray()
        while True:
            message = await receive()
            self.frames += 1
            self.received += message["body"]
            if not message.get("more_body", False):
                break
        payload = str(len(self.received)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class IgnoreBodyApp:
    """Answers immediately without ever reading the body (no Connection hdr)."""

    async def __call__(self, scope, receive, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


# ── streaming ──────────────────────────────────────────────────────────


def test_large_body_streams_in_multiple_frames_and_keeps_alive():
    app = DrainApp()
    server = WireServer(
        asgi_app=app, host="127.0.0.1", port=0, lifespan="off", body_buffer_threshold=1024
    )
    server.start()
    payload = b"x" * (2 * 1024 * 1024)
    try:
        with _connect(server) as sock:
            head = (
                f"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
            )
            sock.sendall(head + payload)
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            assert int(body) == len(payload)
            assert app.frames > 1  # actually streamed, not one frame
            assert bytes(app.received) == payload  # byte-exact
            # keep-alive survives a fully consumed upload
            sock.sendall(b"GET /again HTTP/1.1\r\nHost: t\r\n\r\n")
            status, body = _read_response(sock, buf)
            assert status == 200
            assert int(body) == 0
    finally:
        server.stop()


def test_small_body_stays_a_single_frame():
    # the hot path: default threshold, tiny JSON-sized body — one frame
    app = DrainApp()
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 11\r\n\r\nhello world")
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            assert int(body) == 11
            assert app.frames == 1
            assert bytes(app.received) == b"hello world"
    finally:
        server.stop()


def test_chunked_upload_streams_and_keeps_alive():
    app = DrainApp()
    server = WireServer(
        asgi_app=app, host="127.0.0.1", port=0, lifespan="off", body_buffer_threshold=1024
    )
    server.start()
    chunk = b"c" * 8192
    try:
        with _connect(server) as sock:
            sock.sendall(b"POST /up HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n")
            for _ in range(12):  # 96 KiB, undeclared size → streams
                sock.sendall(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            sock.sendall(b"0\r\n\r\n")
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            assert int(body) == 12 * len(chunk)
            assert app.frames > 1
            assert bytes(app.received) == chunk * 12
            sock.sendall(b"GET /again HTTP/1.1\r\nHost: t\r\n\r\n")
            status, body = _read_response(sock, buf)
            assert status == 200
    finally:
        server.stop()


def test_unconsumed_body_forces_close():
    # the app responds without reading the upload: response bytes go out,
    # then the connection closes — unread bytes in flight would desync
    server = WireServer(
        asgi_app=IgnoreBodyApp(),
        host="127.0.0.1",
        port=0,
        lifespan="off",
        body_buffer_threshold=1024,
    )
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 4096\r\n\r\n" + b"z" * 4096
            )
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 200
            sock.settimeout(2.0)
            assert sock.recv(1) == b""  # closed despite keep-alive framing
    finally:
        server.stop()


def test_chunked_over_running_cap_is_413():
    # chunked has no declared size to pre-check — the RUNNING total trips
    app = DrainApp()
    server = WireServer(
        asgi_app=app,
        host="127.0.0.1",
        port=0,
        lifespan="off",
        max_body=4096,
        body_buffer_threshold=512,
    )
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"POST /up HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n"
                + b"2000\r\n"
                + b"y" * 0x2000
                + b"\r\n"  # 8 KiB > cap 4 KiB
            )
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 413
    finally:
        server.stop()


def test_cancelled_receive_shares_the_in_flight_read():
    # wait_for(receive()) times out mid-read; the retry must await the SAME
    # in-flight future — no lost bytes, no duplicate, no second reader
    class SlowBodyApp:
        async def __call__(self, scope, receive, send) -> None:
            total = 0
            # first receive: the threshold read-ahead arrives instantly
            message = await receive()
            total += len(message["body"])
            more = bool(message.get("more_body", True))
            # subsequent receives block on the socket — the tight deadline
            # cancels the await mid-read; the retry must collect the SAME
            # in-flight read's result (no bytes lost, none duplicated)
            while more:
                try:
                    message = await asyncio.wait_for(receive(), timeout=0.05)
                except TimeoutError:
                    continue  # the read outlived the deadline — retry shares it
                total += len(message["body"])
                more = bool(message.get("more_body", False))
            payload = str(total).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", str(len(payload)).encode())],
                }
            )
            await send({"type": "http.response.body", "body": payload})

    server = WireServer(
        asgi_app=SlowBodyApp(), host="127.0.0.1", port=0, lifespan="off", body_buffer_threshold=512
    )
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 2000\r\n\r\n")
            sock.sendall(b"y" * 600)  # past the threshold — the reader attaches
            time.sleep(0.2)  # the in-flight read outlives two 50ms deadlines
            sock.sendall(b"y" * 1400)
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            assert int(body) == 2000  # nothing lost, nothing duplicated
    finally:
        server.stop()
