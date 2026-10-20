"""RFC6455 WebSocket — the flip gate for card ecea49c8 (minimal scope).

One route-level truth: the app's only WS consumer is FastAPI
``add_api_websocket_route("/ws/generate", ...)`` using Starlette's
``WebSocket`` API (``accept`` / ``receive_text`` / ``send_json`` /
``close``). These tests pin the handshake (101 + accept value), the frame
codec (client-mandatory masking, all three length forms, fragmentation,
ping/close control frames), the ASGI websocket message contract, and the
concurrent-send property (server may push while a receive is pending —
the design pin: frame reads run off-loop, never blocking sends).
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket

import pytest
from wire.server import WireServer

try:  # pragma: no cover - absent only in fastapi-less envs
    # Module-level on purpose: this file uses `from __future__ import
    # annotations`, so FastAPI resolves the endpoint's `websocket: WebSocket`
    # annotation against MODULE globals — importing WebSocket inside the test
    # would silently degrade it to a required query param (loc: query,
    # websocket) and the route would 403-reject instead of accepting.
    from fastapi import FastAPI, WebSocket
except ImportError:  # pragma: no cover
    FastAPI = WebSocket = None  # type: ignore[assignment]

_WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_KEY_RAW = b"0123456789abcdef"  # 16 bytes → fixed Sec-WebSocket-Key
_KEY_B64 = base64.b64encode(_KEY_RAW).decode()
_MASK = b"\x11\x22\x33\x44"

OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


# ── client-side helpers ───────────────────────────────────────────────


def _connect(server: WireServer) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=5.0)


def _handshake(
    sock: socket.socket,
    *,
    key: str = _KEY_B64,
    version: str = "13",
    path: str = "/ws/echo",
) -> tuple[int, dict[bytes, bytes], bytes]:
    """Send the upgrade request; returns (status, headers, leftover bytes)."""
    req = (
        f"GET {path} HTTP/1.1\r\nHost: t\r\nUpgrade: websocket\r\n"
        f"Connection: keep-alive, Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: {version}\r\n\r\n"
    )
    sock.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = sock.recv(4096)
        assert data, "closed before handshake response"
        buf += data
    head, _, rest = buf.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n")[0].split()[1])
    headers: dict[bytes, bytes] = {}
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        headers[name.strip().lower()] = value.strip()
    return status, headers, rest


def _client_frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
    """Client frames MUST be masked (RFC6455 §5.3)."""
    head = bytes([(0x80 if fin else 0) | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + n.to_bytes(2, "big")
    else:
        head += bytes([0x80 | 127]) + n.to_bytes(8, "big")
    masked = bytes(b ^ _MASK[i % 4] for i, b in enumerate(payload))
    return head + _MASK + masked


class _Frames:
    """Server-frame reader over a socket + any bytes left from handshake."""

    def __init__(self, sock: socket.socket, initial: bytes = b"") -> None:
        self._sock = sock
        self._buf = initial

    def _take(self, n: int) -> bytes:
        while len(self._buf) < n:
            data = self._sock.recv(65536)
            assert data, f"eof mid-frame (have {len(self._buf)}, need {n})"
            self._buf += data
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def read(self) -> tuple[int, bytes]:
        """One server frame → (opcode, payload); asserts unmasked (server→client)."""
        b0, b1 = self._take(2)
        assert not (b0 & 0x70), "RSV bits must be clear"
        assert not (b1 & 0x80), "server frames must NOT be masked"
        fin, opcode = bool(b0 & 0x80), b0 & 0x0F
        n = b1 & 0x7F
        if n == 126:
            n = int.from_bytes(self._take(2), "big")
        elif n == 127:
            n = int.from_bytes(self._take(8), "big")
        payload = self._take(n)
        assert fin, "fragmented server frame not expected in these tests"
        return opcode, payload


def _close_code(payload: bytes) -> int:
    return int.from_bytes(payload[:2], "big") if len(payload) >= 2 else 1005


# ── hermetic ASGI websocket app ───────────────────────────────────────


class WsApp:
    """Accepts, echoes text/binary, records what the app saw."""

    def __init__(self, *, reject: bool = False, push_first: bool = False) -> None:
        self.received: list[tuple[str, object]] = []
        self.disconnects: list[int] = []
        self.scopes: list[dict] = []
        self._reject = reject
        self._push_first = push_first

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        message = await receive()
        assert message["type"] == "websocket.connect"
        if self._reject:
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({"type": "websocket.accept"})
        if self._push_first:
            await send({"type": "websocket.send", "text": "welcome"})
        while True:
            message = await receive()
            if message["type"] == "websocket.disconnect":
                self.disconnects.append(int(message.get("code", 1005)))
                return
            kind = "text" if "text" in message else "bytes"
            self.received.append((kind, message[kind]))
            await send({"type": "websocket.send", kind: message[kind]})


def _serve(app) -> WireServer:
    # lifespan off: these tests are about the upgrade branch, and the
    # lifespan-attempt scope would pollute app.scopes
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="off")
    server.start()
    return server


# ── handshake ─────────────────────────────────────────────────────────


def test_handshake_returns_101_with_spec_accept_value():
    app = WsApp()
    server = _serve(app)
    try:
        with _connect(server) as sock:
            status, headers, _ = _handshake(sock)
            assert status == 101
            assert headers[b"upgrade"] == b"websocket"
            assert b"upgrade" in headers[b"connection"].lower()
            expected = base64.b64encode(hashlib.sha1(_KEY_RAW + _WS_GUID).digest())
            assert headers[b"sec-websocket-accept"] == expected
    finally:
        server.stop()
    scope = app.scopes[0]
    assert scope["type"] == "websocket"
    assert scope["path"] == "/ws/echo"
    assert scope["scheme"] == "ws"


def test_handshake_rejects_missing_key_with_400():
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"GET /ws/echo HTTP/1.1\r\nHost: t\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            head = b""
            while b"\r\n\r\n" not in head:
                data = sock.recv(4096)
                assert data, "closed before response"
                head += data
            assert head.startswith(b"HTTP/1.1 400")
    finally:
        server.stop()


def test_handshake_rejects_bad_version_with_426():
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            status, headers, _ = _handshake(sock, version="12")
            assert status == 426
            assert headers[b"sec-websocket-version"] == b"13"
    finally:
        server.stop()


def test_reject_before_accept_is_403_not_101():
    server = _serve(WsApp(reject=True))
    try:
        with _connect(server) as sock:
            status, _headers, _rest = _handshake(sock)
            assert status == 403
    finally:
        server.stop()


# ── frames ────────────────────────────────────────────────────────────


def test_text_echo_round_trip():
    app = WsApp()
    server = _serve(app)
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            sock.sendall(_client_frame(OP_TEXT, b"hello ws"))
            opcode, payload = frames.read()
            assert (opcode, payload) == (OP_TEXT, b"hello ws")
            sock.sendall(_client_frame(OP_CLOSE, (1000).to_bytes(2, "big")))
            opcode, payload = frames.read()
            assert opcode == OP_CLOSE
            assert _close_code(payload) == 1000
    finally:
        server.stop()
    assert app.received == [("text", "hello ws")]
    assert app.disconnects == [1000]


def test_binary_echo_16bit_and_64bit_lengths():
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            for size in (200, 70_000):  # 16-bit then 64-bit length form
                blob = bytes(range(256)) * (size // 256) + bytes(size % 256)
                sock.sendall(_client_frame(OP_BIN, blob))
                opcode, payload = frames.read()
                assert (opcode, payload) == (OP_BIN, blob)
    finally:
        server.stop()


def test_fragmented_text_assembles_into_one_message():
    app = WsApp()
    server = _serve(app)
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            sock.sendall(_client_frame(OP_TEXT, b"frag-", fin=False))
            sock.sendall(_client_frame(OP_CONT, b"mented", fin=True))
            opcode, payload = frames.read()
            assert (opcode, payload) == (OP_TEXT, b"frag-mented")
    finally:
        server.stop()
    assert app.received == [("text", "frag-mented")]


def test_client_ping_gets_pong_and_does_not_reach_app():
    app = WsApp()
    server = _serve(app)
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            sock.sendall(_client_frame(OP_PING, b"probe"))
            opcode, payload = frames.read()
            assert (opcode, payload) == (OP_PONG, b"probe")
            sock.sendall(_client_frame(OP_TEXT, b"after-ping"))
            opcode, payload = frames.read()
            assert (opcode, payload) == (OP_TEXT, b"after-ping")
    finally:
        server.stop()
    assert app.received == [("text", "after-ping")]  # ping never surfaced


def test_unmasked_client_frame_is_a_protocol_error_close():
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            # mask bit clear — RFC6455 §5.1: client frames MUST be masked
            sock.sendall(bytes([0x80 | OP_TEXT, 0x02]) + b"hi")
            opcode, payload = frames.read()
            assert opcode == OP_CLOSE
            assert _close_code(payload) == 1002
    finally:
        server.stop()


def test_server_can_push_while_receive_is_pending():
    # the design pin: frame reads never block the loop, so an app may send
    # before the client has sent anything (no deadlock)
    server = _serve(WsApp(push_first=True))
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            opcode, payload = frames.read()  # arrives with no client input
            assert (opcode, payload) == (OP_TEXT, b"welcome")
            sock.sendall(_client_frame(OP_TEXT, b"and-you"))
            opcode, payload = frames.read()
            assert (opcode, payload) == (OP_TEXT, b"and-you")
    finally:
        server.stop()


def test_server_close_ends_the_connection_with_eof():
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            sock.sendall(_client_frame(OP_CLOSE, (1001).to_bytes(2, "big")))
            opcode, _payload = frames.read()
            assert opcode == OP_CLOSE
            sock.settimeout(3.0)
            assert sock.recv(1) == b""  # server closes TCP after close frame
    finally:
        server.stop()


def test_plain_http_requests_untouched_by_upgrade_branch():
    # non-upgrade traffic must never enter the websocket path
    server = _serve(WsApp())
    try:
        with _connect(server) as sock:
            sock.sendall(b"GET /plain HTTP/1.1\r\nHost: t\r\n\r\n")
            buf = b""
            while b"\r\n\r\n" not in buf:
                data = sock.recv(4096)
                assert data
                buf += data
            # WsApp is not an http app: the adapter's 500 path, but crucially
            # NOT a 101 and not a hang
            assert not buf.startswith(b"HTTP/1.1 101")
    finally:
        server.stop()


def test_cancelled_receive_does_not_strand_session_teardown():
    # starlette's receive-with-timeout CANCELS a pending receive; the
    # in-flight read must not block session end (asyncio.run joins the
    # executor it ran on → deadlock if that read can never complete).
    import asyncio

    class FlakyApp:
        async def __call__(self, scope, receive, send) -> None:
            await receive()  # connect
            await send({"type": "websocket.accept"})
            try:
                await asyncio.wait_for(receive(), timeout=0.15)  # client silent
            except TimeoutError:
                pass
            await send({"type": "websocket.close", "code": 1001})

    server = _serve(FlakyApp())
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock)
            assert status == 101
            frames = _Frames(sock, rest)
            opcode, payload = frames.read()  # app's close after the timeout
            assert (opcode, payload) == (OP_CLOSE, (1001).to_bytes(2, "big"))
            sock.settimeout(3.0)
            assert sock.recv(1) == b""  # session ended, TCP closed
    finally:
        server.stop()  # must return promptly — no stranded join


# ── real-app interop (the flip-gate acceptance) ───────────────────────


def test_fastapi_websocket_route_interop():
    # mirrors the app's actual surface: add_api_websocket_route +
    # accept / receive_text / send_json / close (inference.py ws_generate)
    if FastAPI is None:
        pytest.skip("fastapi not installed")

    app = FastAPI()

    @app.websocket("/ws/generate")
    async def ws_generate(websocket: WebSocket) -> None:
        await websocket.accept()
        raw = await websocket.receive_text()
        await websocket.send_json({"echo": raw})
        await websocket.close()

    server = _serve(app)
    try:
        with _connect(server) as sock:
            status, _, rest = _handshake(sock, path="/ws/generate")
            assert status == 101
            frames = _Frames(sock, rest)
            sock.sendall(_client_frame(OP_TEXT, b'{"prompt": "hi"}'))
            opcode, payload = frames.read()
            assert opcode == OP_TEXT
            assert json.loads(payload) == {"echo": '{"prompt": "hi"}'}
            opcode, payload = frames.read()
            assert opcode == OP_CLOSE
            sock.settimeout(3.0)
            assert sock.recv(1) == b""  # TCP closes after the close frame
    finally:
        server.stop()
