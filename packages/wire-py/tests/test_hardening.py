"""Tier-1 hardening for card ecea49c8 — the parity gaps vs uvicorn.

The A/B comparison made the capability gap list explicit; this file pins
the wire-local fixes:

- **proxy headers** — uvicorn translates ``X-Forwarded-For`` /
  ``X-Forwarded-Proto`` from trusted peers by default (peer in
  ``forwarded_allow_ips``, default ``127.0.0.1``); behind the Rust gateway
  ``scope["client"]``/``scheme`` must reflect the original client.
- **keep-alive idle timeout** — uvicorn ``timeout_keep_alive=5``: a peer
  that sits silent between requests (or before the first one) must not
  hold a thread forever.
- **request body cap** — wire buffers whole bodies (uvicorn streams), so
  an over-cap body must be 413, not memory.
- **Expect: 100-continue** — the server sends the interim response before
  reading the body; unknown expectations get 417.
- **access log** — uvicorn's per-request INFO line.

Smuggling guards (CL+TE, conflicting CL, obs-fold rejection) already live
in ``framing.py`` and ``test_framing.py`` — not re-tested here.
"""

from __future__ import annotations

import json
import logging
import socket

from wire.framing import Request
from wire.server import Response, WireServer

# ── helpers ────────────────────────────────────────────────────────────


def _connect(server) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=5.0)


def _parse_head(raw: bytes) -> tuple[int, dict[bytes, bytes]]:
    status = int(raw.split(b"\r\n")[0].split()[1])
    headers: dict[bytes, bytes] = {}
    for line in raw.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        headers[name.strip().lower()] = value.strip()
    return status, headers


def _read_head(sock: socket.socket, buf: bytearray) -> tuple[int, dict[bytes, bytes]]:
    """Read one response head; pipelined leftovers stay in ``buf``.

    One recv may carry MULTIPLE responses — tiny responses coalesce on
    loopback — so a helper that discards the surplus leaves the next read
    waiting on bytes the socket has already delivered.
    """
    while True:
        end = buf.find(b"\r\n\r\n")
        if end >= 0:
            status, headers = _parse_head(bytes(buf[:end]))
            del buf[: end + 4]
            return status, headers
        data = sock.recv(4096)
        assert data, "connection closed before a response head"
        buf += data


def _read_response(sock: socket.socket, buf: bytearray) -> tuple[int, bytes]:
    """One full Content-Length response; leftovers stay in ``buf``."""
    while True:
        end = buf.find(b"\r\n\r\n")
        if end >= 0:
            status, headers = _parse_head(bytes(buf[:end]))
            length = int(headers.get(b"content-length", b"0"))
            if len(buf) >= end + 4 + length:
                body = bytes(buf[end + 4 : end + 4 + length])
                del buf[: end + 4 + length]
                return status, body
        data = sock.recv(4096)
        assert data, "connection closed before a response"
        buf += data


def _ok_handler(request: Request) -> Response:
    return Response(status=200, headers={"Content-Type": "text/plain"}, body=b"ok")


class ScopeApp:
    """Echoes the interesting scope fields as JSON (proxy-header probe)."""

    async def __call__(self, scope, receive, send) -> None:
        body = json.dumps(
            {
                "client": list(scope["client"]) if scope.get("client") else None,
                "scheme": scope["scheme"],
                "path": scope["path"],
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ── proxy headers ──────────────────────────────────────────────────────


def test_x_forwarded_headers_translate_client_and_scheme():
    server = WireServer(asgi_app=ScopeApp(), host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"GET /ip HTTP/1.1\r\nHost: t\r\nX-Forwarded-For: 203.0.113.9\r\n"
                b"X-Forwarded-Proto: https\r\n\r\n"
            )
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            scope = json.loads(body)
            assert scope["client"][0] == "203.0.113.9"
            assert scope["scheme"] == "https"
    finally:
        server.stop()


def test_forwarded_headers_ignored_from_untrusted_listener():
    # trust nobody: the peer (loopback) is not in the (empty) allow list
    server = WireServer(
        asgi_app=ScopeApp(), host="127.0.0.1", port=0, lifespan="off", forwarded_allow_ips=""
    )
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"GET /ip HTTP/1.1\r\nHost: t\r\nX-Forwarded-For: 6.6.6.6\r\n"
                b"X-Forwarded-Proto: https\r\n\r\n"
            )
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            scope = json.loads(body)
            assert scope["client"][0] == "127.0.0.1"
            assert scope["scheme"] == "http"
    finally:
        server.stop()


def test_no_forwarded_headers_keeps_peer_identity():
    server = WireServer(asgi_app=ScopeApp(), host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(b"GET /ip HTTP/1.1\r\nHost: t\r\n\r\n")
            buf = bytearray()
            status, body = _read_response(sock, buf)
            assert status == 200
            scope = json.loads(body)
            assert scope["client"][0] == "127.0.0.1"
            assert scope["scheme"] == "http"
    finally:
        server.stop()


# ── keep-alive idle timeout ────────────────────────────────────────────


def test_silent_connection_closed_by_idle_timeout():
    # a peer that connects and says nothing must not hold a thread forever
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, keep_alive_timeout=0.4)
    server.start()
    try:
        with _connect(server) as sock:
            sock.settimeout(2.0)
            assert sock.recv(1) == b""  # idle → closed, no response owed
    finally:
        server.stop()


def test_idle_peer_closed_after_a_response():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, keep_alive_timeout=0.4)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(b"GET /a HTTP/1.1\r\nHost: t\r\n\r\n")
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 200
            sock.settimeout(2.0)
            assert sock.recv(1) == b""  # keep-alive window expires → close
    finally:
        server.stop()


def test_pipelined_requests_survive_the_idle_timer():
    # buffered bytes are NOT idle time: two requests in one write must both
    # be answered even with an aggressive keep-alive timeout (one shared
    # read buffer — responses coalesce on loopback)
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, keep_alive_timeout=0.4)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"GET /one HTTP/1.1\r\nHost: t\r\n\r\nGET /two HTTP/1.1\r\nHost: t\r\n\r\n"
            )
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 200
            status, _ = _read_response(sock, buf)
            assert status == 200  # second response survived the timer
    finally:
        server.stop()


# ── request body cap ───────────────────────────────────────────────────


def test_body_over_cap_is_413_and_connection_closes():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, max_body=512)
    server.start()
    try:
        with _connect(server) as sock:
            # declare an over-cap body without sending it — the wire must
            # refuse on the declaration alone (no read needed, then close)
            sock.sendall(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 2048\r\n\r\n")
            buf = bytearray()
            status, headers = _read_head(sock, buf)
            assert status == 413
            assert headers[b"connection"] == b"close"
    finally:
        server.stop()


def test_chunked_body_over_cap_is_413():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, max_body=512)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"POST /up HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n"
                + b"400\r\n"
                + b"x" * 0x400
                + b"\r\n"  # 1024 > cap 512
            )
            buf = bytearray()
            status, headers = _read_head(sock, buf)
            assert status == 413
            assert headers[b"connection"] == b"close"
    finally:
        server.stop()


def test_body_under_cap_passes_untouched():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0, max_body=512)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\n\r\nhello")
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 200
    finally:
        server.stop()


# ── Expect: 100-continue ───────────────────────────────────────────────


def test_expect_100_continue_exchange():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"POST /up HTTP/1.1\r\nHost: t\r\nExpect: 100-continue\r\nContent-Length: 5\r\n\r\n"
            )  # headers only — the body waits for the interim response
            raw = b""
            while b"\r\n\r\n" not in raw:
                data = sock.recv(4096)
                assert data, "closed before the 100 Continue"
                raw += data
            assert raw.startswith(b"HTTP/1.1 100 Continue")
            assert raw == b"HTTP/1.1 100 Continue\r\n\r\n"  # bare interim, no framing
            sock.sendall(b"hello")
            buf = bytearray()
            status, _ = _read_response(sock, buf)
            assert status == 200
    finally:
        server.stop()


def test_unknown_expectation_is_417_then_close():
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0)
    server.start()
    try:
        with _connect(server) as sock:
            sock.sendall(
                b"POST /up HTTP/1.1\r\nHost: t\r\nExpect: teleport-me\r\n"
                b"Content-Length: 5\r\n\r\nhello"
            )
            buf = bytearray()
            status, headers = _read_head(sock, buf)
            assert status == 417
            assert headers[b"connection"] == b"close"
    finally:
        server.stop()


# ── access log ─────────────────────────────────────────────────────────


def test_access_log_line_handler_mode(caplog):
    server = WireServer(handler=_ok_handler, host="127.0.0.1", port=0)
    server.start()
    try:
        with caplog.at_level(logging.INFO, logger="wire.access"):
            with _connect(server) as sock:
                sock.sendall(b"GET /hello HTTP/1.1\r\nHost: t\r\n\r\n")
                buf = bytearray()
                status, _ = _read_response(sock, buf)
                assert status == 200
    finally:
        server.stop()
    lines = [r.getMessage() for r in caplog.records if r.name == "wire.access"]
    assert any('"GET /hello HTTP/1.1" 200' in line for line in lines), lines


def test_access_log_line_asgi_mode(caplog):
    server = WireServer(asgi_app=ScopeApp(), host="127.0.0.1", port=0, lifespan="off")
    server.start()
    try:
        with caplog.at_level(logging.INFO, logger="wire.access"):
            with _connect(server) as sock:
                sock.sendall(b"GET /hello HTTP/1.1\r\nHost: t\r\n\r\n")
                buf = bytearray()
                status, _ = _read_response(sock, buf)
                assert status == 200
    finally:
        server.stop()
    lines = [r.getMessage() for r in caplog.records if r.name == "wire.access"]
    assert any('"GET /hello HTTP/1.1" 200' in line for line in lines), lines
