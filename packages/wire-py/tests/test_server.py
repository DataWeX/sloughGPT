"""WireServer — stdlib sync TCP accept loop with keep-alive (card ecea49c8, phase 1).

One thread per connection; the accept loop and the connection loops are all
synchronous (no asyncio anywhere in this package — the ASGI adapter in phase 2
sits ABOVE this seam, per the execution-philosophy rule).
"""

from __future__ import annotations

import socket
import threading

import pytest
from wire.framing import Request
from wire.server import Response, WireServer


def _recv_all(sock: socket.socket, timeout: float = 3.0) -> bytes:
    sock.settimeout(timeout)
    chunks: list[bytes] = []
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
            # responses in these tests always end with a body the test can
            # size-check; one recv round is not enough, but once we have the
            # full expected response the caller stops reading via shutdown.
            if b"\r\n\r\n" in b"".join(chunks):
                head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
                length = _header_int(head, b"Content-Length")
                if length is not None and len(body) >= length:
                    break
                if _header_has(head, b"Transfer-Encoding") and body.endswith(b"0\r\n\r\n"):
                    break
    except TimeoutError:
        pass
    return b"".join(chunks)


def _header_int(head: bytes, name: bytes) -> int | None:
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(name.lower() + b":"):
            try:
                return int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _header_has(head: bytes, name: bytes) -> bool:
    """Match ``b"Header"`` or ``b"Header: value"`` against response head lines."""
    want = name.lower()
    want_key, _, want_val = want.partition(b":")
    want_key, want_val = want_key.strip(), want_val.strip()
    for line in head.split(b"\r\n")[1:]:
        if b":" not in line:
            continue
        key, _, value = line.partition(b":")
        key, value = key.strip().lower(), value.strip().lower()
        if key != want_key:
            continue
        if not want_val or value == want_val:
            return True
    return False


def _parse_response(raw: bytes) -> tuple[int, bytes, bytes]:
    head, sep, body = raw.partition(b"\r\n\r\n")
    assert sep, f"no header/body separator in {raw[:120]!r}"
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    length = _header_int(head, b"Content-Length")
    if length is not None:
        body = body[:length]
    return status, head, body


@pytest.fixture()
def echo_server():
    def handler(req: Request) -> Response:
        if req.target == "/boom":
            raise RuntimeError("handler exploded")
        if req.target == "/stream":
            return Response(200, {"Content-Type": "text/plain"}, b"chunked!", chunked=True)
        return Response(
            200,
            {"Content-Type": "text/plain"},
            req.method.encode() + b" " + req.target.encode() + b" " + req.body,
        )

    server = WireServer(handler, host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _connect(server: WireServer) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=3.0)
    return sock


def test_get_round_trip(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /hello HTTP/1.1\r\nHost: t\r\n\r\n")
        status, head, body = _parse_response(_recv_all(sock))
        assert status == 200
        assert body == b"GET /hello "
        assert _header_int(head, b"Content-Length") == len(body)
    finally:
        sock.close()


def test_keep_alive_serves_two_requests_on_one_connection(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /one HTTP/1.1\r\nHost: t\r\n\r\n")
        status, _, body = _parse_response(_recv_all(sock))
        assert status == 200
        assert body == b"GET /one "

        # second request on the SAME socket — the connection must persist
        sock.sendall(b"GET /two HTTP/1.1\r\nHost: t\r\n\r\n")
        status, _, body = _parse_response(_recv_all(sock))
        assert status == 200
        assert body == b"GET /two "
    finally:
        sock.close()


def test_connection_close_terminates_after_response(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /bye HTTP/1.0\r\n\r\n")  # 1.0 defaults to close
        raw = _recv_all(sock)
        status, head, _ = _parse_response(raw)
        assert status == 200
        assert _header_has(head, b"Connection: close") or not _header_has(
            head, b"Connection: keep-alive"
        )
        # server must close the connection after the response
        sock.settimeout(3.0)
        assert sock.recv(1) == b""
    finally:
        sock.close()


def test_malformed_request_gets_400_and_close(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GARBAGE\r\n\r\n")
        status, head, _ = _parse_response(_recv_all(sock))
        assert status == 400
        assert _header_has(head, b"Connection: close")
        assert sock.recv(1) == b""
    finally:
        sock.close()


def test_handler_exception_becomes_500_and_close(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /boom HTTP/1.1\r\nHost: t\r\n\r\n")
        status, head, _ = _parse_response(_recv_all(sock))
        assert status == 500
        assert _header_has(head, b"Connection: close")
    finally:
        sock.close()


def test_post_body_is_echoed(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        body = b'{"tokens": [1, 2, 3]}'
        sock.sendall(
            b"POST /echo HTTP/1.1\r\nHost: t\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        status, _, got = _parse_response(_recv_all(sock))
        assert status == 200
        assert got == b"POST /echo " + body
    finally:
        sock.close()


def test_chunked_response_framing(echo_server: WireServer):
    sock = _connect(echo_server)
    try:
        sock.sendall(b"GET /stream HTTP/1.1\r\nHost: t\r\n\r\n")
        raw = _recv_all(sock)
        head, _, body = raw.partition(b"\r\n\r\n")
        assert _header_has(head, b"Transfer-Encoding: chunked")
        assert body.endswith(b"0\r\n\r\n")
        assert not _header_has(head, b"Content-Length")
    finally:
        sock.close()


def test_server_stops_cleanly(echo_server: WireServer):
    """stop() closes the listener — no zombie acceptor holding the port."""
    port = echo_server.port
    echo_server.stop()
    # after stop, connecting must fail (listener closed)
    with pytest.raises((ConnectionRefusedError, OSError)):
        sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
        sock.close()


def test_concurrent_clients(echo_server: WireServer):
    errors: list[BaseException] = []

    def client(i: int) -> None:
        try:
            sock = _connect(echo_server)
            try:
                sock.sendall(f"GET /c{i} HTTP/1.1\r\nHost: t\r\n\r\n".encode())
                status, _, body = _parse_response(_recv_all(sock))
                assert status == 200
                assert body == f"GET /c{i} ".encode()
            finally:
                sock.close()
        except BaseException as exc:  # noqa: BLE001 - collected for assertion
            errors.append(exc)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors
