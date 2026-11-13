"""ASGI lifespan protocol — startup/shutdown handshakes (card ecea49c8, phase 3).

The http cycles run on short-lived per-request loops; the lifespan protocol
must run exactly ONCE across the server's life: startup completes before the
first request can be served, shutdown completes after the last. These tests
pin the ordering, the failure modes (strict ``lifespan="on"`` vs uvicorn-style
``auto`` fallback vs ``off``), and real FastAPI interop — the acceptance for
hosting the actual app's StartupOrchestrator under this wire.
"""

from __future__ import annotations

import json
import socket

import pytest
from wire.asgi_host import Lifespan, LifespanError
from wire.framing import Request
from wire.server import Response, WireServer

# ── helpers ───────────────────────────────────────────────────────────


def _connect(server: WireServer) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=3.0)


def _get(sock: socket.socket, target: str = "/ping") -> bytes:
    sock.sendall(f"GET {target} HTTP/1.1\r\nHost: t\r\n\r\n".encode())
    chunks: list[bytes] = []
    while True:
        data = sock.recv(65536)
        assert data, "connection closed before a full response"
        chunks.append(data)
        raw = b"".join(chunks)
        if b"\r\n\r\n" not in raw:
            continue
        head, _, body = raw.partition(b"\r\n\r\n")
        headers = {}
        for line in head.split(b"\r\n")[1:]:
            key, _, value = line.partition(b":")
            headers[key.strip().lower()] = value.strip()
        length = int(headers.get(b"content-length", b"0"))
        if len(body) >= length:
            return body[:length]


# ── hermetic lifespan apps ────────────────────────────────────────────


class LifespanApp:
    """Handles lifespan scopes, records phases, echoes them in the body."""

    def __init__(
        self,
        *,
        fail_startup: bool = False,
        fail_message: str = "startup refused",
        crash_startup: bool = False,
        hang_startup: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.scope: dict | None = None
        self._fail_startup = fail_startup
        self._fail_message = fail_message
        self._crash_startup = crash_startup
        self._hang_startup = hang_startup
        self._fail_shutdown = fail_shutdown

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            self.scope = scope
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    if self._hang_startup:
                        import asyncio

                        await asyncio.Event().wait()  # never completes
                    if self._crash_startup:
                        raise RuntimeError("lifespan boom")
                    if self._fail_startup:
                        await send(
                            {"type": "lifespan.startup.failed", "message": self._fail_message}
                        )
                        return
                    self.events.append("startup")
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    if self._fail_shutdown:
                        await send(
                            {"type": "lifespan.shutdown.failed", "message": "shutdown refused"}
                        )
                        return
                    self.events.append("shutdown")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        payload = ",".join(self.events).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _serve(app, mode: str = "on", timeout: float = 5.0) -> WireServer:
    server = WireServer(
        asgi_app=app, host="127.0.0.1", port=0, lifespan=mode, lifespan_timeout=timeout
    )
    server.start()
    return server


# ── ordering + protocol ───────────────────────────────────────────────


def test_startup_completes_before_first_request_is_served():
    app = LifespanApp()
    server = _serve(app)
    try:
        # the accept thread only spawns AFTER the handshake — no request can
        # possibly be served before startup finished
        assert app.events == ["startup"]
        with _connect(server) as sock:
            body = _get(sock)
        assert body == b"startup"
    finally:
        server.stop()
    assert app.events == ["startup", "shutdown"]


def test_lifespan_scope_is_spec_conformant():
    app = LifespanApp()
    server = _serve(app)
    try:
        scope = app.scope
        assert scope is not None
        assert scope["type"] == "lifespan"
        assert scope["asgi"]["version"].startswith("3.")
        assert isinstance(scope["state"], dict)
    finally:
        server.stop()


# ── failure modes ─────────────────────────────────────────────────────


def test_startup_failure_raises_when_lifespan_is_on():
    app = LifespanApp(fail_startup=True, fail_message="nope: engine cold")
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="on", lifespan_timeout=5.0)
    with pytest.raises(LifespanError, match="nope: engine cold"):
        server.start()
    # start failed → nothing is serving and the listener is unwound
    with pytest.raises(RuntimeError, match="not started"):
        _ = server.port
    server.stop()  # a failed start must leave stop() a safe no-op


def test_raw_crash_in_lifespan_startup_raises_with_exception_text():
    app = LifespanApp(crash_startup=True)
    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="on")
    with pytest.raises(LifespanError, match="lifespan boom"):
        server.start()
    server.stop()


def test_startup_timeout_fails_fast_and_leaves_no_thread():
    app = LifespanApp(hang_startup=True)
    lifespan = Lifespan(app)
    with pytest.raises(LifespanError, match="timeout"):
        lifespan.startup(timeout=0.3)
    # the aborted handshake must not leak a parked loop thread
    assert lifespan.thread is None


def test_auto_mode_continues_without_lifespan():
    # lifespan-unaware app (crashes on the lifespan scope) — uvicorn's
    # default is to log and keep serving; ours must too
    app = LifespanApp(crash_startup=True)
    server = _serve(app, mode="auto")
    try:
        with _connect(server) as sock:
            body = _get(sock)
        assert body == b""  # never recorded startup, but http still serves
    finally:
        server.stop()


def test_shutdown_failure_does_not_break_stop():
    app = LifespanApp(fail_shutdown=True)
    server = _serve(app, mode="on")
    try:
        with _connect(server) as sock:
            _get(sock)
    finally:
        server.stop()  # must not raise despite lifespan.shutdown.failed
    assert app.events == ["startup"]


def test_off_mode_never_touches_the_app():
    app = LifespanApp()
    server = _serve(app, mode="off")
    try:
        assert app.scope is None
        with _connect(server) as sock:
            assert _get(sock) == b""
    finally:
        server.stop()


# ── unit: the driver itself ───────────────────────────────────────────


def test_driver_handshake_is_idempotent_on_shutdown():
    app = LifespanApp()
    lifespan = Lifespan(app)
    lifespan.startup(timeout=5.0)
    assert app.events == ["startup"]
    lifespan.shutdown(timeout=5.0)
    lifespan.shutdown(timeout=5.0)  # double-stop is a no-op, not an error
    assert app.events == ["startup", "shutdown"]
    assert lifespan.thread is None


def test_handler_mode_never_constructs_a_driver():
    def handler(req: Request) -> Response:
        return Response(200, {}, b"plain")

    server = WireServer(handler, host="127.0.0.1", port=0)
    server.start()
    try:
        assert server.lifespan is None
        with _connect(server) as sock:
            _get(sock)
    finally:
        server.stop()


# ── real-app interop (the phase-3 acceptance) ─────────────────────────


def test_fastapi_lifespan_interop():
    pytest.importorskip("fastapi")
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    state = {"phase": "cold"}

    @asynccontextmanager
    async def lifespan(app):
        state["phase"] = "up"
        yield
        state["phase"] = "down"

    app = FastAPI(lifespan=lifespan)

    @app.get("/ping")
    def ping():
        return {"phase": state["phase"]}

    server = WireServer(asgi_app=app, host="127.0.0.1", port=0, lifespan="on", lifespan_timeout=5.0)
    server.start()
    try:
        with _connect(server) as sock:
            body = _get(sock)
        assert json.loads(body) == {"phase": "up"}
    finally:
        server.stop()
    assert state["phase"] == "down"
