"""WireServer — stdlib sync TCP accept loop for HTTP/1.1 (keep-alive aware).

Threading, not asyncio: the wire core stays synchronous by design (AGENTS
execution philosophy — the async seam lives above it, in the phase-2 ASGI
host adapter, never in this layer). One thread per connection, an accept
thread with a polling timeout so ``stop()`` is prompt and deterministic.

Lifespan (phase 3): ASGI mode runs the lifespan handshake around the accept
loop — ``lifespan="on"`` (strict: startup failure fails start),
``"auto"`` (uvicorn-compatible: log and serve anyway — the default, so
lifespan-unaware apps keep working), ``"off"`` (skip). Handler mode never
constructs a driver.

WebSocket (phase 3): an ``Upgrade: websocket`` request in ASGI mode is
handed to ``run_websocket`` (see :mod:`wire.asgi_host`) instead of the
HTTP loop — the session consumes the connection (handshake, frames, close),
so no keep-alive follows it.

Connection lifecycle:
- clean EOF between requests ends the connection (no error);
- framing errors are answered with their status (400/431) + ``Connection:
  close`` — a connection that desynchronized is never reused;
- handler exceptions become 500 + close, never a dead thread or a hung
  socket.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .asgi_host import Lifespan, LifespanError, run_http, run_websocket
from .framing import DEFAULT_MAX_BODY, Request, WireError, build_response, read_request
from .websocket import DEFAULT_MAX_PAYLOAD, is_ws_upgrade

logger = logging.getLogger(__name__)
# uvicorn-parity access line (same logger name as asgi_host's ASGI mode;
# logging by name keeps the two modules import-independent)
access_logger = logging.getLogger("wire.access")

LifespanMode = Literal["auto", "on", "off"]


@dataclass
class Response:
    """Handler result: status + headers + body, framing decided by the wire."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    chunked: bool = False
    keep_alive: bool | None = None  # None = follow the request's persistence


Handler = Callable[[Request], Response]


class WireServer:
    """Sync TCP server: bind, accept, serve requests until stopped."""

    def __init__(
        self,
        handler: Handler | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        backlog: int = 64,
        *,
        asgi_app: Callable | None = None,
        lifespan: LifespanMode = "auto",
        lifespan_timeout: float = 30.0,
        keep_alive_timeout: float | None = 5.0,
        max_body: int = DEFAULT_MAX_BODY,
        ws_max_size: int = DEFAULT_MAX_PAYLOAD,
        forwarded_allow_ips: str | Sequence[str] | None = "127.0.0.1",
        access_log: bool = True,
    ) -> None:
        if (handler is None) == (asgi_app is None):
            raise ValueError("exactly one of 'handler' or 'asgi_app' is required")
        if lifespan not in ("auto", "on", "off"):
            raise ValueError(f"lifespan must be 'auto', 'on', or 'off', got {lifespan!r}")
        self._handler = handler
        self._asgi_app = asgi_app
        self._host = host
        self._port = port
        self._backlog = backlog
        self._lifespan_mode: LifespanMode = lifespan
        self._lifespan_timeout = lifespan_timeout
        self._keep_alive_timeout = keep_alive_timeout
        self._max_body = max_body
        self._ws_max_size = ws_max_size
        if forwarded_allow_ips is None:
            forwarded_allow_ips = "127.0.0.1"  # uvicorn parity: loopback default
        if isinstance(forwarded_allow_ips, str):
            ips = [part.strip() for part in forwarded_allow_ips.split(",")]
        else:
            ips = [str(ip).strip() for ip in forwarded_allow_ips]
        self._forwarded_allow: frozenset[str] = frozenset(ip for ip in ips if ip)
        self._access_log = access_log
        self._lifespan_driver: Lifespan | None = None
        self._listener: socket.socket | None = None
        self._bound: tuple[str, int] = (host, port)
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────

    @property
    def asgi_app(self) -> Callable | None:
        """The ASGI app this server drives, if constructed in ASGI mode."""
        return self._asgi_app

    @property
    def lifespan(self) -> Lifespan | None:
        """The live lifespan driver (None in handler mode, lifespan off, or
        after a startup failure in auto mode)."""
        return self._lifespan_driver

    @property
    def port(self) -> int:
        """Bound port (resolved after ``start()`` — supports port 0)."""
        if self._listener is None:
            raise RuntimeError("server not started")
        return int(self._listener.getsockname()[1])

    def start(self) -> None:
        """Bind, run the lifespan startup handshake, spawn the accept thread."""
        if self._listener is not None:
            raise RuntimeError("server already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(self._backlog)
        listener.settimeout(0.2)  # stop() responsiveness
        self._listener = listener
        self._bound = listener.getsockname()[:2]
        self._stop.clear()
        if self._asgi_app is not None and self._lifespan_mode != "off":
            # uvicorn's order: bind first (port reserved), handshake second,
            # accept thread last — no request can beat the startup handshake
            driver = Lifespan(self._asgi_app)
            try:
                driver.startup(timeout=self._lifespan_timeout)
            except LifespanError as exc:
                if self._lifespan_mode == "on":
                    self._listener = None
                    try:
                        listener.close()
                    except OSError:
                        pass
                    raise
                logger.warning("lifespan startup failed — serving without it: %s", exc)
            else:
                self._lifespan_driver = driver
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="wire-accept", daemon=True
        )
        self._accept_thread.start()

    def serve_forever(self) -> None:
        """Blocking convenience wrapper: run the accept loop until stop()."""
        if self._listener is None:
            self.start()
        assert self._accept_thread is not None
        self._accept_thread.join()

    def stop(self) -> None:
        """Close the listener and every open client connection."""
        self._stop.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        driver, self._lifespan_driver = self._lifespan_driver, None
        if driver is not None:
            try:
                driver.shutdown(timeout=self._lifespan_timeout)
            except LifespanError as exc:
                # teardown must always complete — a failed shutdown handshake
                # is logged, never raised out of stop()
                logger.warning("lifespan shutdown failed: %s", exc)

    # ── accept / serve ─────────────────────────────────────────────────

    def _accept_loop(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                conn, _addr = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break  # listener closed by stop()
            with self._lock:
                if self._stop.is_set():
                    conn.close()
                    break
                self._clients.add(conn)
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _log_access(self, peer, request: Request, status: int) -> None:
        # uvicorn-parity access line: raw TCP peer (not the forwarded
        # identity — that's for the app, not the log)
        if self._access_log:
            access_logger.info(
                '%s:%d - "%s %s HTTP/%s" %d',
                peer[0],
                peer[1],
                request.method,
                request.target,
                request.version,
                status,
            )

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            reader = conn.makefile("rb")
            peer = conn.getpeername()[:2]
        except OSError:
            return  # connection already torn down by stop()

        def _expect_continue() -> None:
            # bare interim response (RFC 9110 §10.1.1) — fired only after
            # the framing and the body cap validated (read_request's word)
            conn.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")

        try:
            while True:
                if self._keep_alive_timeout is not None:
                    # uvicorn timeout_keep_alive parity: the READ phase owns
                    # the idle timer (first byte included — a slowloris peer
                    # holds a thread for at most this long); buffered bytes
                    # cost no syscall, so pipelined requests never false-trip
                    conn.settimeout(self._keep_alive_timeout)
                try:
                    request = read_request(
                        reader, max_body=self._max_body, on_expect=_expect_continue
                    )
                except WireError as exc:
                    # desync: answer the status, then close — never reuse
                    conn.sendall(
                        build_response(
                            exc.status,
                            {},
                            f"{exc.message}\n".encode(),
                            keep_alive=False,
                        )
                    )
                    return
                finally:
                    # clear the timer before any response write — SSE
                    # streams and slow readers run unbounded, like uvicorn
                    if self._keep_alive_timeout is not None:
                        conn.settimeout(None)
                if request is None:  # clean EOF between requests
                    return
                if self._asgi_app is not None:
                    if is_ws_upgrade(request):
                        # HTTP→WS upgrade: the session consumes the
                        # connection (handshake, frames, close) — no
                        # keep-alive loop after it
                        run_websocket(
                            self._asgi_app,
                            request,
                            reader,
                            conn.sendall,
                            client=peer,
                            server=self._bound,
                            forwarded_allow_ips=self._forwarded_allow,
                            access_log=self._access_log,
                            ws_max_size=self._ws_max_size,
                        )
                        return
                    # ASGI mode: the adapter owns response bytes and the
                    # keep-alive decision (app Connection header wins)
                    keep = run_http(
                        self._asgi_app,
                        request,
                        conn.sendall,
                        client=peer,
                        server=self._bound,
                        forwarded_allow_ips=self._forwarded_allow,
                        access_log=self._access_log,
                    )
                    if not keep:
                        return
                    continue
                try:
                    response = self._handler(request)
                except Exception:  # noqa: BLE001 - boundary: handler crash -> 500
                    conn.sendall(
                        build_response(
                            500,
                            {"Content-Type": "text/plain"},
                            b"internal server error\n",
                            keep_alive=False,
                        )
                    )
                    self._log_access(peer, request, 500)
                    return
                keep = request.keep_alive if response.keep_alive is None else response.keep_alive
                conn.sendall(
                    build_response(
                        response.status,
                        response.headers,
                        response.body,
                        keep_alive=keep,
                        chunked=response.chunked,
                    )
                )
                self._log_access(peer, request, response.status)
                if not keep:
                    return
        except OSError:
            # peer vanished mid-write — and the idle timer firing mid-read
            # (a silent keep-alive client) lands here too: clean close
            return
        finally:
            with self._lock:
                self._clients.discard(conn)
            # wake any read still blocked on this socket first: the ws
            # session's reader thread holds the makefile lock inside a
            # blocking read, and reader.close() waits on that same lock —
            # close() alone does not interrupt a blocked recv (only
            # shutdown does), so closing the reader first deadlocks.
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                reader.close()
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
