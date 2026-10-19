"""WireServer — stdlib sync TCP accept loop for HTTP/1.1 (keep-alive aware).

Threading, not asyncio: the wire core stays synchronous by design (AGENTS
execution philosophy — the async seam lives above it, in the phase-2 ASGI
host adapter, never in this layer). One thread per connection, an accept
thread with a polling timeout so ``stop()`` is prompt and deterministic.

Connection lifecycle:
- clean EOF between requests ends the connection (no error);
- framing errors are answered with their status (400/431) + ``Connection:
  close`` — a connection that desynchronized is never reused;
- handler exceptions become 500 + close, never a dead thread or a hung
  socket.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .framing import Request, WireError, build_response, read_request


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
        handler: Handler,
        host: str = "127.0.0.1",
        port: int = 0,
        backlog: int = 64,
    ) -> None:
        self._handler = handler
        self._host = host
        self._port = port
        self._backlog = backlog
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────

    @property
    def port(self) -> int:
        """Bound port (resolved after ``start()`` — supports port 0)."""
        if self._listener is None:
            raise RuntimeError("server not started")
        return int(self._listener.getsockname()[1])

    def start(self) -> None:
        """Bind, listen, and spawn the accept thread."""
        if self._listener is not None:
            raise RuntimeError("server already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(self._backlog)
        listener.settimeout(0.2)  # stop() responsiveness
        self._listener = listener
        self._stop.clear()
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

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            reader = conn.makefile("rb")
        except OSError:
            return  # connection already torn down by stop()
        try:
            while True:
                try:
                    request = read_request(reader)
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
                if request is None:  # clean EOF between requests
                    return
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
                if not keep:
                    return
        except OSError:
            return  # peer vanished mid-write; connection teardown
        finally:
            with self._lock:
                self._clients.discard(conn)
            try:
                reader.close()
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
