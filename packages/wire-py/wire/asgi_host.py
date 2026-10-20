"""ASGI host adapter — the one place asyncio lives in this package.

Sync core, async seam (AGENTS execution philosophy): ``WireServer`` stays
synchronous and hands one parsed request over; the ASGI app runs in an
event loop created and destroyed inside :func:`run_http`, and every
response byte streams straight back through the sync socket write as the
app produces it — which is exactly why SSE works without buffering the
whole body.

Framing decision on the FIRST response body frame:
- app set ``Content-Length``  -> pass through (single frame only);
- app set ``Transfer-Encoding`` -> honor app chunking;
- ``more_body`` on the first frame -> wire switches to chunked (a
  streaming response has no knowable length up front — buffering it to
  compute one would defeat the point);
- otherwise -> ``Content-Length`` from the frame.

WebSocket (phase 3): :func:`run_websocket` serves the upgrade — handshake
validation/101 live in :mod:`wire.websocket` (sync), then one session loop
owns the connection for its lifetime. Frame reads run in a dedicated
single-worker reader executor (shielded so a cancelled ``receive`` never
spawns a second reader on the buffered stream, and torn down without
waiting so a stranded read cannot deadlock session end) — which is what
lets an app send while a receive is pending. Writes go straight to the
sync socket from the loop thread, same as :func:`run_http`.

Lifespan (phase 3): :class:`Lifespan` drives the ASGI lifespan protocol on a
dedicated long-lived loop thread — the spec requires ONE startup handshake
before the first request may be served and ONE shutdown handshake after the
last, while each http cycle keeps its own short-lived loop. Loop-affinity
constraint (documented, not silent): lifespan handlers must create
loop-AGNOSTIC resources — sync clients, engine handles, threads — because
request cycles run on other loops; an async client bound to the driver loop
cannot be used from a request. This repo's execution philosophy (sync core,
engines behind ThreadPoolExecutors, PGQ event-loop-free) makes its FastAPI
lifespan loop-agnostic in practice.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import unquote

from .framing import Request, build_response, serialize_head
from .websocket import (
    CLOSE_TOO_BIG,
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    WsProtocolError,
    encode_frame,
    handshake_problem,
)
from .websocket import (
    close_payload as _ws_close_payload,
)
from .websocket import (
    read_frame as _ws_read_frame,
)
from .websocket import (
    reject_response as _ws_reject_response,
)
from .websocket import (
    switch_response as _ws_switch_response,
)

# statuses whose responses carry no body — no Content-Length may be sent
_BODILESS = frozenset({204, 304})

SendBytes = Callable[[bytes], None]

# uvicorn-parity access line (same logger name in server.py handler mode;
# logging by name keeps the two modules import-independent)
access_logger = logging.getLogger("wire.access")

# proxy headers (X-Forwarded-For / -Proto) are honoured only from trusted
# direct peers — uvicorn's forwarded_allow_ips default
DEFAULT_TRUSTED_PROXIES = frozenset({"127.0.0.1"})


def _apply_forwarded(
    request: Request,
    client: tuple[str, int],
    trusted: frozenset[str],
    base_scheme: str,
) -> tuple[tuple[str, int], str]:
    """uvicorn-parity proxy-header translation for trusted direct peers.

    ``X-Forwarded-For`` (leftmost = original client) replaces the peer
    address, ``X-Forwarded-Proto: https`` upgrades the scheme
    (``http``→``https``, ``ws``→``wss``). From an untrusted peer — or
    when no header is present — the scope keeps the real TCP identity:
    spoofed headers must not reshape it.
    """
    if not trusted or client[0] not in trusted:
        return client, base_scheme
    xff = request.header("x-forwarded-for")
    if xff:
        host = xff.split(",")[0].strip()
        if host:
            client = (host, client[1])
    if (request.header("x-forwarded-proto") or "").strip().lower() == "https":
        base_scheme = {"http": "https", "ws": "wss"}.get(base_scheme, base_scheme)
    return client, base_scheme


def _build_scope(
    request: Request,
    client: tuple[str, int],
    server: tuple[str, int],
    trusted: frozenset[str] = DEFAULT_TRUSTED_PROXIES,
) -> dict:
    target = request.target
    path, _, query = target.partition("?")
    client, scheme = _apply_forwarded(request, client, trusted, "http")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": request.version,
        "method": request.method,
        "scheme": scheme,
        "path": unquote(path),
        "raw_path": path.encode("ascii"),
        "query_string": query.encode("ascii"),
        "root_path": "",
        "headers": [
            (key.lower().encode("ascii"), value.encode("latin-1")) for key, value in request.headers
        ],
        "client": client,
        "server": server,
    }


def _chunk(data: bytes) -> bytes:
    if not data:
        return b""
    return f"{len(data):X}\r\n".encode() + data + b"\r\n"


def run_http(
    app: Callable,
    request: Request,
    on_bytes: SendBytes,
    *,
    client: tuple[str, int],
    server: tuple[str, int],
    forwarded_allow_ips: frozenset[str] = DEFAULT_TRUSTED_PROXIES,
    access_log: bool = False,
) -> bool:
    """Run one ASGI http cycle; return whether the connection stays alive.

    Never raises for application defects: a crash before any bytes hit the
    wire becomes a 500 + close, a crash after (the response is already
    partially written) just closes — a desynced stream can't be repaired
    with a late status line. Transport failures close the connection the
    same way. With ``access_log``, one uvicorn-style line per completed
    response goes to the ``wire.access`` logger (raw TCP peer, not the
    forwarded identity).
    """
    scope = _build_scope(request, client, server, forwarded_allow_ips)
    state: dict = {"head_written": False, "keep": False, "framing": None}

    def _log_access(status: int) -> None:
        if access_log:
            access_logger.info(
                '%s:%d - "%s %s HTTP/%s" %d',
                client[0],
                client[1],
                request.method,
                request.target,
                request.version,
                status,
            )

    def _send_500() -> None:
        try:
            on_bytes(
                build_response(
                    500,
                    {"Content-Type": "text/plain"},
                    b"internal server error\n",
                    keep_alive=False,
                )
            )
        except OSError:
            pass
        state["head_written"] = True  # nothing further may be written

    async def cycle() -> bool:
        disconnect = asyncio.Event()  # never set in v1 (no read-polling yet)
        body_sent = False

        async def receive() -> dict:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": request.body, "more_body": False}
            # after the body the only message left is disconnect; without a
            # read-poller nothing ever sets it, so this parks the caller
            # (same shape as a client that simply hasn't hung up yet)
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            kind = message["type"]
            if kind == "http.response.start":
                if state["head_written"]:
                    raise RuntimeError("response already started")
                status = message["status"]
                if not 100 <= status <= 599:
                    raise RuntimeError(f"bad status from app: {status}")
                app_headers = [
                    (k.decode("latin-1"), v.decode("latin-1")) for k, v in message["headers"]
                ]
                state["status"] = status
                state["headers"] = app_headers
                state["app_lower"] = {k.lower() for k, _ in app_headers}
                conn_tokens = {
                    part.strip().lower()
                    for key, value in app_headers
                    if key.lower() == "connection"
                    for part in value.split(",")
                }
                state["keep"] = request.keep_alive and "close" not in conn_tokens
                return

            if kind != "http.response.body":
                raise RuntimeError(f"unsupported ASGI message {kind!r}")
            if "status" not in state:
                raise RuntimeError("response body before response start")

            body = message.get("body", b"") or b""
            more = bool(message.get("more_body", False))
            status = state["status"]

            if not state["head_written"]:
                app_lower = state["app_lower"]
                if status in _BODILESS or status < 200:
                    if body or more:
                        raise RuntimeError("body frame on a bodiless status")
                    on_bytes(serialize_head(status, state["headers"], keep_alive=state["keep"]))
                    state["head_written"] = True
                    state["framing"] = "none"
                    return
                if "content-length" in app_lower:
                    if more:
                        raise RuntimeError("streaming body with app Content-Length")
                    on_bytes(
                        serialize_head(status, state["headers"], keep_alive=state["keep"]) + body
                    )
                    state["head_written"] = True
                    state["framing"] = "cl"
                    return
                if "transfer-encoding" in app_lower:
                    on_bytes(serialize_head(status, state["headers"], keep_alive=state["keep"]))
                    state["head_written"] = True
                    state["framing"] = "chunked"
                    on_bytes(_chunk(body))
                    if not more:
                        on_bytes(b"0\r\n\r\n")
                    return
                if more:
                    # streaming: no knowable length -> wire-owned chunking
                    on_bytes(
                        serialize_head(
                            status,
                            [*state["headers"], ("Transfer-Encoding", "chunked")],
                            keep_alive=state["keep"],
                        )
                    )
                    state["head_written"] = True
                    state["framing"] = "chunked"
                    on_bytes(_chunk(body))
                    return
                # single frame: Content-Length from the frame itself
                on_bytes(
                    serialize_head(
                        status,
                        [*state["headers"], ("Content-Length", str(len(body)))],
                        keep_alive=state["keep"],
                    )
                    + body
                )
                state["head_written"] = True
                state["framing"] = "cl"
                return

            # head already written
            if state["framing"] == "chunked":
                on_bytes(_chunk(body))
                if not more:
                    on_bytes(b"0\r\n\r\n")
                    state["framing"] = "closed"
                return
            raise RuntimeError(f"extra body frame after {state['framing']} framing")

        try:
            await app(scope, receive, send)
        except Exception:  # noqa: BLE001 - ASGI app is an untrusted boundary
            if not state["head_written"]:
                _send_500()
                _log_access(500)
            else:
                _log_access(int(state.get("status", 500)))
            return False
        if not state["head_written"]:
            # app returned without ever starting a response
            _send_500()
            _log_access(500)
            return False
        if state["framing"] == "chunked":
            # app streamed without a final frame — close the framing cleanly
            on_bytes(b"0\r\n\r\n")
        _log_access(int(state["status"]))
        return bool(state["keep"])

    try:
        return asyncio.run(cycle())
    except OSError:
        # transport died (peer reset during send)
        return False


def _build_ws_scope(
    request: Request,
    client: tuple[str, int],
    server: tuple[str, int],
    trusted: frozenset[str] = DEFAULT_TRUSTED_PROXIES,
) -> dict:
    target = request.target
    path, _, query = target.partition("?")
    client, scheme = _apply_forwarded(request, client, trusted, "ws")
    subprotocols: list[str] = []
    if proto := request.header("sec-websocket-protocol"):
        subprotocols = [p.strip() for p in proto.split(",") if p.strip()]
    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": request.version,
        "scheme": scheme,
        "path": unquote(path),
        "raw_path": path.encode("ascii"),
        "query_string": query.encode("ascii"),
        "root_path": "",
        "headers": [
            (key.lower().encode("ascii"), value.encode("latin-1")) for key, value in request.headers
        ],
        "client": client,
        "server": server,
        "subprotocols": subprotocols,
    }


def run_websocket(
    app: Callable,
    request: Request,
    reader,  # buffered binary reader for the connection (makefile)
    on_bytes: SendBytes,
    *,
    client: tuple[str, int],
    server: tuple[str, int],
    forwarded_allow_ips: frozenset[str] = DEFAULT_TRUSTED_PROXIES,
    access_log: bool = False,
    ws_max_size: int = 16 * 1024 * 1024,
) -> None:
    """Serve one WebSocket upgrade; blocks until the session ends.

    Handshake problems answer as HTTP (400 / 426 + version header) before
    the app is called; the app rejecting pre-accept answers 403. The session
    then runs on ONE event loop for the connection's lifetime — frame reads
    off-loop in a dedicated single-worker reader executor (a cancelled
    ``receive`` shares the in-flight read, never strands a second reader),
    writes direct to the sync socket from the loop thread, exactly like
    ``run_http``. ``ws_max_size`` bounds every message (frames and
    assembled fragments) — over-cap closes with 1009.
    """
    problem = handshake_problem(request)
    if problem is not None:
        on_bytes(_ws_reject_response(problem))
        return
    # single-worker reader executor owned by this session: the default
    # executor is joined by asyncio.run at teardown, so a cancelled receive
    # leaving its read in flight would deadlock session end (the connection
    # close that would wake the reader only happens after we return).
    read_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wire-ws-read")
    try:
        asyncio.run(
            _ws_session(
                app,
                request,
                reader,
                on_bytes,
                client=client,
                server=server,
                read_exec=read_exec,
                forwarded_allow_ips=forwarded_allow_ips,
                access_log=access_log,
                max_payload=ws_max_size,
            )
        )
    except OSError:
        return  # peer vanished mid-frame/write; teardown is the caller's
    finally:
        # never wait: an in-flight read wakes with EOF when _serve_conn
        # closes the connection right after this returns
        read_exec.shutdown(wait=False, cancel_futures=True)


async def _ws_session(
    app: Callable,
    request: Request,
    reader,
    on_bytes: SendBytes,
    *,
    client: tuple[str, int],
    server: tuple[str, int],
    read_exec: ThreadPoolExecutor,
    forwarded_allow_ips: frozenset[str],
    access_log: bool,
    max_payload: int,
) -> None:
    loop = asyncio.get_running_loop()
    scope = _build_ws_scope(request, client, server, forwarded_allow_ips)

    accepted = False
    closed = False  # close frame (server- or client-initiated) is settled
    settled: dict | None = None  # disconnect message once the session ends
    receiving = False
    read_future: asyncio.Future | None = None
    asm = bytearray()
    asm_opcode = 0

    def _settle(code: int) -> dict:
        nonlocal closed, settled
        closed = True
        settled = {"type": "websocket.disconnect", "code": code}
        return settled

    async def _fail(code: int) -> dict:
        """Protocol-level close: answer with a close frame, then settle."""
        if accepted and not closed:
            try:
                on_bytes(encode_frame(OP_CLOSE, _ws_close_payload(code)))
            except OSError:
                pass
        return _settle(code)

    async def _next_frame():
        nonlocal read_future
        if read_future is None:
            read_future = loop.run_in_executor(
                read_exec, partial(_ws_read_frame, reader, max_payload=max_payload)
            )
        fut = read_future
        try:
            # shield: a cancelled receive() must not cancel the in-flight
            # read (that would let a later receive spawn a SECOND reader
            # on the same buffered stream)
            return await asyncio.shield(fut)
        finally:
            if fut.done() and read_future is fut:
                read_future = None

    async def receive() -> dict:
        nonlocal receiving, asm, asm_opcode
        if not accepted:
            return {"type": "websocket.connect"}
        if settled is not None:
            return settled
        if receiving:
            raise RuntimeError("concurrent websocket.receive is not allowed")
        receiving = True
        try:
            while True:
                try:
                    frame = await _next_frame()
                except WsProtocolError as exc:
                    return await _fail(exc.code)  # 1002 protocol, 1009 cap
                except (ConnectionError, OSError):
                    return _settle(1006)  # abnormal close: no close frame
                if frame is None:
                    return _settle(1006)  # clean EOF without a close frame
                if frame.opcode == OP_PING:
                    on_bytes(encode_frame(OP_PONG, frame.payload))  # server duty
                    continue
                if frame.opcode == OP_PONG:
                    continue
                if frame.opcode == OP_CLOSE:
                    code = (
                        int.from_bytes(frame.payload[:2], "big")
                        if len(frame.payload) >= 2
                        else 1005
                    )
                    if accepted and not closed:
                        try:
                            on_bytes(encode_frame(OP_CLOSE, frame.payload))  # echo §5.5.1
                        except OSError:
                            pass
                    return _settle(code)
                if frame.opcode in (OP_TEXT, OP_BINARY):
                    if asm_opcode:
                        return await _fail(1002)  # data frame mid-fragmentation
                    if frame.fin:
                        return await _deliver(frame.opcode, bytes(frame.payload))
                    asm = bytearray(frame.payload)
                    asm_opcode = frame.opcode
                    continue
                # OP_CONT
                if asm_opcode == 0:
                    return await _fail(1002)  # continuation without a start
                if len(asm) + len(frame.payload) > max_payload:
                    return await _fail(CLOSE_TOO_BIG)  # assembled > cap
                asm.extend(frame.payload)
                if frame.fin:
                    message = await _deliver(asm_opcode, bytes(asm))
                    asm = bytearray()
                    asm_opcode = 0
                    return message
        finally:
            receiving = False

    async def _deliver(opcode: int, payload: bytes) -> dict:
        if opcode == OP_TEXT:
            try:
                return {"type": "websocket.receive", "text": payload.decode("utf-8")}
            except UnicodeDecodeError:
                return await _fail(1007)  # invalid UTF-8 text (RFC6455 §8.1)
        return {"type": "websocket.receive", "bytes": payload}

    async def send(message: dict) -> None:
        nonlocal accepted, closed, settled
        kind = message.get("type")
        if kind == "websocket.accept":
            if accepted or settled is not None:
                raise RuntimeError("websocket.accept after accept/close")
            on_bytes(_ws_switch_response(request, message.get("subprotocol")))
            accepted = True
            if access_log:
                access_logger.info(
                    '%s:%d - "%s %s HTTP/%s" 101',
                    client[0],
                    client[1],
                    request.method,
                    request.target,
                    request.version,
                )
        elif kind == "websocket.send":
            if not accepted:
                raise RuntimeError("websocket.accept must precede sends")
            if closed:
                raise RuntimeError("websocket.send after close")
            if "text" in message:
                on_bytes(encode_frame(OP_TEXT, message["text"].encode()))
            elif "bytes" in message:
                on_bytes(encode_frame(OP_BINARY, bytes(message["bytes"])))
            else:
                raise RuntimeError("websocket.send requires 'text' or 'bytes'")
        elif kind == "websocket.close":
            if not accepted:
                # app rejected the connection → HTTP 403, never a frame
                if settled is None:
                    on_bytes(_ws_reject_response((403, "WebSocket upgrade denied", {})))
                    settled = {"type": "websocket.disconnect", "code": 1008}
                    closed = True
                return
            if not closed:
                code = int(message.get("code", 1000))
                on_bytes(encode_frame(OP_CLOSE, _ws_close_payload(code)))
                _settle(code)
        else:
            raise RuntimeError(f"unexpected websocket send type: {kind!r}")

    try:
        await app(scope, receive, send)
    except Exception:  # noqa: BLE001 - ASGI app is an untrusted boundary
        pass  # no HTTP response exists post-upgrade; teardown closes TCP
    finally:
        if accepted and not closed:
            # app returned without close — complete the handshake anyway
            try:
                on_bytes(encode_frame(OP_CLOSE, _ws_close_payload(1000)))
            except OSError:
                pass
        elif not accepted and settled is None:
            try:
                on_bytes(_ws_reject_response((403, "WebSocket upgrade denied", {})))
            except OSError:
                pass


class LifespanError(RuntimeError):
    """Lifespan handshake failed (app reported failure, crashed, or timed out)."""


class Lifespan:
    """Drives one ASGI app's lifespan protocol on a dedicated loop thread.

    Exactly one driver per server lifetime, matching the spec: the app gets a
    ``lifespan`` scope, must complete ``startup`` before any http request may
    be served, and must complete ``shutdown`` after the last one. The app's
    single ``__call__`` task stays alive across both handshakes (it parks in
    ``receive()`` between them), so startup handlers see the same objects
    shutdown handlers do.

    Failure paths all leave no leaked thread: ``startup()`` raises
    :class:`LifespanError` (text includes the app's ``message`` or the task's
    exception) after unwinding the loop; ``shutdown()`` raises after unwinding
    so the caller's teardown always completes.
    """

    def __init__(self, app: Callable) -> None:
        self._app = app
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._inbox: deque[dict] = deque()
        self._lock = threading.Lock()
        self._ready = threading.Event()  # driver-side handshake wait
        self._error: str | None = None  # ``.failed`` message, set before ready
        self._got: dict[str, bool] = {"startup": False, "shutdown": False}
        self._scope: dict = {}

    @property
    def thread(self) -> threading.Thread | None:
        """The driver loop thread, while a handshake is live (None otherwise)."""
        return self._thread

    # ── driver side (any thread) ───────────────────────────────────────

    def startup(self, timeout: float = 30.0) -> None:
        """Run the startup handshake; raise LifespanError on any failure."""
        if self._thread is not None:
            raise LifespanError("lifespan already started")
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error = None
        self._got = {"startup": False, "shutdown": False}
        self._scope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "state": {},
        }
        self._thread = threading.Thread(target=self._run, name="wire-lifespan", daemon=True)
        self._thread.start()
        self._post({"type": "lifespan.startup"})
        if not self._ready.wait(timeout):
            self._stop_loop()
            raise LifespanError(f"timeout ({timeout}s) waiting for lifespan.startup")
        if (error := self._handshake_error("startup")) is not None:
            self._stop_loop()
            raise LifespanError(error)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Run the shutdown handshake; a no-op if never started or already done."""
        if self._thread is None:
            return  # never started / already shut down — idempotent
        self._ready.clear()
        self._error = None
        error: str | None
        if self._task is not None and self._task.done():
            # nothing left to consume the message — classify the dead task
            error = self._task_exit_error("shutdown")
        else:
            self._post({"type": "lifespan.shutdown"})
            if not self._ready.wait(timeout):
                error = f"timeout ({timeout}s) waiting for lifespan.shutdown"
            else:
                error = self._handshake_error("shutdown")
        self._stop_loop()
        if error is not None:
            raise LifespanError(error)

    def _handshake_error(self, phase: str) -> str | None:
        """Classify the just-woken handshake: None = that phase completed."""
        if self._error is not None:
            return self._error
        if self._got[phase]:
            return None
        return self._task_exit_error(phase)

    def _task_exit_error(self, phase: str) -> str:
        """Why the app task can no longer complete this phase."""
        if self._task is None:
            return f"app exited without completing lifespan.{phase}"
        if not self._task.done():
            return f"app did not complete lifespan.{phase}"
        if not self._task.cancelled():
            exc = self._task.exception()
            if exc is not None:
                return f"{type(exc).__name__}: {exc}"
        return f"app exited without completing lifespan.{phase}"

    def _post(self, message: dict) -> None:
        """Queue an inbox message and wake the app's ``receive()``."""
        with self._lock:
            self._inbox.append(message)
        assert self._loop is not None
        # callback looks up self._wakeup at run time — _run creates it before
        # run_forever starts processing callbacks
        self._loop.call_soon_threadsafe(self._wakeup_set)

    def _wakeup_set(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

    def _stop_loop(self) -> None:
        """Stop the driver loop and join its thread (unwinds the app task)."""
        thread, self._thread = self._thread, None
        if thread is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass  # loop already closed
        thread.join(timeout=5.0)

    # ── app side (driver loop thread) ──────────────────────────────────

    def _run(self) -> None:
        loop = self._loop
        assert loop is not None
        asyncio.set_event_loop(loop)
        try:
            self._wakeup = asyncio.Event()
            self._task = loop.create_task(self._app(self._scope, self._receive, self._send))
            self._task.add_done_callback(self._task_done)
            loop.run_forever()
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _task_done(self, task: asyncio.Task) -> None:
        # The handshake can no longer proceed — wake the driver either way;
        # it classifies the dead task (crash vs. clean early exit).
        del task
        self._ready.set()

    async def _receive(self) -> dict:
        while True:
            with self._lock:
                if self._inbox:
                    return self._inbox.popleft()
            assert self._wakeup is not None
            self._wakeup.clear()
            with self._lock:
                # posted between the first check and clear — still valid
                if self._inbox:
                    return self._inbox.popleft()
            await self._wakeup.wait()

    async def _send(self, message: dict) -> None:
        kind = str(message.get("type", ""))
        if kind == "lifespan.startup.complete":
            self._got["startup"] = True
            self._ready.set()
        elif kind == "lifespan.shutdown.complete":
            self._got["shutdown"] = True
            self._ready.set()
        elif kind in ("lifespan.startup.failed", "lifespan.shutdown.failed"):
            self._error = str(message.get("message") or kind)
            self._ready.set()
        # other message types are not part of the spec — ignored
