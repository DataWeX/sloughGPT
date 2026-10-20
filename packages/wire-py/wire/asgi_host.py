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

Websocket scopes are out of scope here (deferred on card ecea49c8; the
default transport flip stays gated on WS support).

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
import threading
from collections import deque
from collections.abc import Callable
from urllib.parse import unquote

from .framing import Request, build_response, serialize_head

# statuses whose responses carry no body — no Content-Length may be sent
_BODILESS = frozenset({204, 304})

SendBytes = Callable[[bytes], None]


def _build_scope(request: Request, client: tuple[str, int], server: tuple[str, int]) -> dict:
    target = request.target
    path, _, query = target.partition("?")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": request.version,
        "method": request.method,
        "scheme": "http",
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
) -> bool:
    """Run one ASGI http cycle; return whether the connection stays alive.

    Never raises for application defects: a crash before any bytes hit the
    wire becomes a 500 + close, a crash after (the response is already
    partially written) just closes — a desynced stream can't be repaired
    with a late status line. Transport failures close the connection the
    same way.
    """
    scope = _build_scope(request, client, server)
    state: dict = {"head_written": False, "keep": False, "framing": None}

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
            return False
        if not state["head_written"]:
            # app returned without ever starting a response
            _send_500()
            return False
        if state["framing"] == "chunked":
            # app streamed without a final frame — close the framing cleanly
            on_bytes(b"0\r\n\r\n")
        return bool(state["keep"])

    try:
        return asyncio.run(cycle())
    except OSError:
        # transport died (peer reset during send)
        return False


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
