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
"""

from __future__ import annotations

import asyncio
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
