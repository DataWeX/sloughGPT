"""RFC6455 WebSocket — handshake validation + frame codec (sync, stdlib).

The transport half of minimal WS for card ecea49c8 (flip gate): this module
never imports asyncio — it validates the HTTP upgrade, computes the
``Sec-WebSocket-Accept`` value, encodes server frames (unmasked), and reads
client frames (MASKED — RFC6455 §5.1 makes masking mandatory for
client→server). The async session that drives the ASGI app over these bytes
lives in :mod:`wire.asgi_host` (the one place asyncio lives in this
package).

Protocol notes (minimal scope, spec-shaped):
- all three payload-length forms are supported (7-bit, 16-bit, 64-bit);
- fragmentation is assembled across continuation frames;
- ping is answered automatically with a pong of the same payload and never
  surfaces to the app (server-level duty, like uvicorn);
- a client frame arriving with the mask bit clear is a protocol error →
  close 1002; a bad handshake is an HTTP error (400, or 426 +
  ``Sec-WebSocket-Version: 13`` on a version mismatch — RFC6455 §9).
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

from .framing import Request, build_response

WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

_CLOSE_NORMAL = 1000
_CLOSE_PROTOCOL = 1002


class WsProtocolError(Exception):
    """A frame violated RFC6455 (unmasked client frame, bad opcode, …)."""


def is_ws_upgrade(request: Request) -> bool:
    """True when this request is an HTTP→WebSocket upgrade attempt."""
    upgrade = request.header("upgrade")
    if upgrade is None or upgrade.lower() != "websocket":
        return False
    connection = request.header("connection") or ""
    return any(tok.strip().lower() == "upgrade" for tok in connection.split(","))


def handshake_problem(request: Request) -> tuple[int, str, dict[str, str]] | None:
    """Validate the upgrade request; None = valid.

    Returns ``(status, message, extra_headers)`` on failure: 400 for a
    malformed handshake, 426 (+ required version header) for an unsupported
    ``Sec-WebSocket-Version``.
    """
    if request.method != "GET":
        return 400, "websocket upgrade must be GET", {}
    if not is_ws_upgrade(request):
        return 400, "missing Upgrade: websocket headers", {}
    version = request.header("sec-websocket-version")
    if version != "13":
        return 426, "unsupported Sec-WebSocket-Version", {"Sec-WebSocket-Version": "13"}
    key = request.header("sec-websocket-key")
    if key is None or _key_digest(key) is None:
        return 400, "missing or invalid Sec-WebSocket-Key", {}
    return None


def _key_digest(key: str) -> bytes | None:
    """sha1 of key+GUID (the accept digest), or None for an invalid key."""
    try:
        raw = base64.b64decode(key, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) != 16:
        return None
    return hashlib.sha1(raw + WS_GUID).digest()


def accept_value(key: str) -> str:
    """``Sec-WebSocket-Accept`` for a validated handshake key."""
    digest = _key_digest(key)
    if digest is None:
        raise ValueError(f"invalid Sec-WebSocket-Key: {key!r}")
    return base64.b64encode(digest).decode()


def switch_response(request: Request, subprotocol: str | None = None) -> bytes:
    """The 101 Switching Protocols reply (key is validated by then)."""
    key = request.header("sec-websocket-key")
    assert key is not None  # handshake_problem() gates callers
    lines = [
        "HTTP/1.1 101 Switching Protocols",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Accept: {accept_value(key)}",
    ]
    if subprotocol:
        lines.append(f"Sec-WebSocket-Protocol: {subprotocol}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def reject_response(problem: tuple[int, str, dict[str, str]]) -> bytes:
    """HTTP error reply for a failed handshake or an app-level reject."""
    status, message, extra = problem
    return build_response(
        status,
        {"Content-Type": "text/plain", **extra},
        f"{message}\n".encode(),
        keep_alive=False,
    )


def encode_frame(opcode: int, payload: bytes = b"", fin: bool = True) -> bytes:
    """One server→client frame (never masked)."""
    head = bytes([(0x80 if fin else 0) | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + n.to_bytes(2, "big")
    else:
        head += bytes([127]) + n.to_bytes(8, "big")
    return head + payload


@dataclass(frozen=True)
class Frame:
    """One parsed client frame (payload already unmasked)."""

    fin: bool
    opcode: int
    payload: bytes


def read_frame(reader) -> Frame | None:
    """Read one client frame from a buffered binary reader.

    Returns ``None`` only on a clean EOF at a frame boundary (the client
    went away between messages); a frame cut short mid-payload raises
    :class:`ConnectionError`. Protocol violations raise
    :class:`WsProtocolError`.
    """

    def _take(n: int) -> bytes:
        out = reader.read(n)
        if len(out) != n:
            raise ConnectionError("eof mid-frame")
        return out

    first = reader.read(1)
    if not first:
        return None  # clean EOF at a frame boundary
    second = _take(1)
    b0, b1 = first[0], second[0]
    if b0 & 0x70:
        raise WsProtocolError("RSV bits set without an extension")
    if not (b1 & 0x80):
        raise WsProtocolError("client frames must be masked (RFC6455 §5.1)")
    fin = bool(b0 & 0x80)
    opcode = b0 & 0x0F
    n = b1 & 0x7F
    if n == 126:
        n = int.from_bytes(_take(2), "big")
    elif n == 127:
        n = int.from_bytes(_take(8), "big")
    if opcode not in (OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG):
        raise WsProtocolError(f"unknown opcode {opcode:#x}")
    mask = _take(4)
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(_take(n)))
    return Frame(fin=fin, opcode=opcode, payload=payload)


def close_payload(code: int) -> bytes:
    """Close-frame payload: 2-byte big-endian status code."""
    return int(code).to_bytes(2, "big")
