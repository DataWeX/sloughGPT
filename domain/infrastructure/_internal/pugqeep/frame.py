"""
Frame protocol — the wire format of the execution pipeline.

Every message an isolated child sends back to its parent travels as an
explicit frame::

    [version u8][type u8][len u32 BE][payload x len]

This replaces the implicit ``multiprocessing.Pipe`` envelope (arbitrary
pickle) with a closed, versioned, size-bounded protocol:

  - ``version``  mismatched peers fail loudly instead of misparsing.
  - ``type``     a closed enum instead of ``isinstance(msg, tuple)`` duck-typing.
  - ``len``      validated BEFORE allocation — a length prefix is an
                 allocation primitive, so an oversized claim raises
                 ``FrameTooLarge`` rather than attempting the read.
  - ``payload``  a total codec: every input yields a value or a typed
                 ``ProtocolError``, never a silent truncation. numpy arrays
                 travel as dtype + shape + raw buffer, so results do not have
                 to be picklable.

``MAX_FRAME`` is part of the execution stack's bound: it is reported by
``Pipe.headroom()`` alongside queue capacity, because bounding the queue while
leaving the channel open would bound half the pipeline.
"""

from __future__ import annotations

import logging
import struct
from enum import IntEnum
from typing import Any, Callable

PROTOCOL_VERSION = 1

#: Largest payload this side will accept or emit, in bytes.
MAX_FRAME = 1 << 24  # 16 MiB

#: Captured stdout/stderr is split across frames of at most this many chars, so
#: a chatty child cannot produce a single frame that busts ``MAX_FRAME``.
CHUNK_CHARS = 1 << 16  # 64 Ki

logger = logging.getLogger("slo.pugqeep.frame")

_HEADER = struct.Struct(">BBI")  # version, type, payload length
assert _HEADER.size == 6

#: Default poll interval for the reader loop (mirrors the old ``conn.poll``).
POLL_INTERVAL = 0.5

#: How long each side waits for the peer's HELLO during :meth:`FrameHandler.handshake`.
HANDSHAKE_TIMEOUT = 30.0


class MsgType(IntEnum):
    """Closed set of messages the execution pipeline may carry."""

    READY = 0x01
    HEARTBEAT = 0x02
    RESULT = 0x03
    ERROR = 0x04
    STDOUT = 0x05
    STDERR = 0x06
    EXIT = 0x07
    HELLO = 0x08


# ── Value tags ──────────────────────────────────────────────────────────────
_T_NONE = 0x00
_T_BOOL = 0x01
_T_INT = 0x02
_T_FLOAT = 0x03
_T_STR = 0x04
_T_BYTES = 0x05
_T_LIST = 0x06
_T_TUPLE = 0x07
_T_DICT = 0x08
_T_NDARRAY = 0x09
_T_CUSTOM = 0x0A
_T_BIGINT = 0x0B

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: Registry tags for user types must live in this range so they can never
#: collide with the built-in tags above.
_CUSTOM_TAG_MIN = 0x40
_CUSTOM_TAG_MAX = 0xFE

#: Distinguishes "no argument" from an explicit ``None`` on :meth:`FrameHandler.send`.
_MISSING = object()


class ProtocolError(Exception):
    """Malformed frame or payload — the stream is no longer trustworthy."""


class FrameTooLarge(ProtocolError):
    """Peer claimed (or we tried to emit) more than ``MAX_FRAME``."""


class VersionMismatch(ProtocolError):
    """Peer speaks a different protocol version."""


class UnknownMessage(ProtocolError):
    """Frame type outside :class:`MsgType`."""


class TruncatedFrame(ProtocolError):
    """Stream ended part-way through a frame."""


class UnsupportedType(ProtocolError):
    """Value has no encoder — raised, never silently dropped."""


class FrameEOF(EOFError):
    """Clean end of stream observed at a frame boundary."""


# ── Codec registry (escape hatch) ───────────────────────────────────────────
_REGISTRY: dict[int, tuple[type, Callable[[Any], bytes], Callable[[bytes], Any]]] = {}


def register_type(
    tag: int,
    type_: type,
    encoder: Callable[[Any], bytes],
    decoder: Callable[[bytes], Any],
) -> None:
    """Teach the codec an application type.

    ``tag`` must be in ``0x40..0xFE`` so it can never shadow a built-in tag.
    Both peers must register the same ``(tag, type)`` pair; tags are explicit
    on purpose — deriving them from registration *order* would make two
    processes that initialise differently silently disagree.
    """
    if not _CUSTOM_TAG_MIN <= tag <= _CUSTOM_TAG_MAX:
        raise ValueError(f"custom tag 0x{tag:02x} outside 0x40..0xFE")
    if tag in _REGISTRY:
        raise ValueError(f"custom tag 0x{tag:02x} already registered")
    _REGISTRY[tag] = (type_, encoder, decoder)


# ── Encoding ────────────────────────────────────────────────────────────────


def _put_str(out: bytearray, tag: int, raw: bytes) -> None:
    out.append(tag)
    out += struct.pack(">I", len(raw))
    out += raw


def _encode_into(out: bytearray, value: Any) -> None:
    if value is None:
        out.append(_T_NONE)
        return
    if isinstance(value, bool):  # before int: bool is an int subclass
        out.append(_T_BOOL)
        out.append(1 if value else 0)
        return
    if isinstance(value, int):
        if _INT64_MIN <= value <= _INT64_MAX:
            out.append(_T_INT)
            out += struct.pack(">q", value)
        else:
            # Arbitrary precision survives as decimal text rather than
            # silently narrowing to int64.
            out.append(_T_BIGINT)
            raw = str(value).encode()
            out += struct.pack(">I", len(raw))
            out += raw
        return
    if isinstance(value, float):
        out.append(_T_FLOAT)
        out += struct.pack(">d", value)
        return
    if isinstance(value, str):
        _put_str(out, _T_STR, value.encode())
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        _put_str(out, _T_BYTES, bytes(value))
        return
    if isinstance(value, tuple):
        out.append(_T_TUPLE)
        out += struct.pack(">I", len(value))
        for item in value:
            _encode_into(out, item)
        return
    if isinstance(value, list):
        out.append(_T_LIST)
        out += struct.pack(">I", len(value))
        for item in value:
            _encode_into(out, item)
        return
    if isinstance(value, dict):
        out.append(_T_DICT)
        out += struct.pack(">I", len(value))
        for key, item in value.items():
            _encode_into(out, key)
            _encode_into(out, item)
        return
    # numpy: duck-typed so importing this module never forces a numpy import.
    if (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
        and hasattr(value, "ndim")
    ):
        _encode_ndarray(out, value)
        return
    for tag, (_ty, encoder, _decoder) in _REGISTRY.items():
        if isinstance(value, _ty):
            payload = encoder(value)
            out.append(_T_CUSTOM)
            out += struct.pack(">I", 2 + len(payload))
            out += struct.pack(">H", tag)
            out += payload
            return
    raise UnsupportedType(
        f"no encoder for {type(value).__module__}.{type(value).__qualname__} — "
        f"register one with frame.register_type()"
    )


def _encode_ndarray(out: bytearray, value: Any) -> None:
    dtype_name = str(value.dtype)
    shape = tuple(int(d) for d in value.shape)
    raw = value.tobytes(order="C")
    body = bytearray()
    dtype_raw = dtype_name.encode()
    body += struct.pack(">I", len(dtype_raw))
    body += dtype_raw
    body.append(len(shape))
    for dim in shape:
        body += struct.pack(">Q", dim)
    body += struct.pack(">I", len(raw))
    body += raw
    out.append(_T_NDARRAY)
    out += struct.pack(">I", len(body))
    out += body


def encode_value(value: Any) -> bytes:
    """Encode one value into a frame payload."""
    out = bytearray()
    _encode_into(out, value)
    return bytes(out)


# ── Decoding ────────────────────────────────────────────────────────────────


class _Cursor:
    """Bounds-checked reader — every out-of-range access is a typed error."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise TruncatedFrame(
                f"payload ended at {self.pos}, needed {n} more of {len(self.data)} bytes"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]


def _decode_ndarray(cur: _Cursor) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a hard dep here
        raise UnsupportedType("received an ndarray but numpy is unavailable") from exc
    body_len = cur.u32()
    body = _Cursor(cur.take(body_len))
    dtype_len = body.u32()
    dtype_name = body.take(dtype_len).decode()
    ndim = body.u8()
    shape = tuple(struct.unpack(">Q", body.take(8))[0] for _ in range(ndim))
    raw_len = body.u32()
    raw = body.take(raw_len)
    if body.pos != len(body.data):
        raise ProtocolError(f"{len(body.data) - body.pos} trailing bytes in ndarray")
    return np.frombuffer(raw, dtype=np.dtype(dtype_name)).reshape(shape).copy()


def _decode_from(cur: _Cursor) -> Any:
    tag = cur.u8()
    if tag == _T_NONE:
        return None
    if tag == _T_BOOL:
        return cur.u8() != 0
    if tag == _T_INT:
        return struct.unpack(">q", cur.take(8))[0]
    if tag == _T_BIGINT:
        return int(cur.take(cur.u32()).decode())
    if tag == _T_FLOAT:
        return struct.unpack(">d", cur.take(8))[0]
    if tag == _T_STR:
        return cur.take(cur.u32()).decode()
    if tag == _T_BYTES:
        return cur.take(cur.u32())
    if tag in (_T_LIST, _T_TUPLE):
        count = cur.u32()
        items = [_decode_from(cur) for _ in range(count)]
        return items if tag == _T_LIST else tuple(items)
    if tag == _T_DICT:
        count = cur.u32()
        return {_decode_from(cur): _decode_from(cur) for _ in range(count)}
    if tag == _T_NDARRAY:
        return _decode_ndarray(cur)
    if tag == _T_CUSTOM:
        length = cur.u32()
        payload = cur.take(length)
        inner = _Cursor(payload)
        reg_tag = struct.unpack(">H", inner.take(2))[0]
        entry = _REGISTRY.get(reg_tag)
        if entry is None:
            raise UnsupportedType(f"peer used unregistered custom tag 0x{reg_tag:02x}")
        value = entry[2](inner.take(len(payload) - 2))
        if inner.pos != len(inner.data):
            raise ProtocolError("trailing bytes in custom payload")
        return value
    raise UnsupportedType(f"unknown value tag 0x{tag:02x}")


def decode_value(data: bytes) -> Any:
    """Decode a frame payload. Total: returns a value or raises ProtocolError."""
    cur = _Cursor(data)
    value = _decode_from(cur)
    if cur.pos != len(cur.data):
        raise ProtocolError(f"{len(cur.data) - cur.pos} trailing bytes after value")
    return value


# ── Framing ─────────────────────────────────────────────────────────────────


def encode_frame(msg_type: MsgType | int, payload: bytes = b"") -> bytes:
    """Build one frame, refusing payloads larger than ``MAX_FRAME``."""
    if len(payload) > MAX_FRAME:
        raise FrameTooLarge(f"payload {len(payload)}B exceeds MAX_FRAME {MAX_FRAME}B")
    return _HEADER.pack(PROTOCOL_VERSION, int(msg_type), len(payload)) + payload


def encode_message(msg_type: MsgType | int, value: Any) -> bytes:
    """Build one frame whose payload is the encoding of ``value``."""
    return encode_frame(msg_type, encode_value(value))


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS) -> list[str]:
    """Split captured output so no single frame can bust ``MAX_FRAME``."""
    if not text:
        return [""]
    size = min(chunk_size, MAX_FRAME // 4)  # worst case 4 bytes/char
    return [text[i : i + size] for i in range(0, len(text), size)]


def wait_readable(sock: Any, timeout: float = POLL_INTERVAL) -> bool:
    """Poll a socket for readability — replaces ``conn.poll(timeout)``."""
    import select

    try:
        readable, _, _ = select.select([sock], [], [], timeout)
    except (OSError, ValueError):
        return False
    return bool(readable)


class FrameHandler:
    """Owns one duplex channel and the protocol spoken over it.

    One object owns *both* directions, the protocol version and
    ``max_frame`` — splitting those across a reader and a writer is exactly
    how the two drift apart and end up disagreeing about the bound.

    Use it simplex in normal operation: the child calls :meth:`send`, the
    parent calls :meth:`recv`. The duplex capability is spent on
    :meth:`handshake`, where both sides exchange :data:`MsgType.HELLO` so a
    version mismatch fails loudly before any payload moves.

    ``sock`` is anything exposing ``recv(n) -> bytes`` and
    ``sendall(bytes)`` — a ``socket.socketpair`` end, or a test double.
    """

    __slots__ = ("_sock", "_closed", "protocol", "max_frame", "peer_protocol", "handshake_timeout")

    def __init__(
        self,
        sock: Any,
        *,
        protocol: int = PROTOCOL_VERSION,
        max_frame: int = MAX_FRAME,
        handshake_timeout: float = HANDSHAKE_TIMEOUT,
    ) -> None:
        self._sock = sock
        self._closed = False
        self.protocol = protocol
        self.max_frame = max_frame
        self.handshake_timeout = handshake_timeout
        #: Version spoken by the peer, set by :meth:`handshake`.
        self.peer_protocol: int | None = None

    # ── handshake ────────────────────────────────────────────────────────

    def handshake(self) -> int:
        """Exchange HELLO frames; return the peer's protocol version.

        Both sides send *first*: HELLO is a couple of dozen bytes, so it fits
        the socket buffer and neither side can block on the send — then both
        receive. The socket timeout is scoped to this call only.
        """
        set_timeout = getattr(self._sock, "settimeout", None)
        get_timeout = getattr(self._sock, "gettimeout", None)
        previous = get_timeout() if get_timeout else None
        try:
            if set_timeout:
                set_timeout(self.handshake_timeout)
            try:
                self.send(MsgType.HELLO, self.protocol)
                msg_type, value = self.recv()
            except TimeoutError as exc:
                raise ProtocolError(f"handshake timed out after {self.handshake_timeout}s") from exc
        finally:
            if set_timeout:
                set_timeout(previous)

        if msg_type is not MsgType.HELLO:
            raise ProtocolError(f"expected HELLO, received {msg_type.name}")
        try:
            peer = int(value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed HELLO payload: {value!r}") from exc
        self.peer_protocol = peer
        if peer != self.protocol:
            raise VersionMismatch(f"peer speaks v{peer}, this side speaks v{self.protocol}")
        return peer

    # ── send ─────────────────────────────────────────────────────────────

    def send(self, msg_type: MsgType | int, value: Any = _MISSING) -> None:
        """Send one frame. Omitting ``value`` sends an empty payload."""
        payload = b"" if value is _MISSING else encode_value(value)
        if len(payload) > self.max_frame:
            raise FrameTooLarge(f"payload {len(payload)}B exceeds limit {self.max_frame}B")
        self._sock.sendall(encode_frame(msg_type, payload))

    def send_text(self, msg_type: MsgType | int, text: str, chunk_size: int = CHUNK_CHARS) -> None:
        """Emit captured output as consecutive codec-encoded chunks.

        Every payload in this protocol goes through the codec — raw text next
        to encoded values would recreate the mixed-encoding ambiguity (raw
        sentinel vs pickled tuple) this framing exists to replace.
        """
        for part in chunk_text(text, chunk_size):
            self.send(msg_type, part)

    # ── recv ─────────────────────────────────────────────────────────────

    def recv_raw(self) -> tuple[MsgType, bytes]:
        """Receive one frame as raw payload bytes.

        Header is validated — version, then length — *before* the payload read
        allocates anything: the length prefix must never become an
        unbounded-allocation primitive.
        """
        header = self._read_exact(_HEADER.size)
        version, mtype, length = _HEADER.unpack(header)
        if version != self.protocol:
            raise VersionMismatch(f"peer speaks v{version}, this side speaks v{self.protocol}")
        if length > self.max_frame:
            raise FrameTooLarge(f"peer claimed {length}B > max_frame {self.max_frame}B")
        payload = self._read_exact(length) if length else b""
        try:
            return MsgType(mtype), payload
        except ValueError as exc:
            raise UnknownMessage(f"unknown frame type 0x{mtype:02x}") from exc

    def recv(self) -> tuple[MsgType, Any]:
        """Receive one frame, decoding its payload through the codec."""
        msg_type, payload = self.recv_raw()
        if not payload:
            return msg_type, None
        return msg_type, decode_value(payload)

    # ── channel ──────────────────────────────────────────────────────────

    def poll(self, timeout: float = POLL_INTERVAL) -> bool:
        """True if a frame (or EOF) is ready within ``timeout`` seconds."""
        return wait_readable(self._sock, timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except Exception:
            logger.debug("failed to close frame channel", exc_info=True)

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                if not buf:
                    raise FrameEOF("stream closed at a frame boundary")
                raise TruncatedFrame(f"stream closed after {len(buf)}/{n} bytes")
            buf += chunk
        return bytes(buf)
