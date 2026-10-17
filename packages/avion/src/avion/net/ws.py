"""ws — avion's own RFC 6455 WebSocket client (stdlib asyncio streams).

The transport under CDP: every DevTools message rides one text frame.
avion rented the ``websockets`` library for exactly three verbs — connect,
send, async-iterate messages, close — so this module owns them and the
``cdp`` extra disappears: avion is zero-dependency again. This is the same
move PGQ made dropping ``multiprocessing``: own the protocol you live on.

What CDP needs, implemented honestly:
- client handshake: HTTP/1.1 Upgrade with ``Sec-WebSocket-Accept``
  validation (never trust a socket that isn't the peer we asked for);
- client -> server frames MASKED (spec MUST), server frames unmasked;
- 7/16/64-bit payload lengths both directions — DevTools payloads run to
  megabytes (screenshot frames, DOM dumps);
- continuation-frame assembly (a large message may arrive fragmented);
- control frames: auto-PONG on PING, CLOSE handshake on close.

Named omissions (not silently missing):
- ``wss://`` (TLS): local DevTools sockets are plain; raises honestly.
  Add when avion drives a remote browser.
- extensions (permessage-deflate): never offered in the handshake, so
  never negotiated — CDP does not use them.

The surface mirrors what ``backends/cdp.py`` already calls, so the swap
was one line: ``connect(url)`` returns a connection whose ``send`` takes a
str, whose async iteration yields text messages until the peer closes,
and whose ``close`` completes the close handshake.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct
from typing import AsyncIterator
from urllib.parse import urlsplit

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OP_CONT = 0x0
_OP_TEXT = 0x1
_OP_BIN = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


def accept_key(client_key: str) -> str:
    """The ``Sec-WebSocket-Accept`` for a client key (RFC 6455 §1.3)."""
    digest = hashlib.sha1((client_key + _GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _mask_payload(payload: bytes, mask: bytes) -> bytes:
    return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _encode_frame(opcode: int, payload: bytes, mask: bytes | None) -> bytes:
    """One unfragmented frame. Client frames MUST be masked; tests pass a
    fixed mask (the wire uses a fresh random one per frame)."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack(">H", length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack(">Q", length)
    if mask:
        header += mask
        payload = _mask_payload(payload, mask)
    return bytes(header) + payload


class WsConnection:
    """A client WebSocket over asyncio streams — the three-verb surface."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, buffer: bytes = b""):
        self._reader = reader
        self._writer = writer
        self._buffer = buffer  # bytes read past the handshake headers
        self._close_sent = False
        self._closed = False

    async def _read_exact(self, n: int) -> bytes:
        """n bytes, draining any post-handshake remainder first."""
        if n <= 0:
            return b""
        if self._buffer:
            take, self._buffer = self._buffer[:n], self._buffer[n:]
            if len(take) == n:
                return take
            return take + await self._reader.readexactly(n - len(take))
        return await self._reader.readexactly(n)

    async def _read_frame(self) -> tuple[bool, int, bytes]:
        """One raw frame -> (fin, opcode, unmasked payload)."""
        b1, b2 = await self._read_exact(2)
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack(">H", await self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", await self._read_exact(8))[0]
        mask = await self._read_exact(4) if masked else None
        payload = await self._read_exact(length)
        if mask:
            payload = _mask_payload(payload, mask)
        return fin, opcode, payload

    async def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        """One frame, ONE write call (frame-granular atomicity for
        concurrent senders: reader-task pongs vs. command sends)."""
        frame = _encode_frame(opcode, payload, os.urandom(4))
        self._writer.write(frame)
        await self._writer.drain()

    async def send(self, message: str) -> None:
        """Send one text message (masked, single frame)."""
        if self._closed:
            raise ConnectionError("send after close")
        await self._send_frame(_OP_TEXT, message.encode())

    async def recv(self) -> str:
        """The next complete text message; raises ConnectionError when the
        peer closes (iteration over the connection ends quietly on this)."""
        fragments: list[bytes] = []
        message_op: int | None = None
        while True:
            try:
                fin, opcode, payload = await self._read_frame()
            except (asyncio.IncompleteReadError, ConnectionResetError, OSError) as e:
                self._closed = True
                raise ConnectionError("websocket connection lost") from e
            if opcode == _OP_PING:
                await self._send_frame(_OP_PONG, payload)
                continue
            if opcode == _OP_PONG:
                continue
            if opcode == _OP_CLOSE:
                if not self._close_sent:
                    self._close_sent = True
                    try:
                        await self._send_frame(_OP_CLOSE, payload[:2])
                    except OSError:
                        pass
                self._closed = True
                raise ConnectionError("websocket closed by peer")
            if opcode in (_OP_TEXT, _OP_BIN):
                fragments = [payload]
                message_op = opcode
            elif opcode == _OP_CONT:
                if message_op is None:
                    self._closed = True
                    raise ConnectionError("continuation without an opening frame")
                fragments.append(payload)
            if fin:
                data = b"".join(fragments)
                return data.decode() if message_op == _OP_TEXT else data.decode(errors="replace")

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        try:
            return await self.recv()
        except ConnectionError:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        """Close handshake (best-effort), then drop the transport."""
        if not self._closed and not self._close_sent:
            self._close_sent = True
            try:
                await self._send_frame(_OP_CLOSE)
            except OSError:
                pass
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except OSError:
            pass


async def connect(url: str, *, timeout: float = 15.0) -> WsConnection:
    """Open a client WebSocket at ``ws://`` (validated handshake).

    Raises ``NotImplementedError`` for ``wss://`` (named omission) and
    ``RuntimeError`` if the peer fails the 101/accept validation — a socket
    that does not prove itself is not the DevTools endpoint.
    """
    parts = urlsplit(url)
    if parts.scheme == "wss":
        raise NotImplementedError(
            "wss:// (TLS) is not implemented — local DevTools sockets are plain ws://"
        )
    if parts.scheme != "ws":
        raise ValueError(f"expected a ws:// url, got {parts.scheme!r}: {url}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port), timeout=timeout
    )
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    try:
        writer.write(request.encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=timeout)
    except BaseException:
        writer.close()
        raise
    header_block, _, leftover = head.partition(b"\r\n\r\n")
    lines = header_block.decode(errors="replace").split("\r\n")
    status = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    if " 101" not in status:
        writer.close()
        raise RuntimeError(f"websocket handshake refused: {status!r}")
    if headers.get("sec-websocket-accept", "") != accept_key(key):
        writer.close()
        raise RuntimeError(
            "websocket handshake failed Sec-WebSocket-Accept validation "
            "(wrong or hostile peer)"
        )
    return WsConnection(reader, writer, buffer=leftover)
