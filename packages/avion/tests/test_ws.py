"""ws tests — avion's own RFC 6455 client against RFC vectors and a raw
stdlib frame-speaking server (no third-party anywhere in the loop).

Ground truth is the PROTOCOL: tests hand-build frames byte by byte (the
server side) and assert the client's exact bytes on the wire (client
side), then exercise the full handshake over a real localhost socket.
"""

from __future__ import annotations

import asyncio
import base64
import struct

import pytest
from avion.net.ws import (
    _encode_frame,  # noqa: PLC2701 — wire-level tests need the encoder
    accept_key,
    connect,
)

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class TestAcceptKey:
    def test_rfc_6455_example_vector(self):
        """The literal example from RFC 6455 §1.3."""
        assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    def test_key_is_hashed_with_guid(self):
        import hashlib

        raw = base64.b64encode(hashlib.sha1(("abc" + GUID).encode()).digest()).decode()
        assert accept_key("abc") == raw


class TestEncodeFrame:
    MASK = b"\x01\x02\x03\x04"

    def test_small_text_frame_is_masked_per_spec(self):
        frame = _encode_frame(0x1, b"hi", self.MASK)
        assert frame[0] == 0x81  # FIN + text
        assert frame[1] == 0x80 | 2  # masked, len 2
        assert frame[2:6] == self.MASK
        masked = frame[6:]
        assert bytes(masked[i] ^ self.MASK[i % 4] for i in range(2)) == b"hi"

    def test_16bit_length(self):
        frame = _encode_frame(0x1, b"x" * 130, self.MASK)
        assert frame[1] == 0x80 | 126
        assert struct.unpack(">H", frame[2:4])[0] == 130

    def test_64bit_length(self):
        frame = _encode_frame(0x2, b"x" * 70000, self.MASK)
        assert frame[1] == 0x80 | 127
        assert struct.unpack(">Q", frame[2:10])[0] == 70000


def _server_frame(opcode: int, payload: bytes, *, fin: bool = True) -> bytes:
    """A hand-built UNMASKED server frame (spec: servers never mask)."""
    b1 = (0x80 if fin else 0) | opcode
    length = len(payload)
    if length < 126:
        return bytes([b1, length]) + payload
    if length < 65536:
        return bytes([b1, 126]) + struct.pack(">H", length) + payload
    return bytes([b1, 127]) + struct.pack(">Q", length) + payload


async def _read_client_frame(reader: asyncio.StreamReader) -> tuple[int, bytes, bool]:
    """Decode one possibly-masked client frame -> (opcode, payload, masked)."""
    b1, b2 = await reader.readexactly(2)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if masked else None
    payload = await reader.readexactly(length)
    if mask:
        payload = bytes(payload[i] ^ mask[i % 4] for i in range(length))
    return opcode, payload, masked


async def _handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, accept: str | None = None, status: str = "101 Switching Protocols") -> None:
    """Read the client's Upgrade request; reply with the given status."""
    request = await reader.readuntil(b"\r\n\r\n")
    if accept is None:
        for line in request.decode().split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                accept = accept_key(line.split(":", 1)[1].strip())
                break
    response = (
        f"HTTP/1.1 {status}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    )
    writer.write(response.encode())
    await writer.drain()


class _RawServer:
    """A raw asyncio server speaking the wire protocol by hand."""

    def __init__(self, script):
        self.script = script  # async (reader, writer) -> None
        self._server = None
        self.port = 0

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    async def _serve(self, reader, writer):
        try:
            await self.script(reader, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/devtools/browser/test"


async def _echo_script(reader, writer):
    await _handshake(reader, writer)
    while True:
        try:
            opcode, payload, _ = await _read_client_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        if opcode == 0x8:  # close: reply and end
            writer.write(_server_frame(0x8, payload[:2]))
            await writer.drain()
            return
        if opcode == 0x9:  # ping: pong
            writer.write(_server_frame(0xA, payload))
            await writer.drain()
            continue
        writer.write(_server_frame(0x1, payload))  # echo text
        await writer.drain()


class TestConnection:
    async def test_handshake_send_recv_round_trip(self):
        async with _RawServer(_echo_script) as server:
            conn = await connect(server.url)
            try:
                await conn.send("hello devtools")
                assert await conn.recv() == "hello devtools"
            finally:
                await conn.close()

    async def test_client_masks_every_frame(self):
        seen: list[bytes] = []

        async def capture(reader, writer):
            await _handshake(reader, writer)
            raw = await reader.readexactly(2 + 4 + len(b"masked"))
            seen.append(raw)
            writer.write(_server_frame(0x1, b"ok"))
            await writer.drain()

        async with _RawServer(capture) as server:
            conn = await connect(server.url)
            try:
                await conn.send("masked")
                assert await conn.recv() == "ok"
            finally:
                await conn.close()
        assert seen[0][1] & 0x80, "client frame was not masked (spec MUST)"

    async def test_large_64bit_length_payload(self):
        payload = "y" * 70000  # forces the 64-bit length both ways

        async def big(reader, writer):
            await _handshake(reader, writer)
            opcode, data, _ = await _read_client_frame(reader)
            writer.write(_server_frame(0x1, data))  # already bytes
            await writer.drain()

        async with _RawServer(big) as server:
            conn = await connect(server.url)
            try:
                await conn.send(payload)
                assert await conn.recv() == payload
            finally:
                await conn.close()

    async def test_fragmented_message_is_assembled(self):
        async def fragment(reader, writer):
            await _handshake(reader, writer)
            # No client frame to wait for — the client only receives here.
            writer.write(_server_frame(0x1, b"frag", fin=False))
            writer.write(_server_frame(0x0, b"mented", fin=True))
            await writer.drain()

        async with _RawServer(fragment) as server:
            conn = await connect(server.url)
            try:
                assert await conn.recv() == "fragmented"
            finally:
                await conn.close()

    async def test_ping_gets_auto_pong(self):
        async def pinger(reader, writer):
            await _handshake(reader, writer)
            writer.write(_server_frame(0x9, b"tick"))
            await writer.drain()
            opcode, payload, _ = await _read_client_frame(reader)
            assert (opcode, payload) == (0xA, b"tick")  # the pong arrived
            writer.write(_server_frame(0x1, b"alive"))
            await writer.drain()

        async with _RawServer(pinger) as server:
            conn = await connect(server.url)
            try:
                assert await conn.recv() == "alive"  # ping transparently skipped
            finally:
                await conn.close()

    async def test_close_handshake_ends_iteration_quietly(self):
        async def closing(reader, writer):
            await _handshake(reader, writer)
            await _read_client_frame(reader)
            writer.write(_server_frame(0x8, b"\x03\xe8"))  # 1000 normal
            await writer.drain()

        async with _RawServer(closing) as server:
            conn = await connect(server.url)
            await conn.send("bye")
            messages = [m async for m in conn]  # must end, not raise
            assert messages == []

    async def test_send_after_close_raises(self):
        async with _RawServer(_echo_script) as server:
            conn = await connect(server.url)
            await conn.close()
            with pytest.raises(ConnectionError):
                await conn.send("too late")


class TestHandshakeValidation:
    async def test_non_101_is_refused(self):
        async def refuse(reader, writer):
            await _handshake(reader, writer, status="200 OK")

        async with _RawServer(refuse) as server:
            with pytest.raises(RuntimeError, match="refused"):
                await connect(server.url)

    async def test_wrong_accept_is_rejected(self):
        async def hostile(reader, writer):
            await _handshake(reader, writer, accept="bogus==")

        async with _RawServer(hostile) as server:
            with pytest.raises(RuntimeError, match="Accept"):
                await connect(server.url)

    def test_wss_is_a_named_omission(self):
        with pytest.raises(NotImplementedError, match="wss"):
            asyncio.run(connect("wss://127.0.0.1:1/"))

    def test_wrong_scheme_rejected(self):
        with pytest.raises(ValueError, match="ws://"):
            asyncio.run(connect("http://127.0.0.1:1/"))
