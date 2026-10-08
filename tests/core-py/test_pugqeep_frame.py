"""Tests for the execution-pipeline frame protocol."""

from __future__ import annotations

import socket
import struct
import threading

import numpy as np
import pytest

import domain.infrastructure._internal.pugqeep.frame as frame_mod
from domain.infrastructure._internal.pugqeep.frame import (
    HANDSHAKE_TIMEOUT,
    MAX_FRAME,
    PROTOCOL_VERSION,
    FrameEOF,
    FrameHandler,
    FrameTooLarge,
    MsgType,
    ProtocolError,
    UnknownMessage,
    UnsupportedType,
    VersionMismatch,
    chunk_text,
    decode_value,
    encode_frame,
    encode_value,
    register_type,
    wait_readable,
)


class _Channel:
    """Socket-like test double: buffered inbound bytes, captured outbound."""

    def __init__(
        self, incoming: bytes = b"", *, step: int | None = None, fail_recv: bool = False
    ) -> None:
        self._incoming = bytearray(incoming)
        self._step = step
        self._fail_recv = fail_recv
        self._timeout = None
        self.sent: list[bytes] = []
        self.reads: list[int] = []
        self.closed = False

    def recv(self, n: int) -> bytes:
        self.reads.append(n)
        if self._fail_recv:
            raise TimeoutError("simulated timeout")
        if self._step is not None:
            n = min(n, self._step)
        if not self._incoming:
            return b""
        chunk = bytes(self._incoming[:n])
        del self._incoming[:n]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    def close(self) -> None:
        self.closed = True

    def settimeout(self, value) -> None:
        self._timeout = value

    def gettimeout(self):
        return self._timeout


def _handler(data: bytes = b"", *, step: int | None = None, **kwargs) -> FrameHandler:
    return FrameHandler(_Channel(data, step=step), **kwargs)


def _sent_values(channel: _Channel) -> list[tuple[MsgType, object]]:
    return [_handler(frame).recv() for frame in channel.sent]


def _roundtrip(value):
    return decode_value(encode_value(value))


# ── codec: built-ins ────────────────────────────────────────────────────────


class TestCodecBuiltins:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            False,
            0,
            -1,
            42,
            -(2**62),
            3.5,
            -0.0,
            "",
            "hello",
            "unicode ✓ naïve",
            b"",
            b"\x00\xff binary",
            [],
            [1, 2, 3],
            (),
            (1, "a", None),
            {},
            {"a": 1, "b": [2, 3]},
        ],
    )
    def test_roundtrip(self, value):
        assert _roundtrip(value) == value

    def test_bool_is_not_int(self):
        """bool must not collapse into int (True == 1 would hide a type bug)."""
        assert isinstance(_roundtrip(True), bool)
        assert _roundtrip(True) is True
        assert type(_roundtrip(1)) is int

    def test_nested_structure(self):
        value = {"runs": [{"loss": 0.5, "ok": True}, {"loss": None}]}
        assert _roundtrip(value) == value

    def test_arbitrary_precision_int_survives(self):
        big = 2**200 + 7
        out = _roundtrip(big)
        assert type(out) is int
        assert out == big

    def test_bytes_stay_bytes(self):
        assert isinstance(_roundtrip(b"abc"), bytes)


# ── codec: numpy ────────────────────────────────────────────────────────────


class TestCodecNumpy:
    @pytest.mark.parametrize(
        "array",
        [
            np.zeros((2, 3), dtype=np.float32),
            np.arange(10, dtype=np.int64),
            np.array([1.5, 2.5], dtype=np.float64),
            np.array([True, False]),
            np.zeros((0,), dtype=np.uint8),
            np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        ],
    )
    def test_roundtrip_preserves_dtype_shape_and_values(self, array):
        out = _roundtrip(array)
        assert isinstance(out, np.ndarray)
        assert out.dtype == array.dtype
        assert out.shape == array.shape
        np.testing.assert_array_equal(out, array)

    def test_non_contiguous_array_is_materialised(self):
        array = np.arange(12).reshape(3, 4)[:, ::2]
        out = _roundtrip(array)
        np.testing.assert_array_equal(out, array)


# ── codec: totality ─────────────────────────────────────────────────────────


class TestCodecTotality:
    def test_unsupported_type_raises_typed_error(self):
        class Widget:
            pass

        with pytest.raises(UnsupportedType) as exc:
            encode_value(Widget())
        assert "Widget" in str(exc.value)

    def test_trailing_bytes_rejected(self):
        with pytest.raises(ProtocolError):
            decode_value(encode_value(1) + b"\x00")

    def test_truncated_payload_rejected(self):
        with pytest.raises(ProtocolError):
            decode_value(encode_value({"key": "value"})[:-3])

    def test_unknown_value_tag_rejected(self):
        with pytest.raises(UnsupportedType):
            decode_value(b"\xfe\x00")


# ── codec: registry escape hatch ────────────────────────────────────────────


class TestCodecRegistry:
    def setup_method(self):
        self._added = []

    def teardown_method(self):
        for tag in self._added:
            frame_mod._REGISTRY.pop(tag, None)

    def _register(self, tag, type_, enc, dec):
        register_type(tag, type_, enc, dec)
        self._added.append(tag)

    def test_custom_type_roundtrip(self):
        class Widget:
            def __init__(self, n):
                self.n = n

        self._register(
            0x40,
            Widget,
            lambda w: struct.pack(">q", w.n),
            lambda b: Widget(struct.unpack(">q", b)[0]),
        )
        out = _roundtrip(Widget(7))
        assert isinstance(out, Widget)
        assert out.n == 7

    def test_custom_type_nested_in_structures(self):
        class Widget:
            def __init__(self, n):
                self.n = n

        self._register(
            0x41,
            Widget,
            lambda w: struct.pack(">q", w.n),
            lambda b: Widget(struct.unpack(">q", b)[0]),
        )
        out = _roundtrip({"widgets": [Widget(1), Widget(2)]})
        assert [w.n for w in out["widgets"]] == [1, 2]

    def test_unregistered_custom_tag_rejected(self):
        raw = bytes([0x0A]) + struct.pack(">I", 2) + struct.pack(">H", 0x7E)
        with pytest.raises(UnsupportedType):
            decode_value(raw)

    def test_tag_range_enforced(self):
        with pytest.raises(ValueError):
            register_type(0x10, int, lambda i: b"", lambda b: 0)

    def test_duplicate_tag_rejected(self):
        self._register(0x42, dict, lambda d: b"", lambda b: {})
        with pytest.raises(ValueError):
            register_type(0x42, list, lambda d: b"", lambda b: [])


# ── framing ─────────────────────────────────────────────────────────────────


class TestFraming:
    def test_header_layout(self):
        frame = encode_frame(MsgType.RESULT, b"abc")
        version, mtype, length = struct.unpack(">BBI", frame[:6])
        assert version == PROTOCOL_VERSION
        assert mtype == MsgType.RESULT
        assert length == 3
        assert frame[6:] == b"abc"

    def test_recv_roundtrip(self):
        msg_type, value = _handler(encode_frame(MsgType.RESULT)).recv()
        assert msg_type is MsgType.RESULT
        assert value is None

    def test_recv_decodes_payload(self):
        frame = frame_mod.encode_message(MsgType.RESULT, {"loss": 0.5})
        assert _handler(frame).recv() == (MsgType.RESULT, {"loss": 0.5})

    def test_multiple_frames_back_to_back(self):
        blob = (
            encode_frame(MsgType.READY)
            + frame_mod.encode_message(MsgType.HEARTBEAT, 1.0)
            + frame_mod.encode_message(MsgType.RESULT, [1, 2])
        )
        handler = _handler(blob)
        assert handler.recv() == (MsgType.READY, None)
        assert handler.recv() == (MsgType.HEARTBEAT, 1.0)
        assert handler.recv() == (MsgType.RESULT, [1, 2])

    @pytest.mark.parametrize("step", [1, 2, 3, 5, 7])
    def test_partial_reads_still_frame(self, step):
        """A byte stream delivers no message boundaries — read_exact must cope."""
        blob = frame_mod.encode_message(MsgType.ERROR, "boom") + frame_mod.encode_message(
            MsgType.RESULT, 42
        )
        handler = _handler(blob, step=step)
        assert handler.recv() == (MsgType.ERROR, "boom")
        assert handler.recv() == (MsgType.RESULT, 42)

    def test_send_encodes_value_payload(self):
        channel = _Channel()
        FrameHandler(channel).send(MsgType.RESULT, [1, 2])
        assert _sent_values(channel) == [(MsgType.RESULT, [1, 2])]

    def test_send_without_value_sends_empty_payload(self):
        channel = _Channel()
        FrameHandler(channel).send(MsgType.READY)
        assert channel.sent[0][6:] == b""

    def test_send_rejects_oversized_payload(self):
        with pytest.raises(FrameTooLarge):
            FrameHandler(_Channel()).send(MsgType.RESULT, b"x" * (MAX_FRAME + 1))

    def test_length_prefix_is_not_an_allocation_primitive(self):
        """A peer claiming 4GB must be rejected from the header alone."""
        header = struct.pack(">BBI", PROTOCOL_VERSION, int(MsgType.RESULT), 4_000_000_000)
        with pytest.raises(FrameTooLarge):
            _handler(header).recv()

    def test_length_checked_before_payload_read(self):
        """Prove the header check fires *before* any payload read is requested."""
        header = struct.pack(">BBI", PROTOCOL_VERSION, int(MsgType.RESULT), 999)
        channel = _Channel(header)
        with pytest.raises(FrameTooLarge):
            FrameHandler(channel, max_frame=16).recv()
        assert channel.reads == [6], f"payload read requested: {channel.reads}"

    def test_version_mismatch_rejected(self):
        bad = struct.pack(">BBI", 99, int(MsgType.RESULT), 0)
        with pytest.raises(VersionMismatch):
            _handler(bad).recv()

    def test_unknown_message_type_rejected(self):
        bad = struct.pack(">BBI", PROTOCOL_VERSION, 0x7F, 0)
        with pytest.raises(UnknownMessage):
            _handler(bad).recv()

    def test_clean_eof_raises_frame_eof(self):
        with pytest.raises(FrameEOF):
            _handler(b"").recv()

    def test_mid_frame_close_raises_truncated(self):
        header = struct.pack(">BBI", PROTOCOL_VERSION, int(MsgType.RESULT), 10)
        with pytest.raises(ProtocolError) as exc:
            _handler(header + b"abc").recv()
        assert type(exc.value).__name__ == "TruncatedFrame"


# ── handshake ───────────────────────────────────────────────────────────────


class TestHandshake:
    def test_handshake_records_peer_protocol(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, PROTOCOL_VERSION)
        handler = _handler(peer_hello)
        assert handler.handshake() == PROTOCOL_VERSION
        assert handler.peer_protocol == PROTOCOL_VERSION

    def test_handshake_sends_our_version(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, PROTOCOL_VERSION)
        channel = _Channel(peer_hello)
        FrameHandler(channel).handshake()
        assert _sent_values(channel) == [(MsgType.HELLO, PROTOCOL_VERSION)]

    def test_handshake_rejects_version_mismatch(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, 99)
        with pytest.raises(VersionMismatch):
            _handler(peer_hello).handshake()

    def test_handshake_rejects_non_hello_first_frame(self):
        with pytest.raises(ProtocolError, match="expected HELLO"):
            _handler(encode_frame(MsgType.READY)).handshake()

    def test_handshake_rejects_malformed_payload(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, "not-a-number")
        with pytest.raises(ProtocolError, match="malformed HELLO"):
            _handler(peer_hello).handshake()

    def test_handshake_timeout_becomes_protocol_error(self):
        channel = _Channel(fail_recv=True)
        handler = FrameHandler(channel, handshake_timeout=0.01)
        with pytest.raises(ProtocolError, match="timed out"):
            handler.handshake()

    def test_handshake_restores_socket_timeout(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, PROTOCOL_VERSION)
        channel = _Channel(peer_hello)
        channel.settimeout(7.5)
        FrameHandler(channel).handshake()
        assert channel.gettimeout() == 7.5

    def test_handshake_timeout_is_scoped_not_leaked(self):
        peer_hello = frame_mod.encode_message(MsgType.HELLO, PROTOCOL_VERSION)
        channel = _Channel(peer_hello)
        channel.settimeout(None)
        FrameHandler(channel, handshake_timeout=0.01).handshake()
        assert channel.gettimeout() is None
        assert HANDSHAKE_TIMEOUT == 30.0


# ── output chunking ─────────────────────────────────────────────────────────


class TestChunking:
    def test_chunk_respects_size(self):
        parts = chunk_text("x" * 100, chunk_size=30)
        assert [len(p) for p in parts] == [30, 30, 30, 10]
        assert "".join(parts) == "x" * 100

    def test_empty_text_yields_one_frame(self):
        assert chunk_text("") == [""]

    def test_default_chunk_size_never_exceeds_frame_limit(self):
        assert all(len(p.encode()) <= MAX_FRAME for p in chunk_text("z" * 2_000_000))

    def test_handler_emits_consecutive_stdout_frames(self):
        channel = _Channel()
        FrameHandler(channel).send_text(MsgType.STDOUT, "y" * 100, chunk_size=30)
        results = _sent_values(channel)
        assert len(results) > 1
        assert all(t is MsgType.STDOUT for t, _ in results)
        assert "".join(v for _, v in results) == "y" * 100


# ── socket integration ──────────────────────────────────────────────────────


class TestSocketPair:
    def test_end_to_end_over_socketpair(self):
        parent, child = socket.socketpair()
        try:
            handler = FrameHandler(child)
            handler.send(MsgType.READY)
            handler.send(MsgType.HEARTBEAT, 1.5)
            handler.send(MsgType.RESULT, {"acc": 0.99})
            handler.send(MsgType.ERROR, "kaboom")
            handler.send_text(MsgType.STDOUT, "log line\n")
            handler.send(MsgType.EXIT)
            child.close()

            reader = FrameHandler(parent)
            assert reader.recv() == (MsgType.READY, None)
            assert reader.recv() == (MsgType.HEARTBEAT, 1.5)
            assert reader.recv() == (MsgType.RESULT, {"acc": 0.99})
            assert reader.recv() == (MsgType.ERROR, "kaboom")
            assert reader.recv() == (MsgType.STDOUT, "log line\n")
            assert reader.recv() == (MsgType.EXIT, None)
            with pytest.raises(FrameEOF):
                reader.recv()
        finally:
            parent.close()
            child.close()

    def test_handshake_over_socketpair(self):
        parent, child = socket.socketpair()
        results = {}

        def responder():
            try:
                results["child"] = FrameHandler(child, handshake_timeout=5).handshake()
            finally:
                child.close()

        thread = threading.Thread(target=responder, daemon=True)
        thread.start()
        try:
            parent_handler = FrameHandler(parent, handshake_timeout=5)
            results["parent"] = parent_handler.handshake()
        finally:
            thread.join(timeout=5)
            parent.close()

        assert results["parent"] == PROTOCOL_VERSION
        assert results["child"] == PROTOCOL_VERSION

    def test_wait_readable(self):
        parent, child = socket.socketpair()
        try:
            assert wait_readable(parent, 0.05) is False
            child.sendall(b"x")
            assert wait_readable(parent, 0.5) is True
        finally:
            parent.close()
            child.close()

    def test_poll_helper_agrees_with_wait_readable(self):
        parent, child = socket.socketpair()
        try:
            handler = FrameHandler(parent)
            assert handler.poll(0.05) is False
            child.sendall(b"x")
            assert handler.poll(0.5) is True
        finally:
            parent.close()
            child.close()

    def test_close_is_idempotent(self):
        parent, child = socket.socketpair()
        handler = FrameHandler(parent)
        assert handler.is_open
        handler.close()
        handler.close()
        assert not handler.is_open
        child.close()

    def test_numpy_result_survives_the_wire(self):
        parent, child = socket.socketpair()
        try:
            expected = np.linspace(0, 1, 7, dtype=np.float32)
            FrameHandler(child).send(MsgType.RESULT, expected)
            child.close()
            _t, value = FrameHandler(parent).recv()
            np.testing.assert_array_equal(value, expected)
            assert value.dtype == expected.dtype
        finally:
            parent.close()
            child.close()
