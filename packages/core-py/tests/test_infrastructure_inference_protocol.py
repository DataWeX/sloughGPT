"""Tests for inference IPC protocol — wire format and message types."""
from __future__ import annotations

import struct

from domains.infrastructure.inference_protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    ErrorResponse,
    GenerateRequest,
    GenerateResult,
    HealthRequest,
    HealthResponse,
    StreamDone,
    StreamStartRequest,
    StreamStopRequest,
    StreamToken,
    decode_header,
    encode_message,
)


class TestWireFormat:
    def test_encode_decode_roundtrip(self):
        msg = {"type": "health"}
        encoded = encode_message(msg)
        assert isinstance(encoded, bytes)
        # First 4 bytes are length
        length = struct.unpack(HEADER_FMT, encoded[:4])[0]
        assert length == len(encoded) - HEADER_SIZE

    def test_decode_header(self):
        msg = {"type": "generate", "prompt": "hello"}
        encoded = encode_message(msg)
        length = decode_header(encoded[:4])
        assert length > 0

    def test_header_size(self):
        assert HEADER_SIZE == 4


class TestHealthRequest:
    def test_default(self):
        r = HealthRequest()
        assert r.type == "health"


class TestHealthResponse:
    def test_defaults(self):
        r = HealthResponse()
        assert r.type == "health_ok"
        assert r.loaded is False
        assert r.model_id == ""


class TestGenerateRequest:
    def test_defaults(self):
        r = GenerateRequest()
        assert r.type == "generate"
        assert r.prompt == ""
        assert r.params == {}


class TestGenerateResult:
    def test_defaults(self):
        r = GenerateResult()
        assert r.type == "result"
        assert r.text == ""


class TestStreamStartRequest:
    def test_defaults(self):
        r = StreamStartRequest()
        assert r.type == "stream_start"


class TestStreamStopRequest:
    def test_defaults(self):
        r = StreamStopRequest()
        assert r.type == "stream_stop"


class TestStreamToken:
    def test_defaults(self):
        r = StreamToken()
        assert r.type == "token"
        assert r.token == ""


class TestStreamDone:
    def test_defaults(self):
        r = StreamDone()
        assert r.type == "stream_done"


class TestErrorResponse:
    def test_defaults(self):
        r = ErrorResponse()
        assert r.type == "error"
        assert r.message == ""
