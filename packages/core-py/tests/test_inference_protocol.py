"""Tests for inference_protocol.py — IPC wire format."""

import json
import struct
import pytest

from domains.infrastructure.inference_protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    encode_message,
    decode_header,
    HealthRequest,
    HealthResponse,
    GenerateRequest,
    GenerateResult,
    StreamStartRequest,
    StreamStopRequest,
    StreamToken,
    StreamDone,
    ErrorResponse,
)


class TestWireFormat:
    def test_header_size(self):
        assert HEADER_SIZE == 4

    def test_encode_message_length_prefix(self):
        msg = {"type": "health"}
        encoded = encode_message(msg)
        length = struct.unpack(HEADER_FMT, encoded[:4])[0]
        payload = encoded[4:]
        assert length == len(payload)
        assert json.loads(payload) == msg

    def test_decode_header_matches_length(self):
        msg = {"type": "generate", "id": "abc", "prompt": "hello"}
        encoded = encode_message(msg)
        decoded_len = decode_header(encoded[:4])
        assert decoded_len == len(encoded[4:])

    def test_encode_decode_roundtrip(self):
        msg = {"type": "token", "id": "t1", "token": "hi"}
        encoded = encode_message(msg)
        length = decode_header(encoded[:4])
        payload_bytes = encoded[4:4 + length]
        decoded = json.loads(payload_bytes)
        assert decoded == msg

    def test_empty_payload(self):
        msg = {"type": "health"}
        encoded = encode_message(msg)
        assert len(encoded) >= HEADER_SIZE

    def test_large_payload(self):
        msg = {"type": "result", "text": "x" * 100_000}
        encoded = encode_message(msg)
        length = decode_header(encoded[:4])
        assert length == len(encoded) - HEADER_SIZE
        payload = json.loads(encoded[4:])
        assert len(payload["text"]) == 100_000

    def test_unicode_payload(self):
        msg = {"type": "token", "token": "\u4f60\u597d\u4e16\u754c"}
        encoded = encode_message(msg)
        payload = json.loads(encoded[4:])
        assert payload["token"] == "\u4f60\u597d\u4e16\u754c"

    def test_compact_json(self):
        msg = {"type": "health"}
        encoded = encode_message(msg)
        payload_str = encoded[4:].decode()
        # compact: no spaces after separators
        assert ", " not in payload_str
        assert ": " not in payload_str


class TestMessageTypes:
    def test_health_request_defaults(self):
        m = HealthRequest()
        assert m.type == "health"

    def test_health_response_defaults(self):
        m = HealthResponse()
        assert m.type == "health_ok"
        assert m.model_id == ""
        assert m.loaded is False
        assert m.model_type == ""
        assert m.quantized is False

    def test_health_response_fields(self):
        m = HealthResponse(model_id="gpt2", loaded=True, model_type="slonet", quantized=True)
        assert m.model_id == "gpt2"
        assert m.loaded is True
        assert m.model_type == "slonet"
        assert m.quantized is True

    def test_generate_request_defaults(self):
        m = GenerateRequest()
        assert m.type == "generate"
        assert m.id == ""
        assert m.prompt == ""
        assert m.params == {}

    def test_generate_request_fields(self):
        m = GenerateRequest(id="r1", prompt="hello", params={"max_tokens": 100})
        assert m.id == "r1"
        assert m.prompt == "hello"
        assert m.params == {"max_tokens": 100}

    def test_generate_result_defaults(self):
        m = GenerateResult()
        assert m.type == "result"
        assert m.text == ""
        assert m.meta == {}

    def test_generate_result_fields(self):
        m = GenerateResult(id="r1", text="response", meta={"tokens": 5})
        assert m.text == "response"
        assert m.meta == {"tokens": 5}

    def test_stream_start_request(self):
        m = StreamStartRequest(id="s1", prompt="go", params={"temp": 0.7})
        assert m.type == "stream_start"
        assert m.id == "s1"
        assert m.prompt == "go"
        assert m.params == {"temp": 0.7}

    def test_stream_stop_request(self):
        m = StreamStopRequest(id="s1")
        assert m.type == "stream_stop"
        assert m.id == "s1"

    def test_stream_token(self):
        m = StreamToken(id="s1", token="hello")
        assert m.type == "token"
        assert m.token == "hello"

    def test_stream_done(self):
        m = StreamDone(id="s1", meta={"elapsed_ms": 500})
        assert m.type == "stream_done"
        assert m.meta == {"elapsed_ms": 500}

    def test_error_response(self):
        m = ErrorResponse(id="r1", message="timeout")
        assert m.type == "error"
        assert m.message == "timeout"

    def test_all_message_types_serializable(self):
        """Every dataclass can be encoded without error."""
        messages = [
            HealthRequest(),
            HealthResponse(model_id="m", loaded=True),
            GenerateRequest(id="1", prompt="p"),
            GenerateResult(id="1", text="t"),
            StreamStartRequest(id="1", prompt="p"),
            StreamStopRequest(id="1"),
            StreamToken(id="1", token="t"),
            StreamDone(id="1"),
            ErrorResponse(id="1", message="e"),
        ]
        for m in messages:
            d = m.__dict__
            encoded = encode_message(d)
            assert encoded[:4] == struct.pack(HEADER_FMT, len(encoded[4:]))
