"""Tests for domains/api/sse_envelope.py — standard SSE event formatting."""

import json
import pytest
from domains.api.sse_envelope import (
    SSEEnvelope,
    StreamPhase,
    StreamStatus,
    sse_event,
    sse_error,
    sse_complete,
    sse_token,
    _json_safe,
    TRAINING_SEQUENCE,
    CHAT_SEQUENCE,
)


class TestSSEEnvelopeDataclass:
    def test_to_dict_minimal(self):
        env = SSEEnvelope(stream="chat", phase="STREAMING", status="working")
        d = env.to_dict()
        assert d["stream"] == "chat"
        assert d["phase"] == "STREAMING"
        assert d["status"] == "working"
        assert d["message"] == ""
        assert d["data"] == {}
        assert d["meta"] == {}

    def test_to_dict_with_status_enum(self):
        env = SSEEnvelope(
            stream="chat", phase="STREAMING", status=StreamStatus.COMPLETE
        )
        d = env.to_dict()
        assert d["status"] == "complete"

    def test_to_dict_with_all_fields(self):
        env = SSEEnvelope(
            stream="auto-train",
            phase="TRAIN",
            status="working",
            data={"loss": 0.5, "step": 10},
            meta={"epoch": 1},
            message="loss=0.5000",
        )
        d = env.to_dict()
        assert d["data"]["loss"] == 0.5
        assert d["meta"]["epoch"] == 1
        assert d["message"] == "loss=0.5000"


class TestStreamEnums:
    def test_stream_phase_values(self):
        assert StreamPhase.IDLE.value == "IDLE"
        assert StreamPhase.TRAIN.value == "TRAIN"
        assert StreamPhase.STREAMING.value == "STREAMING"
        assert StreamPhase.ERROR.value == "ERROR"

    def test_stream_status_values(self):
        assert StreamStatus.WORKING.value == "working"
        assert StreamStatus.SUCCESS.value == "success"
        assert StreamStatus.ERROR.value == "error"
        assert StreamStatus.COMPLETE.value == "complete"

    def test_training_sequence_length(self):
        assert len(TRAINING_SEQUENCE) == 9
        assert TRAINING_SEQUENCE[0] == "IDLE"
        assert TRAINING_SEQUENCE[-3] == "COMPLETE"

    def test_chat_sequence_length(self):
        assert len(CHAT_SEQUENCE) == 4
        assert "STREAMING" in CHAT_SEQUENCE


class TestSSEEvent:
    def test_returns_data_line(self):
        result = sse_event("chat", "STREAMING", "working", message="")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_json_payload_structure(self):
        result = sse_event("chat", "STREAMING", "working", data={"token": "hi"})
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["stream"] == "chat"
        assert parsed["phase"] == "STREAMING"
        assert parsed["status"] == "working"
        assert parsed["data"]["token"] == "hi"

    def test_none_data_becomes_empty_dict(self):
        result = sse_event("chat", "STREAMING", "working", data=None)
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["data"] == {}

    def test_none_meta_becomes_empty_dict(self):
        result = sse_event("chat", "STREAMING", "working", meta=None)
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["meta"] == {}


class TestSSEError:
    def test_error_format(self):
        result = sse_error("chat", "STREAMING", "timeout")
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["status"] == "error"
        assert parsed["data"]["error"] == "timeout"
        assert "Error: timeout" in parsed["message"]


class TestSSEComplete:
    def test_complete_format(self):
        result = sse_complete("chat", message="Done")
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["status"] == "complete"
        assert parsed["phase"] == "COMPLETE"
        assert parsed["message"] == "Done"

    def test_complete_default_phase(self):
        result = sse_complete("chat")
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["phase"] == "COMPLETE"


class TestSSEToken:
    def test_token_working(self):
        result = sse_token("chat", "Hello")
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["status"] == "working"
        assert parsed["phase"] == "STREAMING"
        assert parsed["data"]["token"] == "Hello"

    def test_token_done(self):
        result = sse_token("chat", "", done=True)
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["status"] == "complete"
        assert parsed["phase"] == "STREAMING"

    def test_token_done_with_elapsed_ms(self):
        result = sse_token("chat", "", done=True, elapsed_ms=1234.5)
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert parsed["meta"]["elapsed_ms"] == 1234.5

    def test_token_no_elapsed_when_not_done(self):
        result = sse_token("chat", "tok", done=False, elapsed_ms=100.0)
        json_str = result[len("data: ") : -2]
        parsed = json.loads(json_str)
        assert "elapsed_ms" not in parsed["meta"]


class TestJsonSafe:
    def test_normal_values(self):
        assert _json_safe(42) == 42
        assert _json_safe("hello") == "hello"

    def test_numpy_like_object(self):
        class FakeArray:
            def tolist(self):
                return [1, 2, 3]

        assert _json_safe(FakeArray()) == [1, 2, 3]

    def test_numpy_scalar(self):
        class FakeScalar:
            def item(self):
                return 3.14

        assert _json_safe(FakeScalar()) == 3.14
