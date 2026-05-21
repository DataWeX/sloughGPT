"""
SSE Envelope Tests — validates the standard streaming envelope format.

Every SSE endpoint uses this envelope. These tests ensure the format
is consistent and backward-compatible with all frontend parsers.
"""

import json
import sys
import os

# Ensure core-py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from domains.api.sse_envelope import (
    SSEEnvelope,
    StreamPhase,
    StreamStatus,
    sse_event,
    sse_token,
    sse_error,
    sse_complete,
    _json_safe,
)


class TestSSEEnvelope:
    """Core envelope construction."""

    def test_defaults(self):
        """Default SSEEnvelope has empty data/meta and blank message."""
        env = SSEEnvelope(stream="test", phase="STREAMING", status="working")
        d = env.to_dict()
        assert d["stream"] == "test"
        assert d["phase"] == "STREAMING"
        assert d["status"] == "working"
        assert d["data"] == {}
        assert d["meta"] == {}
        assert d["message"] == ""

    def test_with_data(self):
        """Data and meta are properly serialized."""
        env = SSEEnvelope(
            stream="auto-train",
            phase="TRAIN",
            status="working",
            data={"step": 1, "loss": 0.123},
            meta={"epoch": 5, "elapsed_ms": 1500},
            message="loss=0.1230",
        )
        d = env.to_dict()
        assert d["data"]["step"] == 1
        assert d["data"]["loss"] == 0.123
        assert d["meta"]["epoch"] == 5
        assert d["message"] == "loss=0.1230"

    def test_status_enum(self):
        """StreamStatus enum is resolved to string in to_dict."""
        env = SSEEnvelope(stream="chat", phase="STREAMING", status=StreamStatus.WORKING)
        assert env.to_dict()["status"] == "working"

    def test_status_string(self):
        """Plain string status works too."""
        env = SSEEnvelope(stream="chat", phase="STREAMING", status="complete")
        assert env.to_dict()["status"] == "complete"


class TestSSEEventFunction:
    """The sse_event() builder function."""

    def test_produces_data_line(self):
        """Output starts with 'data: ' and ends with double newline."""
        result = sse_event(stream="chat", phase="STREAMING", status="working")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_round_trip_json(self):
        """Parsed JSON has all expected fields."""
        raw = sse_event(
            stream="auto-train",
            phase="TRAIN",
            status="working",
            data={"step": 42},
            meta={"elapsed_ms": 100},
            message="step=42",
        )
        payload = json.loads(raw[6:].strip())
        assert payload["stream"] == "auto-train"
        assert payload["phase"] == "TRAIN"
        assert payload["status"] == "working"
        assert payload["data"]["step"] == 42
        assert payload["meta"]["elapsed_ms"] == 100
        assert payload["message"] == "step=42"

    def test_none_data(self):
        """None data defaults to empty dict."""
        raw = sse_event(stream="chat", phase="STREAMING", status="complete", data=None)
        payload = json.loads(raw[6:].strip())
        assert payload["data"] == {}


class TestSSEToken:
    """Token streaming helper."""

    def test_token_working(self):
        """A token event has status working and token in data."""
        raw = sse_token("chat", "Hello")
        payload = json.loads(raw[6:].strip())
        assert payload["phase"] == "STREAMING"
        assert payload["status"] == "working"
        assert payload["data"]["token"] == "Hello"
        assert payload["meta"] == {}

    def test_token_complete(self):
        """done=True produces status complete and empty token."""
        raw = sse_token("chat", "", done=True)
        payload = json.loads(raw[6:].strip())
        assert payload["phase"] == "STREAMING"
        assert payload["status"] == "complete"
        assert payload["data"]["token"] == ""

    def test_elapsed_ms_on_done(self):
        """elapsed_ms is included in meta on done events."""
        raw = sse_token("chat", "", done=True, elapsed_ms=1234.5)
        payload = json.loads(raw[6:].strip())
        assert payload["meta"]["elapsed_ms"] == 1234.5
        assert payload["status"] == "complete"

    def test_no_elapsed_on_token(self):
        """elapsed_ms is NOT included in working token events."""
        raw = sse_token("chat", "Hello", elapsed_ms=500)
        payload = json.loads(raw[6:].strip())
        assert "elapsed_ms" not in payload["meta"]
        assert payload["status"] == "working"


class TestSSEError:
    """Error event helper."""

    def test_error_shape(self):
        """Error event includes error field in data."""
        raw = sse_error("chat", "GENERATE", "Something went wrong")
        payload = json.loads(raw[6:].strip())
        assert payload["phase"] == "GENERATE"
        assert payload["status"] == "error"
        assert payload["data"]["error"] == "Something went wrong"
        assert "Error:" in payload["message"]


class TestSSEComplete:
    """Complete event helper."""

    def test_complete_default_phase(self):
        """Default phase is COMPLETE."""
        raw = sse_complete("auto-train")
        payload = json.loads(raw[6:].strip())
        assert payload["phase"] == "COMPLETE"
        assert payload["status"] == "complete"
        assert payload["data"] == {}

    def test_complete_with_data(self):
        """Complete event carries final data."""
        raw = sse_complete("auto-train", data={"final_loss": 0.01})
        payload = json.loads(raw[6:].strip())
        assert payload["data"]["final_loss"] == 0.01
        assert payload["status"] == "complete"


class TestJsonSafe:
    """JSON serialization helpers."""

    def test_numpy_scalar(self):
        """Numpy scalar types are converted to Python types."""
        from numpy import float32, int64

        assert _json_safe(float32(1.5)) == 1.5
        assert _json_safe(int64(42)) == 42

    def test_numpy_array(self):
        """Numpy arrays are converted to lists."""
        import numpy as np

        arr = np.array([1, 2, 3])
        assert _json_safe(arr) == [1, 2, 3]

    def test_plain_python(self):
        """Plain Python types pass through unchanged."""
        assert _json_safe(42) == 42
        assert _json_safe("hello") == "hello"
        assert _json_safe([1, 2, 3]) == [1, 2, 3]


class TestStreamPhaseEnum:
    """StreamPhase enum values."""

    def test_training_phases(self):
        """All expected training phases exist."""
        expected = [
            "IDLE", "GENERATE_DATA", "DISTILL", "TRAIN",
            "EVALUATE", "DEPLOY", "COMPLETE", "FAILED", "EARLY_STOP",
        ]
        for name in expected:
            assert StreamPhase[name].value == name

    def test_chat_phases(self):
        """Chat/streaming phases."""
        assert StreamPhase.STREAMING.value == "STREAMING"
        assert StreamPhase.ERROR.value == "ERROR"


class TestStreamStatusEnum:
    """StreamStatus enum values."""

    def test_values(self):
        assert StreamStatus.WORKING.value == "working"
        assert StreamStatus.SUCCESS.value == "success"
        assert StreamStatus.ERROR.value == "error"
        assert StreamStatus.COMPLETE.value == "complete"


class TestFrontendCompatibility:
    """Crucial tests — ensures format expected by frontend parsers.

    The frontend parsers (chat-controller.ts, useStreamingChat.tsx, etc.)
    read `envelope.data?.token` and `envelope.status === 'complete'`.
    These tests guarantee that format.
    """

    def test_token_in_data(self):
        """Token is always in data.token, never at top level."""
        raw = sse_token("chat", "Hello")
        payload = json.loads(raw[6:].strip())
        assert "token" in payload["data"]
        assert "token" not in payload  # not at top level

    def test_status_is_string(self):
        """Status is always a plain string, never an enum."""
        raw = sse_event("chat", "STREAMING", "working")
        payload = json.loads(raw[6:].strip())
        assert isinstance(payload["status"], str)

    def test_error_in_data(self):
        """Error message is in data.error, not just at top level."""
        raw = sse_error("chat", "STREAMING", "fail")
        payload = json.loads(raw[6:].strip())
        assert "error" in payload["data"]
        assert payload["data"]["error"] == "fail"

    def test_complete_signal(self):
        """status=complete is the signal for stream end."""
        raw = sse_token("chat", "", done=True)
        payload = json.loads(raw[6:].strip())
        assert payload["status"] == "complete"

    def test_full_round_trip(self):
        """Simulates a full streaming sequence as the frontend sees it."""
        events = [
            sse_token("chat", "The"),
            sse_token("chat", " quick"),
            sse_token("chat", " brown"),
            sse_token("chat", " fox"),
            sse_token("chat", "", done=True),
        ]
        parsed = [json.loads(e[6:].strip()) for e in events]
        tokens = [e["data"]["token"] for e in parsed if e["status"] == "working"]
        done = any(e["status"] == "complete" for e in parsed)
        assert "".join(tokens) == "The quick brown fox"
        assert done is True
