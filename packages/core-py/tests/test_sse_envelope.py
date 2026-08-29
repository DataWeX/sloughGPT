"""Tests for domains.api.sse_envelope — StreamPhase, StreamStatus, SSEEnvelope, sse_event, sse_error, sse_complete, sse_token, _json_safe."""

import json
import numpy as np
from domains.api.sse_envelope import (
    StreamPhase, StreamStatus, SSEEnvelope, sse_event,
    sse_error, sse_complete, sse_token, _json_safe,
)


class TestStreamPhase:
    def test_all_members(self):
        assert len(StreamPhase) >= 10

    def test_values(self):
        assert StreamPhase.IDLE.value == "IDLE"
        assert StreamPhase.TRAIN.value == "TRAIN"
        assert StreamPhase.COMPLETE.value == "COMPLETE"
        assert StreamPhase.FAILED.value == "FAILED"


class TestStreamStatus:
    def test_all_members(self):
        assert len(StreamStatus) == 4

    def test_values(self):
        assert StreamStatus.WORKING.value == "working"
        assert StreamStatus.SUCCESS.value == "success"
        assert StreamStatus.ERROR.value == "error"
        assert StreamStatus.COMPLETE.value == "complete"


class TestSSEEnvelope:
    def test_fields(self):
        env = SSEEnvelope(
            stream="auto-train", phase="TRAIN", status=StreamStatus.WORKING,
            message="training", data={"loss": 0.5}, meta={"step": 1},
        )
        assert env.stream == "auto-train"
        assert env.status == StreamStatus.WORKING

    def test_to_dict(self):
        env = SSEEnvelope(
            stream="auto-train", phase="TRAIN", status=StreamStatus.WORKING,
            message="training", data={"loss": 0.5}, meta={"step": 1},
        )
        d = env.to_dict()
        assert d["stream"] == "auto-train"
        assert d["status"] == "working"
        assert d["data"]["loss"] == 0.5

    def test_to_dict_string_status(self):
        env = SSEEnvelope(
            stream="chat", phase="STREAMING", status="working",
        )
        d = env.to_dict()
        assert d["status"] == "working"


class TestSSEEvent:
    def test_returns_string(self):
        result = sse_event("chat", "STREAMING", StreamStatus.WORKING, message="hi")
        assert isinstance(result, str)
        assert result.startswith("data: ")

    def test_json_content(self):
        result = sse_event("chat", "STREAMING", StreamStatus.WORKING, data={"token": "hi"})
        payload = result.removeprefix("data: ").strip()
        parsed = json.loads(payload)
        assert parsed["stream"] == "chat"
        assert parsed["data"]["token"] == "hi"

    def test_string_status(self):
        result = sse_event("chat", "STREAMING", "working", message="ok")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["status"] == "working"

    def test_newline_terminated(self):
        result = sse_event("chat", "STREAMING", "working")
        assert result.endswith("\n\n")


class TestSSEError:
    def test_error_event(self):
        result = sse_error("chat", "STREAMING", "timeout")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["status"] == "error"
        assert parsed["data"]["error"] == "timeout"
        assert "Error: timeout" in parsed["message"]

    def test_error_with_meta(self):
        result = sse_error("eval", "TRAIN", "oom", meta={"step": 5})
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["meta"]["step"] == 5

    def test_error_with_code(self):
        result = sse_error("chat", "TIMEOUT", "stalled", code="MODEL_TIMEOUT", http_status=504)
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["data"]["code"] == "MODEL_TIMEOUT"
        assert parsed["data"]["http_status"] == 504
        assert parsed["data"]["error"] == "stalled"

    def test_error_code_optional(self):
        result = sse_error("chat", "IDLE", "loading")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert "code" not in parsed["data"]
        assert "http_status" not in parsed["data"]

    def test_error_with_code_and_meta(self):
        result = sse_error("chat", "IDLE", "oom", code="MODEL_OOM", http_status=503, meta={"mem_pct": 97})
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["data"]["code"] == "MODEL_OOM"
        assert parsed["data"]["http_status"] == 503
        assert parsed["meta"]["mem_pct"] == 97


class TestSSEComplete:
    def test_complete_event(self):
        result = sse_complete("chat")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["status"] == "complete"
        assert parsed["phase"] == "COMPLETE"
        assert parsed["message"] == "Done"

    def test_complete_custom_phase(self):
        result = sse_complete("chat", phase="STREAMING")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["phase"] == "STREAMING"

    def test_complete_with_data(self):
        result = sse_complete("chat", data={"tokens": 10}, message="finished")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["data"]["tokens"] == 10
        assert parsed["message"] == "finished"


class TestSSEToken:
    def test_token_working(self):
        result = sse_token("chat", "Hello")
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["phase"] == "STREAMING"
        assert parsed["status"] == "working"
        assert parsed["data"]["token"] == "Hello"

    def test_token_done(self):
        result = sse_token("chat", "", done=True, elapsed_ms=123.4)
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["status"] == "complete"
        assert parsed["meta"]["elapsed_ms"] == 123.4

    def test_token_done_no_elapsed(self):
        result = sse_token("chat", "", done=True)
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["status"] == "complete"
        assert "elapsed_ms" not in parsed["meta"]

    def test_token_with_meta(self):
        result = sse_token("chat", "world", meta={"session": "abc"})
        parsed = json.loads(result.removeprefix("data: ").strip())
        assert parsed["meta"]["session"] == "abc"


class TestJsonSafe:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _json_safe(arr) == [1.0, 2.0, 3.0]

    def test_numpy_scalar(self):
        val = np.float32(3.14)
        result = _json_safe(val)
        assert isinstance(result, float)

    def test_passthrough(self):
        assert _json_safe(42) == 42
        assert _json_safe("hello") == "hello"
        assert _json_safe(None) is None
