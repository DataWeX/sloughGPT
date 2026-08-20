"""Tests for domains.api.sse_envelope — StreamPhase, StreamStatus, SSEEnvelope, sse_event."""

import json
from domains.api.sse_envelope import (
    StreamPhase, StreamStatus, SSEEnvelope, sse_event,
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
