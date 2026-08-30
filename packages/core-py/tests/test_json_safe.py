import json

import numpy as np
import pytest

from domains.api.sse_envelope import (
    _json_safe,
    sse_event,
    sse_error,
    sse_complete,
    sse_token,
    SSEEnvelope,
    StreamPhase,
    StreamStatus,
    TRAINING_SEQUENCE,
    CHAT_SEQUENCE,
)


class TestJsonSafe:
    def test_numpy_int(self):
        v = np.int64(42)
        result = _json_safe(v)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        v = np.float64(3.14)
        result = _json_safe(v)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_numpy_bool(self):
        v = np.bool_(True)
        result = _json_safe(v)
        assert result is True

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _json_safe(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _json_safe(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_zeros(self):
        arr = np.zeros(5)
        result = _json_safe(arr)
        assert result == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_numpy_empty_array(self):
        arr = np.array([])
        result = _json_safe(arr)
        assert result == []

    def test_plain_int(self):
        assert _json_safe(42) == 42

    def test_plain_float(self):
        assert _json_safe(2.718) == 2.718

    def test_plain_string(self):
        assert _json_safe("hello") == "hello"

    def test_plain_none(self):
        assert _json_safe(None) is None

    def test_plain_bool(self):
        assert _json_safe(True) is True

    def test_plain_list(self):
        assert _json_safe([1, 2, 3]) == [1, 2, 3]

    def test_plain_dict(self):
        assert _json_safe({"a": 1}) == {"a": 1}

    def test_nested_numpy_via_json_dumps(self):
        data = {"loss": np.float32(0.5), "step": np.int32(10), "tags": np.array([1, 2])}
        raw = json.dumps(data, default=_json_safe)
        result = json.loads(raw)
        assert result["loss"] == pytest.approx(0.5)
        assert result["step"] == 10
        assert result["tags"] == [1, 2]

    def test_custom_object_without_tolist_or_item(self):
        class Custom:
            pass
        c = Custom()
        with pytest.raises((TypeError, ValueError)):
            json.dumps(c, default=_json_safe)

    def test_custom_object_with_tolist(self):
        class HasToList:
            def tolist(self):
                return [10, 20]
        result = _json_safe(HasToList())
        assert result == [10, 20]

    def test_custom_object_with_item(self):
        class HasItem:
            def item(self):
                return 99
        result = _json_safe(HasItem())
        assert result == 99

    def test_numpy_complex(self):
        v = np.complex128(1 + 2j)
        result = _json_safe(v)
        assert isinstance(result, complex)

    def test_numpy_nan(self):
        v = np.float64("nan")
        result = _json_safe(v)
        assert result != result

    def test_numpy_inf(self):
        v = np.float64("inf")
        result = _json_safe(v)
        assert result == float("inf")

    def test_numpy_int8(self):
        result = _json_safe(np.int8(7))
        assert result == 7

    def test_numpy_float32(self):
        result = _json_safe(np.float32(1.5))
        assert isinstance(result, float)

    def test_serialize_full_event(self):
        event_str = sse_event(
            stream="chat",
            phase="STREAMING",
            status="working",
            data={"token": "hi", "loss": np.float64(0.123)},
            meta={"step": np.int64(5)},
            message="",
        )
        assert event_str.startswith("data: ")
        payload = json.loads(event_str.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["stream"] == "chat"
        assert payload["data"]["loss"] == pytest.approx(0.123)
        assert payload["meta"]["step"] == 5


class TestSSEEnvelope:
    def test_to_dict_basic(self):
        e = SSEEnvelope(stream="chat", phase="IDLE", status="working", message="hi")
        d = e.to_dict()
        assert d["stream"] == "chat"
        assert d["phase"] == "IDLE"
        assert d["status"] == "working"
        assert d["message"] == "hi"
        assert d["data"] == {}
        assert d["meta"] == {}
        assert "id" not in d

    def test_to_dict_with_id(self):
        e = SSEEnvelope(stream="x", phase="y", status="complete", id="evt-1")
        d = e.to_dict()
        assert d["id"] == "evt-1"

    def test_to_dict_stream_status_enum(self):
        e = SSEEnvelope(stream="x", phase="y", status=StreamStatus.SUCCESS)
        d = e.to_dict()
        assert d["status"] == "success"

    def test_to_dict_stream_phase_values(self):
        for phase in StreamPhase:
            e = SSEEnvelope(stream="s", phase=phase.value, status="working")
            assert e.to_dict()["phase"] == phase.value

    def test_defaults(self):
        e = SSEEnvelope(stream="s", phase="p", status="w")
        assert e.data == {}
        assert e.meta == {}
        assert e.message == ""
        assert e.id is None


class TestSSEEvent:
    def test_returns_data_line(self):
        result = sse_event("chat", "IDLE", "working")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_json_payload(self):
        result = sse_event("train", "TRAIN", "success", data={"loss": 0.5})
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["stream"] == "train"
        assert payload["data"]["loss"] == 0.5

    def test_with_id(self):
        result = sse_event("chat", "IDLE", "working", id="evt-42")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["id"] == "evt-42"

    def test_with_stream_status_enum(self):
        result = sse_event("c", "IDLE", StreamStatus.COMPLETE)
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["status"] == "complete"


class TestSSEError:
    def test_basic_error(self):
        result = sse_error("chat", "TIMEOUT", "model timeout")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["status"] == "error"
        assert payload["data"]["error"] == "model timeout"
        assert payload["message"] == "Error: model timeout"

    def test_error_with_code(self):
        result = sse_error("chat", "ERR", "bad input", code="INVALID_INPUT")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["data"]["code"] == "INVALID_INPUT"

    def test_error_with_http_status(self):
        result = sse_error("chat", "ERR", "not found", http_status=404)
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["data"]["http_status"] == 404

    def test_error_with_meta(self):
        result = sse_error("chat", "ERR", "fail", meta={"attempt": 3})
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["meta"]["attempt"] == 3

    def test_error_without_optional_fields(self):
        result = sse_error("chat", "ERR", "fail")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert "code" not in payload["data"]
        assert "http_status" not in payload["data"]


class TestSSEComplete:
    def test_defaults(self):
        result = sse_complete("chat")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["status"] == "complete"
        assert payload["phase"] == "COMPLETE"
        assert payload["message"] == "Done"

    def test_custom_phase_and_message(self):
        result = sse_complete("train", phase="DEPLOY", message="deployed")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["phase"] == "DEPLOY"
        assert payload["message"] == "deployed"


class TestSSEToken:
    def test_working_token(self):
        result = sse_token("chat", "Hello")
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["phase"] == "STREAMING"
        assert payload["status"] == "working"
        assert payload["data"]["token"] == "Hello"

    def test_done_token(self):
        result = sse_token("chat", "", done=True)
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["status"] == "complete"

    def test_done_with_elapsed_ms(self):
        result = sse_token("chat", "", done=True, elapsed_ms=123.456)
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["meta"]["elapsed_ms"] == 123.5

    def test_done_without_elapsed_ms(self):
        result = sse_token("chat", "", done=True)
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert "elapsed_ms" not in payload["meta"]

    def test_with_meta(self):
        result = sse_token("chat", "x", meta={"custom": 1})
        payload = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert payload["meta"]["custom"] == 1


class TestConstants:
    def test_training_sequence(self):
        assert TRAINING_SEQUENCE[0] == "IDLE"
        assert TRAINING_SEQUENCE[-1] == "EARLY_STOP"
        assert "GENERATE_DATA" in TRAINING_SEQUENCE

    def test_chat_sequence(self):
        assert CHAT_SEQUENCE[0] == "IDLE"
        assert "STREAMING" in CHAT_SEQUENCE
        assert "COMPLETE" in CHAT_SEQUENCE
        assert "ERROR" in CHAT_SEQUENCE
