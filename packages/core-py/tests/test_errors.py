"""
Tests for domains/infrastructure/errors.py — error taxonomy and classification.

Covers:
    - AppError base class (defaults, to_dict, to_json, from_exception)
    - All concrete error subclasses (defaults, inheritance)
    - classify_exception mapping (TimeoutError, MemoryError, ConnectionError, etc.)
    - emit_error_event (fire-and-forget, no crash)
    - error_to_sse conversion
"""

import json
import sys
from pathlib import Path
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.errors import (
    AppError,
    RecoverableError,
    FatalError,
    ValidationError,
    ConfigError,
    ModelError,
    ModelOOMError,
    ModelTimeoutError,
    TaskError,
    ResourceExhaustedError,
    NotFoundError,
    AuthError,
    classify_exception,
    emit_error_event,
    error_to_sse,
)


# ── AppError base class ──────────────────────────────────────────────


class TestAppError:
    def test_defaults(self):
        e = AppError("something")
        assert e.message == "something"
        assert e.code == "general.error"
        assert e.user_message == "Something went wrong."
        assert e.recoverable is False
        assert e.http_status == 500
        assert e.details == {}

    def test_custom_fields(self):
        e = AppError(
            "boom",
            code="custom.code",
            user_message="user sees this",
            recoverable=True,
            http_status=418,
            details={"k": "v"},
        )
        assert e.code == "custom.code"
        assert e.user_message == "user sees this"
        assert e.recoverable is True
        assert e.http_status == 418
        assert e.details == {"k": "v"}

    def test_message_falls_back_to_code(self):
        e = AppError(code="my.code")
        assert e.message == "my.code"

    def test_to_dict(self):
        e = AppError("msg", code="c1", http_status=400)
        d = e.to_dict()
        assert d["code"] == "c1"
        assert d["message"] == "msg"
        assert d["http_status"] == 400
        assert d["recoverable"] is False

    def test_to_json(self):
        e = AppError("msg", code="c1")
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["code"] == "c1"

    def test_repr(self):
        e = AppError("msg", code="c1")
        r = repr(e)
        assert "AppError" in r
        assert "c1" in r
        assert "msg" in r

    def test_is_exception(self):
        e = AppError("x")
        assert isinstance(e, Exception)

    def test_from_exception(self):
        original = ValueError("bad value")
        e = AppError.from_exception(original, code="test.error", user_message="User msg")
        assert e.code == "test.error"
        assert e.user_message == "User msg"
        assert e.cause is original
        assert "bad value" in e.message

    def test_cause_stored(self):
        cause = RuntimeError("root")
        e = AppError("wrapper", cause=cause)
        assert e.cause is cause


# ── Concrete error subclasses ─────────────────────────────────────────


class TestConcreteErrors:
    def test_recoverable(self):
        e = RecoverableError("timeout")
        assert e.recoverable is True
        assert e.http_status == 503

    def test_fatal(self):
        e = FatalError("crash")
        assert e.recoverable is False
        assert e.http_status == 500

    def test_validation(self):
        e = ValidationError("bad input")
        assert e.http_status == 400

    def test_config(self):
        e = ConfigError("missing key")
        assert e.http_status == 500

    def test_model(self):
        e = ModelError("NaN weights")
        assert e.http_status == 503

    def test_model_oom(self):
        e = ModelOOMError("out of memory")
        assert e.recoverable is True
        assert e.http_status == 503

    def test_model_timeout(self):
        e = ModelTimeoutError("timed out")
        assert e.recoverable is True

    def test_task(self):
        e = TaskError("queue error")
        assert e.recoverable is True

    def test_resource_exhausted(self):
        e = ResourceExhaustedError("rate limit")
        assert e.http_status == 429
        assert e.recoverable is True

    def test_not_found(self):
        e = NotFoundError("missing")
        assert e.http_status == 404

    def test_auth(self):
        e = AuthError("denied")
        assert e.http_status == 401

    def test_all_inherit_app_error(self):
        for cls in [
            RecoverableError, FatalError, ValidationError, ConfigError,
            ModelError, ModelOOMError, ModelTimeoutError, TaskError,
            ResourceExhaustedError, NotFoundError, AuthError,
        ]:
            assert issubclass(cls, AppError)

    def test_model_oom_inherits_model(self):
        assert issubclass(ModelOOMError, ModelError)

    def test_model_timeout_inherits_model(self):
        assert issubclass(ModelTimeoutError, ModelError)


# ── classify_exception ───────────────────────────────────────────────


class TestClassifyException:
    def test_app_error_passthrough(self):
        orig = NotFoundError("already classified")
        result = classify_exception(orig)
        assert result is orig

    def test_timeout_error(self):
        result = classify_exception(TimeoutError("timed out"))
        assert isinstance(result, ModelTimeoutError)
        assert result.cause is not None

    def test_memory_error(self):
        result = classify_exception(MemoryError("oom"))
        assert isinstance(result, ModelOOMError)

    def test_connection_error(self):
        for exc_cls in [ConnectionError, ConnectionRefusedError, ConnectionResetError]:
            result = classify_exception(exc_cls("refused"))
            assert isinstance(result, RecoverableError)
            assert result.code == "network.error"

    def test_file_not_found(self):
        result = classify_exception(FileNotFoundError("gone"))
        assert isinstance(result, NotFoundError)

    def test_permission_error(self):
        result = classify_exception(PermissionError("no access"))
        assert isinstance(result, AuthError)

    def test_value_error(self):
        result = classify_exception(ValueError("bad value"))
        assert isinstance(result, ValidationError)

    def test_unknown_exception(self):
        result = classify_exception(RuntimeError("something else"))
        assert isinstance(result, AppError)
        assert result.code == "general.unhandled"

    def test_preserves_message(self):
        result = classify_exception(TimeoutError("specific timeout msg"))
        assert "specific timeout msg" in result.message


# ── emit_error_event ──────────────────────────────────────────────────


class TestEmitErrorEvent:
    def test_does_not_crash(self):
        e = AppError("test")
        emit_error_event(e, source="test")
        # Should not raise

    def test_does_not_crash_without_event_bus(self):
        import sys
        real = sys.modules.get("domains.infrastructure.event_bus")
        sys.modules["domains.infrastructure.event_bus"] = None
        try:
            e = AppError("test")
            emit_error_event(e, source="test")
        finally:
            if real is not None:
                sys.modules["domains.infrastructure.event_bus"] = real
            else:
                sys.modules.pop("domains.infrastructure.event_bus", None)


# ── error_to_sse ──────────────────────────────────────────────────────


class TestErrorToSse:
    def test_basic(self):
        e = NotFoundError("missing", user_message="Not found.")
        sse = error_to_sse(e)
        assert sse["code"] == "resource.not_found"
        assert sse["user_message"] == "Not found."
        assert sse["recoverable"] is False
        assert sse["http_status"] == 404

    def test_no_stack_trace(self):
        e = AppError("err", details={"secret": "value"})
        sse = error_to_sse(e)
        assert "details" not in sse
        assert "traceback" not in sse
        assert "secret" not in sse

    def test_recoverable(self):
        e = ResourceExhaustedError("slow down")
        sse = error_to_sse(e)
        assert sse["recoverable"] is True
