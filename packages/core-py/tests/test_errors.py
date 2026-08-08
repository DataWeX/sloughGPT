"""
Tests for Error Taxonomy (errors.py).
"""

import asyncio

import pytest
from domains.infrastructure.errors import (
    AppError, RecoverableError, FatalError, ValidationError,
    ConfigError, ModelError, ModelOOMError, ModelTimeoutError,
    TaskError, ResourceExhaustedError, NotFoundError, AuthError,
    error_to_sse, classify_exception, emit_error_event,
)


class TestAppError:
    def test_default_fields(self):
        err = AppError()
        assert err.code == "general.error"
        assert err.recoverable is False
        assert err.http_status == 500
        assert err.details == {}
        assert err.cause is None

    def test_custom_fields(self):
        err = AppError(
            "custom msg",
            code="custom.code",
            user_message="User friendly",
            recoverable=True,
            http_status=418,
            details={"key": "val"},
        )
        assert str(err) == "custom msg"
        assert err.code == "custom.code"
        assert err.user_message == "User friendly"
        assert err.recoverable is True
        assert err.http_status == 418
        assert err.details == {"key": "val"}

    def test_to_dict(self):
        err = AppError("test", code="test.code", details={"x": 1})
        d = err.to_dict()
        assert d["code"] == "test.code"
        assert d["message"] == "test"
        assert d["recoverable"] is False
        assert d["details"]["x"] == 1

    def test_to_json(self):
        err = AppError("json", code="json.code")
        import json
        d = json.loads(err.to_json())
        assert d["code"] == "json.code"

    def test_from_exception(self):
        try:
            raise ValueError("bad value")
        except ValueError as e:
            err = AppError.from_exception(e, code="general.test")
            assert err.code == "general.test"
            assert err.message == "bad value"
            assert err.recoverable is False
            assert isinstance(err.cause, ValueError)

    def test_repr(self):
        err = AppError("msg", code="test.code")
        r = repr(err)
        assert "AppError" in r
        assert "test.code" in r

    def test_message_defaults_to_code(self):
        err = AppError(code="silent.error")
        assert err.message == "silent.error"


class TestSubclasses:
    def test_recoverable_error(self):
        err = RecoverableError()
        assert err.recoverable is True
        assert err.http_status == 503
        assert err.code == "general.recoverable"

    def test_fatal_error(self):
        err = FatalError("fatal")
        assert err.recoverable is False
        assert err.http_status == 500

    def test_validation_error(self):
        err = ValidationError("invalid")
        assert err.http_status == 400
        assert err.code == "general.validation"

    def test_config_error(self):
        err = ConfigError("bad config")
        assert err.http_status == 500
        assert err.code == "general.config"

    def test_model_error(self):
        err = ModelError("model fail")
        assert err.http_status == 503
        assert err.code == "model.error"

    def test_model_oom(self):
        err = ModelOOMError()
        assert err.code == "model.oom"
        assert err.recoverable is True

    def test_model_timeout(self):
        err = ModelTimeoutError()
        assert err.code == "model.timeout"
        assert err.recoverable is True

    def test_task_error(self):
        err = TaskError("task fail")
        assert err.recoverable is True

    def test_resource_exhausted(self):
        err = ResourceExhaustedError("too many")
        assert err.http_status == 429
        assert err.recoverable is True

    def test_not_found(self):
        err = NotFoundError("missing")
        assert err.http_status == 404

    def test_auth_error(self):
        err = AuthError("no auth")
        assert err.http_status == 401


class TestErrorToSSE:
    def test_returns_safe_fields(self):
        err = ModelOOMError(details={"secret": "hidden"})
        sse = error_to_sse(err)
        assert sse["code"] == "model.oom"
        assert sse["user_message"] == err.user_message
        assert sse["recoverable"] is True
        assert "details" not in sse
        assert "message" not in sse

    def test_no_traceback_in_sse(self):
        err = AppError("secret_tb")
        sse = error_to_sse(err)
        assert "traceback" not in str(sse)


class TestClassifyException:
    def test_app_error_passthrough(self):
        err = ModelError("test")
        assert classify_exception(err) is err

    def test_timeout_becomes_model_timeout(self):
        err = classify_exception(TimeoutError("timed out"))
        assert isinstance(err, ModelTimeoutError)
        assert err.code == "model.timeout"

    def test_memory_error_becomes_oom(self):
        err = classify_exception(MemoryError("no mem"))
        assert isinstance(err, ModelOOMError)

    def test_connection_error_becomes_recoverable(self):
        err = classify_exception(ConnectionRefusedError("refused"))
        assert isinstance(err, RecoverableError)
        assert err.code == "network.error"

    def test_connection_reset_is_recoverable(self):
        err = classify_exception(ConnectionResetError("reset"))
        assert isinstance(err, RecoverableError)

    def test_file_not_found_becomes_not_found(self):
        err = classify_exception(FileNotFoundError("no file"))
        assert isinstance(err, NotFoundError)

    def test_permission_error_becomes_auth(self):
        err = classify_exception(PermissionError("denied"))
        assert isinstance(err, AuthError)

    def test_value_error_becomes_validation(self):
        err = classify_exception(ValueError("bad"))
        assert isinstance(err, ValidationError)

    def test_unknown_exception_falls_back(self):
        err = classify_exception(RuntimeError("weird"))
        assert isinstance(err, AppError)
        assert err.code == "general.unhandled"
        assert err.recoverable is False

    def test_cause_preserved(self):
        try:
            raise ValueError("root cause")
        except ValueError as e:
            try:
                raise RuntimeError("wrapper") from e
            except RuntimeError as wrapper:
                err = classify_exception(wrapper)
                assert err.cause is wrapper


class TestEmitErrorEvent:
    async def test_emits_via_running_loop(self):
        emit_error_event(ModelError("boom"))
        await asyncio.sleep(0.01)

    def test_emits_silently_when_bus_unavailable(self, monkeypatch):
        import domains.infrastructure.event_bus as eb

        monkeypatch.setattr(
            eb, "get_event_bus",
            lambda: (_ for _ in ()).throw(RuntimeError("no bus")),
        )
        emit_error_event(ModelError("boom"))

    def test_emits_sync_without_running_loop(self, monkeypatch):
        import domains.infrastructure.event_bus as eb

        calls = []

        class _Bus:
            def emit(self, *args, **kwargs):
                calls.append(("emit", args, kwargs))

            def emit_sync(self, *args, **kwargs):
                calls.append(("emit_sync", args, kwargs))

        monkeypatch.setattr(eb, "get_event_bus", lambda: _Bus())
        emit_error_event(ModelError("boom"))
        assert calls, "expected emit_sync call"
        assert calls[0][0] == "emit_sync"
        assert calls[0][1][0] == "error.raised"
