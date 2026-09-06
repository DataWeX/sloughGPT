"""Tests for AppError hierarchy — codes, recovery policies, serialization."""
from __future__ import annotations

import json

import pytest

from domains.infrastructure.errors import (
    AppError,
    AuthError,
    ConfigError,
    FatalError,
    ModelOOMError,
    ModelError,
    ModelTimeoutError,
    NotFoundError,
    RecoverableError,
    ResourceExhaustedError,
    TaskError,
    ValidationError,
)


class TestAppError:
    def test_default_code(self):
        e = AppError("boom")
        assert e.code == "E_INTERNAL"

    def test_custom_code(self):
        e = AppError("boom", code="custom.code")
        assert e.code == "custom.code"

    def test_message(self):
        e = AppError("something broke")
        assert e.message == "something broke"
        assert str(e) == "something broke"

    def test_empty_message_uses_code(self):
        e = AppError(code="my.code")
        assert str(e) == "my.code"

    def test_user_message(self):
        e = AppError("internal", user_message="Try again later.")
        assert e.user_message == "Try again later."

    def test_default_user_message(self):
        e = AppError("x")
        assert e.user_message == "Something went wrong."

    def test_recoverable_default(self):
        e = AppError("x")
        assert e.recoverable is False

    def test_recoverable_override(self):
        e = AppError("x", recoverable=True)
        assert e.recoverable is True

    def test_http_status_default(self):
        e = AppError("x")
        assert e.http_status == 500

    def test_http_status_override(self):
        e = AppError("x", http_status=400)
        assert e.http_status == 400

    def test_details(self):
        e = AppError("x", details={"key": "val"})
        assert e.details == {"key": "val"}

    def test_details_default_empty(self):
        e = AppError("x")
        assert e.details == {}

    def test_cause(self):
        orig = ValueError("orig")
        e = AppError("wrapped", cause=orig)
        assert e.cause is orig

    def test_to_dict(self):
        e = AppError("msg", code="c1", user_message="um", recoverable=True, http_status=422, details={"a": 1})
        d = e.to_dict()
        assert d["code"] == "c1"
        assert d["message"] == "msg"
        assert d["user_message"] == "um"
        assert d["recoverable"] is True
        assert d["http_status"] == 422
        assert d["details"] == {"a": 1}

    def test_to_json(self):
        e = AppError("msg", code="c1")
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["code"] == "c1"

    def test_repr(self):
        e = AppError("msg", code="c1")
        assert "AppError" in repr(e)
        assert "c1" in repr(e)

    def test_from_exception(self):
        orig = ValueError("bad")
        e = AppError.from_exception(orig, code="wrap.code", user_message="user msg")
        assert e.cause is orig
        assert e.code == "wrap.code"
        assert e.user_message == "user msg"
        assert e.recoverable is False

    def test_is_exception(self):
        e = AppError("x")
        assert isinstance(e, Exception)


class TestConcreteErrors:
    def test_recoverable_error(self):
        e = RecoverableError("retry")
        assert e.recoverable is True

    def test_fatal_error(self):
        e = FatalError("crash")
        assert e.recoverable is False
        assert e.http_status == 500

    def test_validation_error(self):
        e = ValidationError("bad input")
        assert e.http_status == 400

    def test_config_error(self):
        e = ConfigError("missing key")
        assert e.http_status == 500

    def test_model_error(self):
        e = ModelError("oom")
        assert e.code == "E_MODEL_ERROR"

    def test_model_oom_error(self):
        e = ModelOOMError("out of memory")
        assert e.code == "E_MODEL_OOM"
        assert e.recoverable is True

    def test_model_timeout_error(self):
        e = ModelTimeoutError("timed out")
        assert e.code == "E_MODEL_TIMEOUT"

    def test_task_error(self):
        e = TaskError("task failed")
        assert e.code == "E_TASK_ERROR"

    def test_resource_exhausted_error(self):
        e = ResourceExhaustedError("no resources")
        assert e.code == "E_RATE_LIMITED"

    def test_not_found_error(self):
        e = NotFoundError("not found")
        assert e.http_status == 404

    def test_auth_error(self):
        e = AuthError("unauthorized")
        assert e.http_status == 401

    def test_all_inherit_app_error(self):
        for cls in [RecoverableError, FatalError, ValidationError, ConfigError,
                    ModelError, ModelOOMError, ModelTimeoutError, TaskError,
                    ResourceExhaustedError, NotFoundError, AuthError]:
            assert issubclass(cls, AppError)
