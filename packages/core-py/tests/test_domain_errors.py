"""Tests for the domain error hierarchies (domains.errors + infrastructure.errors)."""

import json

import pytest

from domains import errors as domain_errors
from domains.infrastructure import errors as infra_errors


class TestDomainErrors:
    def test_slough_gpt_domain_error_defaults(self):
        err = domain_errors.SloughGPTDomainError("boom")
        assert err.http_status == 500
        assert err.code == "domain_error"
        assert isinstance(err, Exception)

    def test_invalid_generation_input_defaults(self):
        err = domain_errors.InvalidGenerationInputError("bad input")
        assert err.http_status == 422
        assert err.code == "invalid_generation_input"

    def test_empty_prompt_default_message(self):
        err = domain_errors.EmptyPromptError()
        assert str(err) == "prompt must not be empty"
        assert err.code == "empty_prompt"
        assert isinstance(err, domain_errors.InvalidGenerationInputError)

    def test_empty_prompt_custom_message(self):
        err = domain_errors.EmptyPromptError("give me text")
        assert str(err) == "give me text"

    def test_require_non_empty_strips(self):
        assert domain_errors.require_non_empty_prompt("  hello  ") == "hello"

    def test_require_non_empty_preserves_inner_ws(self):
        assert domain_errors.require_non_empty_prompt("a b") == "a b"

    def test_require_non_empty_rejects_non_string(self):
        with pytest.raises(domain_errors.InvalidGenerationInputError):
            domain_errors.require_non_empty_prompt(123)

    def test_require_non_empty_rejects_blank(self):
        with pytest.raises(domain_errors.EmptyPromptError):
            domain_errors.require_non_empty_prompt("   ")

    def test_custom_field_name_in_message(self):
        with pytest.raises(domain_errors.EmptyPromptError) as excinfo:
            domain_errors.require_non_empty_prompt("", field_name="content")
        assert "content" in str(excinfo.value)

    def test_require_non_empty_returns_stripped_with_field(self):
        assert domain_errors.require_non_empty_prompt(" x ", field_name="q") == "x"


class TestAppError:
    def test_defaults(self):
        err = infra_errors.AppError("something")
        assert err.code == "general.error"
        assert err.recoverable is False
        assert err.user_message == "Something went wrong."
        assert err.http_status == 500
        assert err.message == "something"
        assert err.details == {}

    def test_str_uses_message(self):
        err = infra_errors.AppError("custom message")
        assert str(err) == "custom message"

    def test_str_falls_back_to_code(self):
        err = infra_errors.AppError()
        assert str(err) == "general.error"

    def test_constructor_overrides(self):
        err = infra_errors.AppError(
            "m", code="custom.code", user_message="friendly",
            recoverable=True, http_status=429, details={"k": "v"},
        )
        assert err.code == "custom.code"
        assert err.user_message == "friendly"
        assert err.recoverable is True
        assert err.http_status == 429
        assert err.details == {"k": "v"}

    def test_cause_stored(self):
        cause = ValueError("inner")
        err = infra_errors.AppError("m", cause=cause)
        assert err.cause is cause

    def test_to_dict(self):
        err = infra_errors.ValidationError("bad", details={"field": "x"})
        d = err.to_dict()
        assert d["code"] == "general.validation"
        assert d["message"] == "bad"
        assert d["user_message"] == "Invalid request."
        assert d["recoverable"] is False
        assert d["http_status"] == 400
        assert d["details"] == {"field": "x"}

    def test_to_json(self):
        err = infra_errors.ValidationError("bad")
        assert json.loads(err.to_json())["code"] == "general.validation"

    def test_repr(self):
        err = infra_errors.ValidationError("bad")
        assert "ValidationError" in repr(err)
        assert "general.validation" in repr(err)
        assert "bad" in repr(err)

    def test_from_exception_wraps(self):
        err = infra_errors.AppError.from_exception(ValueError("boom"))
        assert err.message == "boom"
        assert err.code == "general.unhandled"
        assert err.cause is not None
        assert isinstance(err.cause, ValueError)
        assert err.recoverable is False


class TestErrorSubclasses:
    def test_recoverable_error(self):
        err = infra_errors.RecoverableError("m")
        assert err.recoverable is True
        assert err.http_status == 503
        assert err.code == "general.recoverable"

    def test_fatal_error(self):
        err = infra_errors.FatalError("m")
        assert err.recoverable is False
        assert err.http_status == 500
        assert err.code == "general.fatal"

    def test_validation_error(self):
        err = infra_errors.ValidationError("m")
        assert err.http_status == 400
        assert err.code == "general.validation"
        assert err.recoverable is False

    def test_config_error(self):
        err = infra_errors.ConfigError("m")
        assert err.code == "general.config"

    def test_model_error(self):
        err = infra_errors.ModelError("m")
        assert err.http_status == 503
        assert err.code == "model.error"

    def test_model_oom_recoverable(self):
        err = infra_errors.ModelOOMError("m")
        assert err.recoverable is True
        assert err.code == "model.oom"
        assert isinstance(err, infra_errors.ModelError)

    def test_model_timeout(self):
        err = infra_errors.ModelTimeoutError("m")
        assert err.recoverable is True
        assert err.code == "model.timeout"

    def test_task_error(self):
        err = infra_errors.TaskError("m")
        assert err.recoverable is True
        assert err.code == "task.error"

    def test_resource_exhausted(self):
        err = infra_errors.ResourceExhaustedError("m")
        assert err.http_status == 429
        assert err.recoverable is True
        assert err.code == "resource.exhausted"

    def test_not_found(self):
        err = infra_errors.NotFoundError("m")
        assert err.http_status == 404
        assert err.code == "resource.not_found"

    def test_auth_error(self):
        err = infra_errors.AuthError("m")
        assert err.http_status == 401
        assert err.code == "auth.error"


class TestClassification:
    def test_app_error_passthrough(self):
        err = infra_errors.ValidationError("bad")
        assert infra_errors.classify_exception(err) is err

    def test_timeout_becomes_model_timeout(self):
        err = infra_errors.classify_exception(TimeoutError("slow"))
        assert isinstance(err, infra_errors.ModelTimeoutError)
        assert err.recoverable is True
        assert err.details == {"original_type": "TimeoutError"}

    def test_memory_error_becomes_oom(self):
        err = infra_errors.classify_exception(MemoryError("oom"))
        assert isinstance(err, infra_errors.ModelOOMError)

    def test_connection_error_becomes_recoverable(self):
        err = infra_errors.classify_exception(ConnectionError("offline"))
        assert isinstance(err, infra_errors.RecoverableError)
        assert err.code == "network.error"

    def test_file_not_found_becomes_not_found(self):
        err = infra_errors.classify_exception(FileNotFoundError("missing.txt"))
        assert isinstance(err, infra_errors.NotFoundError)

    def test_permission_error_becomes_auth(self):
        err = infra_errors.classify_exception(PermissionError("denied"))
        assert isinstance(err, infra_errors.AuthError)

    def test_value_error_becomes_validation(self):
        err = infra_errors.classify_exception(ValueError("nope"))
        assert isinstance(err, infra_errors.ValidationError)

    def test_unknown_exception_becomes_app_error(self):
        err = infra_errors.classify_exception(RuntimeError("weird"))
        assert isinstance(err, infra_errors.AppError)
        assert err.code == "general.unhandled"
        assert err.recoverable is False


class TestHelpers:
    def test_error_to_sse(self):
        err = infra_errors.ModelOOMError("boom")
        d = infra_errors.error_to_sse(err)
        assert d == {
            "code": "model.oom",
            "user_message": err.user_message,
            "recoverable": True,
            "http_status": 503,
        }
        assert "message" not in d
        assert "details" not in d

    def test_emit_error_event_noop(self):
        err = infra_errors.ValidationError("bad")
        assert infra_errors.emit_error_event(err) is None
