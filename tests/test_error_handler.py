"""Tests for the unified error taxonomy and modular error handler."""

from __future__ import annotations

from domains.infrastructure.errors import (
    ERROR_REGISTRY,
    AppError,
    AuthError,
    ConfigError,
    ErrorCode,
    ModelError,
    ModelOOMError,
    ModelTimeoutError,
    NotFoundError,
    ResourceExhaustedError,
    TaskError,
    TimeoutAppError,
    ValidationError,
    classify_exception,
    get_error_info,
)
from infrastructure.error_handler import (
    APIErrorHandler,
    AuthErrorHandler,
    DefaultErrorHandler,
    DomainErrorHandler,
    InferenceErrorHandler,
    ResourceErrorHandler,
    TrainingErrorHandler,
    ValidationErrorHandler,
    create_default_error_handler,
)

# ── ErrorCode Registry Tests ──


class TestErrorCodeRegistry:
    def test_all_codes_have_entries(self):
        for code in ErrorCode:
            assert code in ERROR_REGISTRY, f"Missing registry entry for {code}"

    def test_registry_entry_shape(self):
        for code, entry in ERROR_REGISTRY.items():
            assert len(entry) == 4, f"Bad entry for {code}: {entry}"
            class_name, http_status, recoverable, user_msg = entry
            assert isinstance(http_status, int)
            assert isinstance(recoverable, bool)
            assert isinstance(user_msg, str)
            assert 100 <= http_status < 600

    def test_get_error_info_valid(self):
        info = get_error_info("E_NOT_FOUND")
        assert info is not None
        class_name, status, recoverable, msg = info
        assert status == 404
        assert recoverable is False

    def test_get_error_info_invalid(self):
        assert get_error_info("E_NONEXISTENT") is None

    def test_error_codes_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value.startswith("E_")


# ── AppError Base Tests ──


class TestAppError:
    def test_default_code(self):
        err = AppError("test")
        assert err.code == "E_INTERNAL"

    def test_custom_code(self):
        err = AppError("test", code="E_CUSTOM")
        assert err.code == "E_CUSTOM"

    def test_to_dict(self):
        err = AppError("test", code="E_TEST", user_message="Test msg")
        d = err.to_dict()
        assert d["code"] == "E_TEST"
        assert d["message"] == "test"
        assert d["user_message"] == "Test msg"

    def test_to_http_response(self):
        err = AppError("test", code="E_TEST", user_message="Test msg")
        resp = err.to_http_response()
        assert resp["error"] == "Test msg"
        assert resp["code"] == "E_TEST"
        assert "message" not in resp

    def test_to_sse(self):
        err = AppError("test", code="E_TEST", user_message="Test msg", recoverable=True)
        sse = err.to_sse()
        assert sse["code"] == "E_TEST"
        assert sse["recoverable"] is True
        assert "message" not in sse

    def test_from_exception(self):
        orig = ValueError("bad value")
        err = AppError.from_exception(orig, code="E_VAL")
        assert err.cause is orig
        assert err.code == "E_VAL"
        assert "bad value" in err.message

    def test_source_field(self):
        err = AppError("test", source="router.chat")
        assert err.source == "router.chat"


# ── Subclass Code Tests ──


class TestSubclassCodes:
    def test_not_found_error(self):
        assert NotFoundError.code == "E_NOT_FOUND"

    def test_auth_error(self):
        assert AuthError.code == "E_AUTH_MISSING"

    def test_validation_error(self):
        assert ValidationError.code == "E_BAD_REQUEST"

    def test_model_oom(self):
        assert ModelOOMError.code == "E_MODEL_OOM"

    def test_model_timeout(self):
        assert ModelTimeoutError.code == "E_MODEL_TIMEOUT"

    def test_resource_exhausted(self):
        assert ResourceExhaustedError.code == "E_RATE_LIMITED"

    def test_config_error(self):
        assert ConfigError.code == "E_CONFIG"

    def test_task_error(self):
        assert TaskError.code == "E_TASK_ERROR"


# ── classify_exception Tests ──


class TestClassifyException:
    def test_already_app_error(self):
        err = NotFoundError("gone")
        assert classify_exception(err) is err

    def test_timeout_error(self):
        err = classify_exception(TimeoutError("too slow"))
        assert isinstance(err, TimeoutAppError)
        assert err.code == "E_TIMEOUT"

    def test_memory_error(self):
        err = classify_exception(MemoryError())
        assert isinstance(err, ModelOOMError)
        assert err.code == "E_MODEL_OOM"

    def test_connection_error(self):
        err = classify_exception(ConnectionRefusedError())
        assert isinstance(err, AppError)
        assert err.code == "E_NETWORK"

    def test_file_not_found(self):
        err = classify_exception(FileNotFoundError("/tmp/foo"))
        assert isinstance(err, NotFoundError)
        assert err.code == "E_NOT_FOUND"

    def test_permission_error(self):
        err = classify_exception(PermissionError())
        assert isinstance(err, AuthError)
        assert err.code == "E_AUTH_MISSING"

    def test_value_error(self):
        err = classify_exception(ValueError("bad"))
        assert isinstance(err, ValidationError)
        assert err.code == "E_BAD_REQUEST"

    def test_key_error(self):
        err = classify_exception(KeyError("missing"))
        assert isinstance(err, NotFoundError)

    def test_unknown_exception(self):
        err = classify_exception(RuntimeError("something"))
        assert isinstance(err, AppError)
        assert err.code == "E_UNHANDLED"


# ── DomainErrorHandler Base Tests ──


class TestDomainErrorHandler:
    def test_can_handle_returns_false_for_unmatched_error(self):
        handler = DomainErrorHandler(domain="test", handled_errors=(NotFoundError,))
        error = AuthError("test")
        assert handler.can_handle(error) is False

    def test_can_handle_returns_true_for_matched_error(self):
        handler = DomainErrorHandler(domain="test", handled_errors=(AuthError,))
        error = AuthError("test")
        assert handler.can_handle(error) is True

    def test_handle_returns_error_response_shape(self):
        handler = DomainErrorHandler(domain="test", handled_errors=(AppError,))
        error = AppError("test error", code="test.code", user_message="Test message")
        result = handler.handle(error)
        assert "error" in result or "code" in result


# ── TrainingErrorHandler Tests ──


class TestTrainingErrorHandler:
    def test_handles_task_error(self):
        handler = TrainingErrorHandler()
        error = TaskError("training failed")
        assert handler.can_handle(error) is True

    def test_does_not_handle_model_error(self):
        handler = TrainingErrorHandler()
        error = ModelError("model failed")
        assert handler.can_handle(error) is False

    def test_enriches_task_error_message(self):
        handler = TrainingErrorHandler()
        error = TaskError("training failed")
        handler.handle(error)
        assert "GPU memory" in error.user_message or "dataset" in error.user_message


# ── InferenceErrorHandler Tests ──


class TestInferenceErrorHandler:
    def test_handles_model_errors(self):
        handler = InferenceErrorHandler()
        assert handler.can_handle(ModelError("test")) is True
        assert handler.can_handle(ModelOOMError("test")) is True
        assert handler.can_handle(ModelTimeoutError("test")) is True

    def test_does_not_handle_task_error(self):
        handler = InferenceErrorHandler()
        assert handler.can_handle(TaskError("test")) is False

    def test_oom_includes_suggestion(self):
        handler = InferenceErrorHandler()
        error = ModelOOMError("out of memory")
        handler.handle(error)
        assert "smaller model" in error.details.get("suggestion", "").lower() or "batch" in error.details.get("suggestion", "").lower()

    def test_timeout_includes_suggestion(self):
        handler = InferenceErrorHandler()
        error = ModelTimeoutError("timed out")
        handler.handle(error)
        assert "shorter prompt" in error.details.get("suggestion", "").lower() or "max_tokens" in error.details.get("suggestion", "").lower()


# ── AuthErrorHandler Tests ──


class TestAuthErrorHandler:
    def test_handles_auth_errors(self):
        handler = AuthErrorHandler()
        assert handler.can_handle(AuthError("test")) is True

    def test_does_not_handle_other_errors(self):
        handler = AuthErrorHandler()
        assert handler.can_handle(NotFoundError("test")) is False

    def test_401_message(self):
        handler = AuthErrorHandler()
        error = AuthError("unauthorized", http_status=401)
        handler.handle(error)
        assert "token" in error.user_message.lower() or "authentication" in error.user_message.lower()

    def test_403_message(self):
        handler = AuthErrorHandler()
        error = AuthError("forbidden", http_status=403)
        handler.handle(error)
        assert "permission" in error.user_message.lower()


# ── ResourceErrorHandler Tests ──


class TestResourceErrorHandler:
    def test_handles_not_found(self):
        handler = ResourceErrorHandler()
        assert handler.can_handle(NotFoundError("test")) is True

    def test_handles_rate_limit(self):
        handler = ResourceErrorHandler()
        assert handler.can_handle(ResourceExhaustedError("test")) is True

    def test_rate_limit_includes_retry_after(self):
        handler = ResourceErrorHandler()
        error = ResourceExhaustedError("rate limited")
        handler.handle(error)
        assert "retry_after" in error.details


# ── ValidationErrorHandler Tests ──


class TestValidationErrorHandler:
    def test_handles_validation_error(self):
        handler = ValidationErrorHandler()
        assert handler.can_handle(ValidationError("test")) is True

    def test_handles_config_error(self):
        handler = ValidationErrorHandler()
        assert handler.can_handle(ConfigError("test")) is True

    def test_does_not_handle_model_error(self):
        handler = ValidationErrorHandler()
        assert handler.can_handle(ModelError("test")) is False


# ── DefaultErrorHandler Tests ──


class TestDefaultErrorHandler:
    def test_handles_any_app_error(self):
        handler = DefaultErrorHandler()
        assert handler.can_handle(AppError("test")) is True
        assert handler.can_handle(AuthError("test")) is True
        assert handler.can_handle(ModelError("test")) is True


# ── APIErrorHandler Registry Tests ──


class TestAPIErrorHandler:
    def test_register_returns_self(self):
        handler = APIErrorHandler()
        result = handler.register(TrainingErrorHandler())
        assert result is handler

    def test_register_adds_handler(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        assert handler.handler_count == 1

    def test_register_multiple_handlers(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())
        handler.register(AuthErrorHandler())
        assert handler.handler_count == 3

    def test_domains_property(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())
        assert handler.domains == ["training", "inference"]

    def test_unregister_removes_handler(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())
        handler.unregister(TrainingErrorHandler)
        assert handler.handler_count == 1
        assert "training" not in handler.domains

    def test_find_handler_returns_matching(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())
        error = ModelOOMError("oom")
        found = handler.find_handler(error)
        assert isinstance(found, InferenceErrorHandler)

    def test_find_handler_returns_default_when_no_match(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        error = AuthError("unauthorized")
        found = handler.find_handler(error)
        assert isinstance(found, DefaultErrorHandler)

    def test_handle_dispatches_to_correct_handler(self):
        handler = APIErrorHandler()
        handler.register(TrainingErrorHandler())
        handler.register(InferenceErrorHandler())
        error = ModelTimeoutError("timed out")
        result = handler.handle(error)
        assert isinstance(result, dict)

    def test_handle_classifies_raw_exception(self):
        handler = APIErrorHandler()
        result = handler.handle(TimeoutError("raw timeout"))
        assert isinstance(result, dict)

    def test_handle_to_response_returns_json_response(self):
        from fastapi.responses import JSONResponse
        handler = APIErrorHandler()
        error = NotFoundError("not found")
        response = handler.handle_to_response(error)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 404


# ── Factory Tests ──


class TestCreateDefaultErrorHandler:
    def test_creates_with_all_handlers(self):
        handler = create_default_error_handler()
        assert handler.handler_count == 5
        assert "training" in handler.domains
        assert "inference" in handler.domains
        assert "auth" in handler.domains
        assert "resource" in handler.domains
        assert "validation" in handler.domains


# ── Fluent Chaining Tests ──


class TestFluentChaining:
    def test_register_chaining(self):
        handler = APIErrorHandler()
        result = (
            handler
            .register(TrainingErrorHandler())
            .register(InferenceErrorHandler())
            .register(AuthErrorHandler())
        )
        assert result is handler
        assert handler.handler_count == 3
