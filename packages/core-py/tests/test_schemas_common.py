"""
Tests for schemas/common.py — unified error handling infrastructure.

Covers:
    - Correlation ID context vars (set/get)
    - success_response shape
    - error_response shape and correlation_id auto-resolution
    - wrap_controller_result success and error paths
    - raise_error exception types and status codes
    - classify_and_raise classification and fallback
    - safe_audit_log fallback
"""

import contextvars
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure schemas is importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_CORE_PY = _REPO_ROOT / "packages" / "core-py"
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from schemas.common import (
    set_correlation_id,
    get_correlation_id,
    success_response,
    error_response,
    wrap_controller_result,
    raise_error,
    safe_audit_log,
    classify_and_raise,
    StandardResponse,
)
from domains.infrastructure.errors import (
    AppError,
    NotFoundError,
    ValidationError,
    AuthError,
    ResourceExhaustedError,
    ConfigError,
)


# ── Correlation ID ────────────────────────────────────────────────────


class TestCorrelationId:
    def test_get_default_is_none(self):
        token = set_correlation_id(None)
        assert get_correlation_id() is None
        set_correlation_id.reset(token) if hasattr(set_correlation_id, 'reset') else None

    def test_set_and_get(self):
        token = set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"
        set_correlation_id(None)

    def test_set_none_clears(self):
        set_correlation_id("abc-123")
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_overwrite(self):
        set_correlation_id("first")
        set_correlation_id("second")
        assert get_correlation_id() == "second"
        set_correlation_id(None)


# ── success_response ──────────────────────────────────────────────────


class TestSuccessResponse:
    def test_minimal(self):
        result = success_response()
        assert result == {"status": "success", "data": None}

    def test_with_data(self):
        result = success_response(data={"key": "val"})
        assert result["status"] == "success"
        assert result["data"] == {"key": "val"}

    def test_with_message(self):
        result = success_response(data=[], message="ok")
        assert result["message"] == "ok"

    def test_with_meta(self):
        result = success_response(data={}, meta={"page": 1})
        assert result["meta"] == {"page": 1}

    def test_no_message_when_none(self):
        result = success_response(data={})
        assert "message" not in result

    def test_no_meta_when_none(self):
        result = success_response(data={})
        assert "meta" not in result

    def test_full(self):
        result = success_response(data=[1, 2], message="fetched", meta={"count": 2})
        assert result == {
            "status": "success",
            "data": [1, 2],
            "message": "fetched",
            "meta": {"count": 2},
        }


# ── error_response ────────────────────────────────────────────────────


class TestErrorResponse:
    def test_minimal(self):
        result = error_response("something broke")
        assert result["error"] == "something broke"
        assert result["code"] == "E_DOMAIN"
        assert "details" not in result
        assert "correlation_id" not in result

    def test_with_code(self):
        result = error_response("not found", code="E_NOT_FOUND")
        assert result["code"] == "E_NOT_FOUND"

    def test_with_details(self):
        result = error_response("bad", details={"field": "name"})
        assert result["details"] == {"field": "name"}

    def test_with_correlation_id(self):
        result = error_response("err", correlation_id="req-42")
        assert result["correlation_id"] == "req-42"

    def test_auto_correlation_id(self):
        set_correlation_id("auto-cid")
        result = error_response("err")
        assert result["correlation_id"] == "auto-cid"
        set_correlation_id(None)

    def test_explicit_correlation_id_wins(self):
        set_correlation_id("auto-cid")
        result = error_response("err", correlation_id="explicit")
        assert result["correlation_id"] == "explicit"
        set_correlation_id(None)

    def test_no_correlation_id_when_none(self):
        set_correlation_id(None)
        result = error_response("err")
        assert "correlation_id" not in result

    def test_no_details_when_none(self):
        result = error_response("err", details=None)
        assert "details" not in result


# ── wrap_controller_result ────────────────────────────────────────────


class TestWrapControllerResult:
    def test_success(self):
        result = wrap_controller_result({"status": "loaded", "model": "gpt2"})
        assert result["status"] == "success"
        assert result["data"]["status"] == "loaded"

    def test_error(self):
        result = wrap_controller_result(
            {"status": "error", "error": "not found"},
            error_code="E_NOT_FOUND",
        )
        assert result["error"] == "not found"
        assert result["code"] == "E_NOT_FOUND"

    def test_error_falls_back_to_message(self):
        result = wrap_controller_result({"status": "error", "message": "oops"})
        assert result["error"] == "oops"

    def test_error_falls_back_to_generic(self):
        result = wrap_controller_result({"status": "error"})
        assert result["error"] == "Operation failed"

    def test_non_dict_passthrough(self):
        result = wrap_controller_result("hello")
        assert result["status"] == "success"
        assert result["data"] == "hello"

    def test_dict_without_error_status(self):
        result = wrap_controller_result({"status": "ok", "data": 42})
        assert result["status"] == "success"
        assert result["data"]["status"] == "ok"


# ── raise_error ───────────────────────────────────────────────────────


class TestRaiseError:
    def test_not_found(self):
        with pytest.raises(NotFoundError) as exc_info:
            raise_error("missing", "E_NOT_FOUND")
        assert exc_info.value.http_status == 404
        assert "missing" in str(exc_info.value.user_message)

    def test_bad_request(self):
        with pytest.raises(ValidationError) as exc_info:
            raise_error("invalid", "E_BAD_REQUEST")
        assert exc_info.value.http_status == 400

    def test_validation(self):
        with pytest.raises(ValidationError) as exc_info:
            raise_error("bad field", "E_VAL_REQUEST")
        assert exc_info.value.http_status == 422

    def test_auth_forbidden(self):
        with pytest.raises(AuthError) as exc_info:
            raise_error("denied", "E_AUTH_FORBIDDEN")
        assert exc_info.value.http_status == 403

    def test_auth_missing(self):
        with pytest.raises(AuthError) as exc_info:
            raise_error("no token", "E_AUTH_MISSING")
        assert exc_info.value.http_status == 401

    def test_rate_limit(self):
        with pytest.raises(ResourceExhaustedError) as exc_info:
            raise_error("slow down", "E_INFRA_RATE_LIMIT")
        assert exc_info.value.http_status == 429

    def test_busy(self):
        with pytest.raises(ResourceExhaustedError) as exc_info:
            raise_error("busy", "E_INFRA_BUSY")
        assert exc_info.value.http_status == 409

    def test_timeout(self):
        with pytest.raises(ResourceExhaustedError) as exc_info:
            raise_error("timed out", "E_INFRA_TIMEOUT")
        assert exc_info.value.http_status == 408

    def test_startup(self):
        with pytest.raises(ConfigError) as exc_info:
            raise_error("not ready", "E_INFRA_STARTUP")
        assert exc_info.value.http_status == 503

    def test_domain_default(self):
        with pytest.raises(AppError) as exc_info:
            raise_error("generic")
        assert exc_info.value.http_status == 400

    def test_domain_with_status_override(self):
        with pytest.raises(AppError) as exc_info:
            raise_error("custom", "E_DOMAIN", status_code=418)
        assert exc_info.value.http_status == 418

    def test_details_forwarded(self):
        with pytest.raises(AppError) as exc_info:
            raise_error("err", details={"field": "x"})
        assert exc_info.value.details == {"field": "x"}

    def test_code_forwarded(self):
        with pytest.raises(NotFoundError) as exc_info:
            raise_error("gone", "E_NOT_FOUND")
        assert exc_info.value.code == "E_NOT_FOUND"


# ── classify_and_raise ────────────────────────────────────────────────


class TestClassifyAndRaise:
    def test_raises_app_error(self):
        with pytest.raises(AppError) as exc_info:
            classify_and_raise(ValueError("boom"), source="test")
        assert exc_info.value.http_status >= 400

    def test_classifies_oserror(self):
        with pytest.raises(AppError) as exc_info:
            classify_and_raise(OSError("disk full"), source="test")
        assert exc_info.value.http_status >= 400

    def test_preserves_app_error_class(self):
        orig = NotFoundError("already classified")
        with pytest.raises(NotFoundError) as exc_info:
            classify_and_raise(orig, source="test")
        assert exc_info.value.http_status == 404
        assert exc_info.value.code == "resource.not_found"


# ── safe_audit_log ────────────────────────────────────────────────────


class TestSafeAuditLog:
    def test_does_not_crash_on_missing_module(self):
        with patch.dict("sys.modules", {"infrastructure.auth": None}):
            safe_audit_log("test.action", resource="r", detail="d")
            # Should not raise — falls back to _audit_logger

    def test_does_not_crash_on_import_error(self):
        with patch.dict("sys.modules", {"infrastructure.auth": None}):
            # Should fall back gracefully
            safe_audit_log("test.action", resource="r", detail="d")


# ── StandardResponse model ────────────────────────────────────────────


class TestStandardResponse:
    def test_minimal(self):
        resp = StandardResponse()
        assert resp.status == "success"
        assert resp.data == {}
        assert resp.message is None
        assert resp.meta is None

    def test_with_values(self):
        resp = StandardResponse(status="error", data={"x": 1}, message="no", meta={"a": 2})
        assert resp.status == "error"
        assert resp.data == {"x": 1}
        assert resp.message == "no"
        assert resp.meta == {"a": 2}
