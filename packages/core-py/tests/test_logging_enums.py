"""Tests for domains.logging.base — LogLevel, ErrorCode, LogTag, LogRecord."""

from domains.logging.base import LogLevel, ErrorCode, LogTag, LogRecord


class TestLogLevel:
    def test_all_members(self):
        assert len(LogLevel) == 5
    def test_values(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.CRITICAL.value == "critical"
    def test_comparison(self):
        assert LogLevel.ERROR > LogLevel.WARNING
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.WARNING >= LogLevel.WARNING
        assert LogLevel.WARNING <= LogLevel.WARNING
    def test_not_implemented(self):
        assert LogLevel.DEBUG.__ge__("other") is NotImplemented
        assert LogLevel.DEBUG.__gt__("other") is NotImplemented
        assert LogLevel.DEBUG.__le__("other") is NotImplemented
        assert LogLevel.DEBUG.__lt__("other") is NotImplemented


class TestErrorCode:
    def test_auth_codes(self):
        assert ErrorCode.E_AUTH_MISSING.value == "E_AUTH_MISSING"
        assert ErrorCode.E_AUTH_EXPIRED.value == "E_AUTH_EXPIRED"
    def test_model_codes(self):
        assert ErrorCode.E_MODEL_LOAD.value == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM.value == "E_MODEL_OOM"
    def test_inference_codes(self):
        assert ErrorCode.E_INF_GENERATION.value == "E_INF_GENERATION"


class TestLogTag:
    def test_all_members(self):
        assert len(LogTag) >= 10
    def test_values(self):
        assert LogTag.REQ.value == "REQ"
        assert LogTag.MODEL.value == "MODEL"
        assert LogTag.TRAIN.value == "TRAIN"
        assert LogTag.ERROR.value == "ERROR"


class TestLogRecord:
    def test_fields(self):
        lr = LogRecord(level=LogLevel.INFO, message="test", logger="slo.test")
        assert lr.level == LogLevel.INFO
        assert lr.message == "test"
        assert lr.logger == "slo.test"
        assert lr.exception is None
        assert lr.error_code is None
    def test_defaults(self):
        lr = LogRecord(level=LogLevel.DEBUG, message="x")
        assert lr.context == {}
        assert lr.tag is None
    def test_frozen(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        try:
            lr.message = "changed"
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass
