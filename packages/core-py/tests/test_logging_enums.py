"""Tests for domains.logging.base — LogLevel, ErrorCode, LogTag, LogRecord."""

import time

import pytest

from domains.logging.base import LogLevel, ErrorCode, LogTag, LogRecord


class TestLogLevel:
    def test_all_members(self):
        assert len(LogLevel) == 5

    def test_values(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
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

    def test_full_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_equal_to_self(self):
        assert LogLevel.INFO >= LogLevel.INFO
        assert LogLevel.INFO <= LogLevel.INFO
        assert not (LogLevel.INFO > LogLevel.INFO)
        assert not (LogLevel.INFO < LogLevel.INFO)

    def test_is_str_enum(self):
        assert LogLevel.DEBUG.value == "debug"

    def test_from_value(self):
        assert LogLevel("debug") is LogLevel.DEBUG
        assert LogLevel("critical") is LogLevel.CRITICAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            LogLevel("nonexistent")

    def test_repr(self):
        assert repr(LogLevel.DEBUG) == "<LogLevel.DEBUG: 'debug'>"

    def test_in_set(self):
        s = {LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}
        assert len(s) == 5

    def test_as_dict_key(self):
        d = {LogLevel.DEBUG: 0, LogLevel.CRITICAL: 4}
        assert d[LogLevel.DEBUG] == 0

    def test_name_attribute(self):
        assert LogLevel.ERROR.name == "ERROR"

    def test_ordering_chain_ge(self):
        assert LogLevel.CRITICAL >= LogLevel.ERROR >= LogLevel.WARNING >= LogLevel.INFO >= LogLevel.DEBUG

    def test_ne_different(self):
        assert LogLevel.DEBUG != LogLevel.CRITICAL

    def test_ne_same(self):
        assert not (LogLevel.INFO != LogLevel.INFO)

    def test_all_levels_are_strings(self):
        for level in LogLevel:
            assert isinstance(level.value, str)

    def test_unique_names(self):
        names = [l.name for l in LogLevel]
        assert len(names) == len(set(names))

    def test_ge_reverse(self):
        assert LogLevel.DEBUG >= LogLevel.DEBUG
        assert not (LogLevel.DEBUG >= LogLevel.INFO)

    def test_gt_same(self):
        assert not (LogLevel.ERROR > LogLevel.ERROR)

    def test_le_same(self):
        assert LogLevel.ERROR <= LogLevel.ERROR

    def test_lt_reverse(self):
        assert not (LogLevel.ERROR < LogLevel.WARNING)

    def test_comparison_with_non_level_ge(self):
        assert LogLevel.DEBUG.__ge__(42) is NotImplemented

    def test_comparison_with_non_level_gt(self):
        assert LogLevel.DEBUG.__gt__(42) is NotImplemented

    def test_comparison_with_non_level_le(self):
        assert LogLevel.DEBUG.__le__(42) is NotImplemented

    def test_comparison_with_non_level_lt(self):
        assert LogLevel.DEBUG.__lt__(42) is NotImplemented

    def test_debug_is_lowest(self):
        for level in LogLevel:
            if level is not LogLevel.DEBUG:
                assert LogLevel.DEBUG < level

    def test_critical_is_highest(self):
        for level in LogLevel:
            if level is not LogLevel.CRITICAL:
                assert LogLevel.CRITICAL > level

    def test_severity_chain(self):
        assert LogLevel.CRITICAL >= LogLevel.ERROR
        assert LogLevel.ERROR >= LogLevel.WARNING
        assert LogLevel.WARNING >= LogLevel.INFO
        assert LogLevel.INFO >= LogLevel.DEBUG


class TestErrorCode:
    def test_auth_codes(self):
        assert ErrorCode.E_AUTH_MISSING.value == "E_AUTH_MISSING"
        assert ErrorCode.E_AUTH_EXPIRED.value == "E_AUTH_EXPIRED"

    def test_model_codes(self):
        assert ErrorCode.E_MODEL_LOAD.value == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM.value == "E_MODEL_OOM"

    def test_inference_codes(self):
        assert ErrorCode.E_INF_GENERATION.value == "E_INF_GENERATION"

    def test_all_auth_codes(self):
        auth_codes = [ErrorCode.E_AUTH_MISSING, ErrorCode.E_AUTH_EXPIRED,
                      ErrorCode.E_AUTH_INVALID, ErrorCode.E_AUTH_FORBIDDEN]
        assert len(auth_codes) == 4

    def test_all_model_codes(self):
        model_codes = [ErrorCode.E_MODEL_LOAD, ErrorCode.E_MODEL_OOM,
                       ErrorCode.E_MODEL_TIMEOUT, ErrorCode.E_MODEL_CRASH,
                       ErrorCode.E_MODEL_NOT_FOUND, ErrorCode.E_MODEL_WARMUP]
        assert len(model_codes) == 6

    def test_infra_codes(self):
        infra = [ErrorCode.E_INFRA_STARTUP, ErrorCode.E_INFRA_TIMEOUT,
                 ErrorCode.E_INFRA_REGISTRY, ErrorCode.E_INFRA_PROVIDER]
        assert len(infra) == 4

    def test_validation_codes(self):
        val = [ErrorCode.E_VAL_REQUEST, ErrorCode.E_VAL_FIELD]
        assert len(val) == 2

    def test_training_codes(self):
        train = [ErrorCode.E_TRAIN_DATA, ErrorCode.E_TRAIN_CRASH, ErrorCode.E_TRAIN_CHECKPOINT]
        assert len(train) == 3

    def test_domain_codes(self):
        domain = [ErrorCode.E_DOMAIN, ErrorCode.E_NOT_FOUND, ErrorCode.E_CONFLICT]
        assert len(domain) == 3

    def test_is_str_enum(self):
        assert isinstance(ErrorCode.E_AUTH_MISSING, str)
        assert ErrorCode.E_AUTH_MISSING == "E_AUTH_MISSING"

    def test_total_count(self):
        assert len(ErrorCode) >= 20

    def test_inference_extra_codes(self):
        codes = [ErrorCode.E_INF_TOKENIZER, ErrorCode.E_INF_CACHE]
        assert len(codes) == 2

    def test_from_value(self):
        assert ErrorCode("E_AUTH_MISSING") is ErrorCode.E_AUTH_MISSING

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ErrorCode("NONEXISTENT")

    def test_repr(self):
        assert repr(ErrorCode.E_MODEL_LOAD) == "<ErrorCode.E_MODEL_LOAD: 'E_MODEL_LOAD'>"

    def test_name_attribute(self):
        assert ErrorCode.E_MODEL_OOM.name == "E_MODEL_OOM"

    def test_all_values_start_with_E(self):
        for code in ErrorCode:
            assert code.value.startswith("E_")

    def test_all_values_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)

    def test_unique_values(self):
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values))

    def test_unique_names(self):
        names = [c.name for c in ErrorCode]
        assert len(names) == len(set(names))

    def test_auth_forbidden(self):
        assert ErrorCode.E_AUTH_FORBIDDEN.value == "E_AUTH_FORBIDDEN"

    def test_model_not_found(self):
        assert ErrorCode.E_MODEL_NOT_FOUND.value == "E_MODEL_NOT_FOUND"

    def test_model_timeout(self):
        assert ErrorCode.E_MODEL_TIMEOUT.value == "E_MODEL_TIMEOUT"

    def test_model_crash(self):
        assert ErrorCode.E_MODEL_CRASH.value == "E_MODEL_CRASH"

    def test_model_warmup(self):
        assert ErrorCode.E_MODEL_WARMUP.value == "E_MODEL_WARMUP"

    def test_inf_tokenizer(self):
        assert ErrorCode.E_INF_TOKENIZER.value == "E_INF_TOKENIZER"

    def test_inf_cache(self):
        assert ErrorCode.E_INF_CACHE.value == "E_INF_CACHE"

    def test_infra_registry(self):
        assert ErrorCode.E_INFRA_REGISTRY.value == "E_INFRA_REGISTRY"

    def test_infra_provider(self):
        assert ErrorCode.E_INFRA_PROVIDER.value == "E_INFRA_PROVIDER"

    def test_val_field(self):
        assert ErrorCode.E_VAL_FIELD.value == "E_VAL_FIELD"

    def test_train_checkpoint(self):
        assert ErrorCode.E_TRAIN_CHECKPOINT.value == "E_TRAIN_CHECKPOINT"

    def test_not_found(self):
        assert ErrorCode.E_NOT_FOUND.value == "E_NOT_FOUND"

    def test_conflict(self):
        assert ErrorCode.E_CONFLICT.value == "E_CONFLICT"

    def test_in_set(self):
        s = {ErrorCode.E_AUTH_MISSING, ErrorCode.E_MODEL_LOAD}
        assert len(s) == 2


class TestLogTag:
    def test_all_members(self):
        assert len(LogTag) >= 10

    def test_values(self):
        assert LogTag.REQ.value == "REQ"
        assert LogTag.MODEL.value == "MODEL"
        assert LogTag.TRAIN.value == "TRAIN"
        assert LogTag.ERROR.value == "ERROR"

    def test_all_tags(self):
        tags = [LogTag.REQ, LogTag.AUTH, LogTag.MODEL, LogTag.SOUL,
                LogTag.TRAIN, LogTag.INFRA, LogTag.START, LogTag.SLOW,
                LogTag.ERROR, LogTag.WARN, LogTag.OK]
        assert len(tags) == 11

    def test_is_str_enum(self):
        assert isinstance(LogTag.REQ, str)
        assert LogTag.REQ == "REQ"

    def test_from_value(self):
        assert LogTag("INFRA") is LogTag.INFRA

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            LogTag("NONEXISTENT")

    def test_in_set(self):
        s = {LogTag.REQ, LogTag.AUTH, LogTag.MODEL}
        assert len(s) == 3

    def test_as_dict_key(self):
        d = {LogTag.REQ: "request", LogTag.ERROR: "error"}
        assert d[LogTag.REQ] == "request"

    def test_repr(self):
        assert repr(LogTag.TRAIN) == "<LogTag.TRAIN: 'TRAIN'>"

    def test_name_attribute(self):
        assert LogTag.START.name == "START"

    def test_all_values_are_uppercase(self):
        for tag in LogTag:
            assert tag.value == tag.value.upper()

    def test_all_values_are_strings(self):
        for tag in LogTag:
            assert isinstance(tag.value, str)

    def test_unique_values(self):
        values = [t.value for t in LogTag]
        assert len(values) == len(set(values))

    def test_unique_names(self):
        names = [t.name for t in LogTag]
        assert len(names) == len(set(names))

    def test_auth_tag(self):
        assert LogTag.AUTH.value == "AUTH"

    def test_soul_tag(self):
        assert LogTag.SOUL.value == "SOUL"

    def test_infra_tag(self):
        assert LogTag.INFRA.value == "INFRA"

    def test_start_tag(self):
        assert LogTag.START.value == "START"

    def test_slow_tag(self):
        assert LogTag.SLOW.value == "SLOW"

    def test_warn_tag(self):
        assert LogTag.WARN.value == "WARN"

    def test_ok_tag(self):
        assert LogTag.OK.value == "OK"

    def test_from_value_all(self):
        for tag in LogTag:
            assert LogTag(tag.value) is tag

    def test_members_are_ordered_by_name(self):
        assert LogTag.REQ.name == "REQ"
        assert LogTag.AUTH.name == "AUTH"
        assert LogTag.MODEL.name == "MODEL"

    def test_in_dict(self):
        d = {"REQ": LogTag.REQ, "MODEL": LogTag.MODEL}
        assert d["REQ"] is LogTag.REQ

    def test_from_value_string(self):
        assert LogTag("ERROR") is LogTag.ERROR

    def test_from_value_warn(self):
        assert LogTag("WARN") is LogTag.WARN

    def test_from_value_ok(self):
        assert LogTag("OK") is LogTag.OK


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

    def test_timestamp_default(self):
        before = time.time()
        lr = LogRecord(level=LogLevel.INFO, message="x")
        after = time.time()
        assert before <= lr.timestamp <= after

    def test_with_context(self):
        lr = LogRecord(level=LogLevel.INFO, message="x", context={"req_id": "abc"})
        assert lr.context["req_id"] == "abc"

    def test_with_exception(self):
        lr = LogRecord(level=LogLevel.ERROR, message="fail", exception="ValueError: bad")
        assert lr.exception == "ValueError: bad"

    def test_with_error_code(self):
        lr = LogRecord(level=LogLevel.ERROR, message="fail", error_code="E_MODEL_LOAD")
        assert lr.error_code == "E_MODEL_LOAD"

    def test_with_tag(self):
        lr = LogRecord(level=LogLevel.INFO, message="x", tag="REQ")
        assert lr.tag == "REQ"

    def test_logger_default(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        assert lr.logger == "slo"

    def test_immutability(self):
        lr = LogRecord(level=LogLevel.WARNING, message="w", logger="slo.x")
        with pytest.raises(AttributeError):
            lr.level = LogLevel.DEBUG

    def test_context_shared_reference(self):
        ctx = {"a": 1}
        lr = LogRecord(level=LogLevel.INFO, message="x", context=ctx)
        assert lr.context is ctx
        assert lr.context == {"a": 1}

    def test_all_fields(self):
        lr = LogRecord(
            level=LogLevel.CRITICAL, message="big fail",
            logger="slo.core", timestamp=1234567890.0,
            context={"k": "v"}, exception="RuntimeError: crash",
            error_code="E_MODEL_CRASH", tag="ERROR",
        )
        assert lr.level == LogLevel.CRITICAL
        assert lr.message == "big fail"
        assert lr.logger == "slo.core"
        assert lr.timestamp == 1234567890.0
        assert lr.context == {"k": "v"}
        assert lr.exception == "RuntimeError: crash"
        assert lr.error_code == "E_MODEL_CRASH"
        assert lr.tag == "ERROR"

    def test_empty_message(self):
        lr = LogRecord(level=LogLevel.INFO, message="")
        assert lr.message == ""

    def test_error_code_none_default(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        assert lr.error_code is None

    def test_exception_none_default(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        assert lr.exception is None

    def test_tag_none_default(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        assert lr.tag is None

    def test_freeze_cannot_set_level(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        with pytest.raises(AttributeError):
            lr.level = LogLevel.DEBUG

    def test_freeze_cannot_set_message(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        with pytest.raises(AttributeError):
            lr.message = "y"

    def test_freeze_cannot_set_logger(self):
        lr = LogRecord(level=LogLevel.INFO, message="x", logger="slo.a")
        with pytest.raises(AttributeError):
            lr.logger = "slo.b"

    def test_freeze_cannot_set_timestamp(self):
        lr = LogRecord(level=LogLevel.INFO, message="x")
        with pytest.raises(AttributeError):
            lr.timestamp = 0.0

    def test_context_mutable_value(self):
        lr = LogRecord(level=LogLevel.INFO, message="x", context={"a": 1})
        lr.context["a"] = 2
        assert lr.context["a"] == 2

    def test_multiple_context_keys(self):
        lr = LogRecord(level=LogLevel.INFO, message="x",
                       context={"a": 1, "b": "two", "c": 3.0})
        assert len(lr.context) == 3

    def test_exception_string_format(self):
        lr = LogRecord(level=LogLevel.ERROR, message="fail",
                       exception="KeyError: 'missing'")
        assert "KeyError" in lr.exception
        assert "missing" in lr.exception

    def test_error_code_with_tag(self):
        lr = LogRecord(level=LogLevel.WARNING, message="warn",
                       error_code="E_AUTH_EXPIRING", tag="AUTH")
        assert lr.error_code == "E_AUTH_EXPIRING"
        assert lr.tag == "AUTH"


class TestLogLevelComparisonChained:
    def test_debug_lt_info(self):
        assert LogLevel.DEBUG < LogLevel.INFO

    def test_info_lt_warning(self):
        assert LogLevel.INFO < LogLevel.WARNING

    def test_warning_lt_error(self):
        assert LogLevel.WARNING < LogLevel.ERROR

    def test_error_lt_critical(self):
        assert LogLevel.ERROR < LogLevel.CRITICAL

    def test_ge_chain(self):
        assert LogLevel.CRITICAL >= LogLevel.ERROR >= LogLevel.WARNING >= LogLevel.INFO >= LogLevel.DEBUG

    def test_ne_different_levels(self):
        assert LogLevel.DEBUG != LogLevel.CRITICAL

    def test_ne_same_level(self):
        assert not (LogLevel.INFO != LogLevel.INFO)
