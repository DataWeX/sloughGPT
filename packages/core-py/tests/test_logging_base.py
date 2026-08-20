"""
Tests for domains/logging/base.py — LogLevel, ErrorCode, LogTag, LogRecord,
Logger ABC, TaggedLogger, ChildLogger.

Covers:
    - LogLevel: ordering operators, invalid comparison
    - ErrorCode: all values present
    - LogTag: all values present
    - LogRecord: frozen, default fields
    - Logger: level filtering, emit delegation, context management,
      convenience methods, tagged and child loggers
    - TaggedLogger: delegates to parent, injects tag
    - ChildLogger: delegates to parent, inherits level
"""

import sys
from pathlib import Path
from typing import List

import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.logging.base import (
    LogLevel,
    ErrorCode,
    LogTag,
    LogRecord,
    Logger,
    TaggedLogger,
    ChildLogger,
)


class CollectorLogger(Logger):
    """In-memory logger that collects emitted records for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: List[LogRecord] = []

    def emit(self, record: LogRecord) -> None:
        self.records.append(record)


class TestLogLevel:
    def test_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_ge(self):
        assert LogLevel.WARNING >= LogLevel.INFO
        assert LogLevel.WARNING >= LogLevel.WARNING
        assert not (LogLevel.INFO >= LogLevel.WARNING)

    def test_gt(self):
        assert LogLevel.ERROR > LogLevel.WARNING
        assert not (LogLevel.INFO > LogLevel.INFO)

    def test_le(self):
        assert LogLevel.INFO <= LogLevel.WARNING
        assert LogLevel.INFO <= LogLevel.INFO
        assert not (LogLevel.WARNING <= LogLevel.INFO)

    def test_lt(self):
        assert LogLevel.DEBUG < LogLevel.CRITICAL
        assert not (LogLevel.INFO < LogLevel.INFO)

    def test_invalid_comparison_returns_not_implemented(self):
        result = LogLevel.INFO.__gt__("not a log level")
        assert result is NotImplemented

    def test_values(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"


class TestErrorCode:
    def test_auth_codes(self):
        assert ErrorCode.E_AUTH_MISSING.value == "E_AUTH_MISSING"
        assert ErrorCode.E_AUTH_EXPIRED.value == "E_AUTH_EXPIRED"
        assert ErrorCode.E_AUTH_INVALID.value == "E_AUTH_INVALID"
        assert ErrorCode.E_AUTH_FORBIDDEN.value == "E_AUTH_FORBIDDEN"

    def test_model_codes(self):
        assert ErrorCode.E_MODEL_LOAD.value == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM.value == "E_MODEL_OOM"
        assert ErrorCode.E_MODEL_TIMEOUT.value == "E_MODEL_TIMEOUT"

    def test_infra_codes(self):
        assert ErrorCode.E_INFRA_STARTUP.value == "E_INFRA_STARTUP"
        assert ErrorCode.E_INFRA_TIMEOUT.value == "E_INFRA_TIMEOUT"

    def test_all_codes_are_strings(self):
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value.startswith("E_")


class TestLogTag:
    def test_all_values(self):
        tags = [t.value for t in LogTag]
        assert "REQ" in tags
        assert "AUTH" in tags
        assert "MODEL" in tags
        assert "ERROR" in tags


class TestLogRecord:
    def test_frozen(self):
        r = LogRecord(level=LogLevel.INFO, message="test")
        with pytest.raises(AttributeError):
            r.message = "changed"

    def test_defaults(self):
        r = LogRecord(level=LogLevel.INFO, message="hi")
        assert r.logger == "slo"
        assert r.timestamp > 0
        assert r.context == {}
        assert r.exception is None
        assert r.error_code is None
        assert r.tag is None

    def test_custom_fields(self):
        r = LogRecord(
            level=LogLevel.ERROR,
            message="boom",
            logger="slo.api",
            context={"request_id": "abc"},
            exception="ValueError: bad",
            error_code="E_VAL_REQUEST",
            tag="REQ",
        )
        assert r.logger == "slo.api"
        assert r.context == {"request_id": "abc"}
        assert r.exception == "ValueError: bad"
        assert r.error_code == "E_VAL_REQUEST"
        assert r.tag == "REQ"


class TestLogger:
    def test_emit_called(self):
        log = CollectorLogger("slo.test", level=LogLevel.DEBUG)
        log.info("hello")
        assert len(log.records) == 1
        assert log.records[0].message == "hello"

    def test_level_filtering(self):
        log = CollectorLogger("slo.test", level=LogLevel.WARNING)
        log.debug("nope")
        log.info("nope")
        log.warning("yes")
        log.error("yes")
        assert len(log.records) == 2
        assert [r.level for r in log.records] == [LogLevel.WARNING, LogLevel.ERROR]

    def test_set_level(self):
        log = CollectorLogger("slo.test", level=LogLevel.ERROR)
        log.info("filtered")
        assert len(log.records) == 0
        log.level = LogLevel.INFO
        log.info("emitted")
        assert len(log.records) == 1

    def test_context_merge(self):
        log = CollectorLogger("slo.test", context={"env": "prod"})
        log.set_context(version="1.0")
        log.info("msg")
        assert log.records[0].context == {"env": "prod", "version": "1.0"}

    def test_clear_context(self):
        log = CollectorLogger("slo.test", context={"a": 1})
        log.clear_context()
        log.info("msg")
        assert log.records[0].context == {}

    def test_debug(self):
        log = CollectorLogger("slo.test", level=LogLevel.DEBUG)
        log.debug("d")
        assert log.records[0].level == LogLevel.DEBUG

    def test_info(self):
        log = CollectorLogger("slo.test")
        log.info("i")
        assert log.records[0].level == LogLevel.INFO

    def test_warning(self):
        log = CollectorLogger("slo.test")
        log.warning("w")
        assert log.records[0].level == LogLevel.WARNING

    def test_error_with_exception(self):
        log = CollectorLogger("slo.test")
        log.error("e", exception="ValueError: x", error_code="E_VAL_REQUEST")
        assert log.records[0].level == LogLevel.ERROR
        assert log.records[0].exception == "ValueError: x"
        assert log.records[0].error_code == "E_VAL_REQUEST"

    def test_critical(self):
        log = CollectorLogger("slo.test")
        log.critical("c", exception="RuntimeError: boom")
        assert log.records[0].level == LogLevel.CRITICAL
        assert log.records[0].exception == "RuntimeError: boom"

    def test_exception_convenience(self):
        log = CollectorLogger("slo.test")
        try:
            raise ValueError("bad input")
        except ValueError as e:
            log.exception("caught", exc=e)
        assert log.records[0].level == LogLevel.ERROR
        assert "ValueError: bad input" in log.records[0].exception

    def test_context_with_explicit_overrides_default(self):
        log = CollectorLogger("slo.test", context={"a": 1})
        log.info("msg", a=2, b=3)
        assert log.records[0].context == {"a": 2, "b": 3}

    def test_repr(self):
        log = CollectorLogger("slo.test", level=LogLevel.DEBUG)
        assert "slo.test" in repr(log)
        assert "debug" in repr(log)


class TestTaggedLogger:
    def test_tag_applied(self):
        parent = CollectorLogger("slo.api")
        tagged = parent.tag("REQ")
        tagged.info("handling")
        assert parent.records[0].tag == "REQ"

    def test_tag_in_context(self):
        parent = CollectorLogger("slo.api")
        tagged = parent.tag("MODEL")
        tagged.error("load failed")
        assert parent.records[0].tag == "MODEL"
        assert parent.records[0].level == LogLevel.ERROR

    def test_tagged_delegates_to_parent_emit(self):
        parent = CollectorLogger("slo.api")
        tagged = parent.tag("TRAIN")
        tagged.info("x")
        assert len(parent.records) == 1

    def test_tagged_inherits_parent_level(self):
        parent = CollectorLogger("slo.api", level=LogLevel.ERROR)
        tagged = parent.tag("REQ")
        tagged.info("filtered")
        assert len(parent.records) == 0

    def test_tagged_level_set_propagates_to_parent(self):
        parent = CollectorLogger("slo.api", level=LogLevel.ERROR)
        tagged = parent.tag("REQ")
        tagged.level = LogLevel.DEBUG
        assert parent.level == LogLevel.DEBUG


class TestChildLogger:
    def test_child_name(self):
        parent = CollectorLogger("slo.api")
        child = parent.child("inference")
        assert child.name == "slo.api.inference"

    def test_child_delegates_to_parent(self):
        parent = CollectorLogger("slo.api")
        child = parent.child("inference")
        child.info("generating")
        assert len(parent.records) == 1
        assert parent.records[0].logger == "slo.api.inference"

    def test_child_inherits_parent_level(self):
        parent = CollectorLogger("slo.api", level=LogLevel.WARNING)
        child = parent.child("inference")
        child.info("filtered")
        assert len(parent.records) == 0

    def test_child_level_set_propagates(self):
        parent = CollectorLogger("slo.api", level=LogLevel.ERROR)
        child = parent.child("inference")
        child.level = LogLevel.DEBUG
        assert parent.level == LogLevel.DEBUG

    def test_child_context_merges(self):
        parent = CollectorLogger("slo.api", context={"env": "prod"})
        child = parent.child("inference", version="2")
        child.info("msg")
        assert parent.records[0].context == {"env": "prod", "version": "2"}

    def test_child_logger_name_in_record(self):
        parent = CollectorLogger("slo.api")
        child = parent.child("inference")
        child.info("msg")
        assert parent.records[0].logger == "slo.api.inference"
