"""Tests for logging.base — LogLevel, ErrorCode, LogTag, LogRecord, Logger, TaggedLogger, ChildLogger, CompositeLogger."""

from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock

import pytest

from domains.logging.base import (
    LogLevel, ErrorCode, LogTag, LogRecord, Logger,
    TaggedLogger, ChildLogger, CompositeLogger,
)


# ── LogLevel ────────────────────────────────────────────────────────────────


class TestLogLevel:

    def test_values(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_ge(self):
        assert LogLevel.ERROR >= LogLevel.WARNING
        assert LogLevel.INFO >= LogLevel.DEBUG
        assert not (LogLevel.DEBUG >= LogLevel.INFO)

    def test_gt(self):
        assert LogLevel.CRITICAL > LogLevel.ERROR
        assert not (LogLevel.INFO > LogLevel.INFO)

    def test_le(self):
        assert LogLevel.DEBUG <= LogLevel.INFO
        assert LogLevel.INFO <= LogLevel.INFO
        assert not (LogLevel.ERROR <= LogLevel.DEBUG)

    def test_lt(self):
        assert LogLevel.WARNING < LogLevel.ERROR
        assert not (LogLevel.INFO < LogLevel.DEBUG)

    def test_comparison_not_impl(self):
        assert LogLevel.ERROR.__ge__("not a level") is NotImplemented
        assert LogLevel.ERROR.__gt__(42) is NotImplemented
        assert LogLevel.ERROR.__le__("x") is NotImplemented
        assert LogLevel.ERROR.__lt__([]) is NotImplemented


# ── ErrorCode ───────────────────────────────────────────────────────────────


class TestErrorCode:

    def test_auth_codes(self):
        assert ErrorCode.E_AUTH_MISSING.value == "E_AUTH_MISSING"
        assert ErrorCode.E_AUTH_EXPIRED.value == "E_AUTH_EXPIRED"

    def test_model_codes(self):
        assert ErrorCode.E_MODEL_LOAD.value == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM.value == "E_MODEL_OOM"

    def test_is_str(self):
        assert isinstance(ErrorCode.E_DOMAIN, str)


# ── LogTag ──────────────────────────────────────────────────────────────────


class TestLogTag:

    def test_values(self):
        assert LogTag.REQ.value == "REQ"
        assert LogTag.TRAIN.value == "TRAIN"
        assert LogTag.ERROR.value == "ERROR"

    def test_is_str(self):
        assert isinstance(LogTag.OK, str)


# ── LogRecord ───────────────────────────────────────────────────────────────


class TestLogRecord:

    def test_defaults(self):
        r = LogRecord(level=LogLevel.INFO, message="hello")
        assert r.level == LogLevel.INFO
        assert r.message == "hello"
        assert r.logger == "slo"
        assert r.context == {}
        assert r.exception is None
        assert r.error_code is None
        assert r.tag is None

    def test_custom(self):
        r = LogRecord(
            level=LogLevel.ERROR,
            message="fail",
            logger="slo.api",
            context={"req": "123"},
            exception="ValueError: bad",
            error_code="E_VAL_FIELD",
            tag="REQ",
        )
        assert r.logger == "slo.api"
        assert r.context["req"] == "123"
        assert r.exception == "ValueError: bad"

    def test_frozen(self):
        r = LogRecord(level=LogLevel.INFO, message="x")
        with pytest.raises(AttributeError):
            r.message = "y"


# ── Concrete Logger for testing ─────────────────────────────────────────────


class ConcreteLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record: LogRecord) -> None:
        self.records.append(record)


# ── Logger ──────────────────────────────────────────────────────────────────


class TestLogger:

    def test_init(self):
        log = ConcreteLogger("test")
        assert log.name == "test"
        assert log.level == LogLevel.INFO

    def test_level_setter(self):
        log = ConcreteLogger("test", level=LogLevel.DEBUG)
        log.level = LogLevel.ERROR
        assert log.level == LogLevel.ERROR

    def test_context(self):
        log = ConcreteLogger("test", context={"a": 1})
        assert log.context["a"] == 1

    def test_set_context(self):
        log = ConcreteLogger("test")
        log.set_context(b=2)
        assert log.context["b"] == 2

    def test_clear_context(self):
        log = ConcreteLogger("test", context={"a": 1})
        log.clear_context()
        assert log.context == {}

    def test_debug(self):
        log = ConcreteLogger("test", level=LogLevel.DEBUG)
        log.debug("msg", k="v")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.DEBUG
        assert log.records[0].message == "msg"

    def test_info(self):
        log = ConcreteLogger("test")
        log.info("hello", error_code="E_DOMAIN")
        assert log.records[0].level == LogLevel.INFO
        assert log.records[0].error_code == "E_DOMAIN"

    def test_warning(self):
        log = ConcreteLogger("test")
        log.warning("careful")
        assert log.records[0].level == LogLevel.WARNING

    def test_error(self):
        log = ConcreteLogger("test")
        log.error("fail", exception="RuntimeError: boom")
        assert log.records[0].level == LogLevel.ERROR
        assert "boom" in log.records[0].exception

    def test_critical(self):
        log = ConcreteLogger("test")
        log.critical("fatal")
        assert log.records[0].level == LogLevel.CRITICAL

    def test_exception(self):
        log = ConcreteLogger("test")
        try:
            raise ValueError("oops")
        except ValueError as e:
            log.exception("caught", exc=e)
        assert "ValueError: oops" in log.records[0].exception

    def test_step(self):
        log = ConcreteLogger("test")
        log.step("doing thing")
        assert log.records[0].level == LogLevel.INFO

    def test_success(self):
        log = ConcreteLogger("test")
        log.success("done")
        assert log.records[0].level == LogLevel.INFO

    def test_should_emit_filters(self):
        log = ConcreteLogger("test", level=LogLevel.WARNING)
        log.debug("nope")
        log.info("nope")
        log.warning("yes")
        assert len(log.records) == 1

    def test_repr(self):
        log = ConcreteLogger("test")
        assert "test" in repr(log)

    def test_tag(self):
        log = ConcreteLogger("test")
        tagged = log.tag("REQ")
        assert isinstance(tagged, TaggedLogger)
        tagged.info("hello")
        assert log.records[0].tag == "REQ"

    def test_child(self):
        log = ConcreteLogger("slo.api")
        child = log.child("inference")
        assert child.name == "slo.api.inference"
        child.info("generating")
        assert len(log.records) == 1


# ── TaggedLogger ────────────────────────────────────────────────────────────


class TestTaggedLogger:

    def test_emit_delegates_to_parent(self):
        parent = ConcreteLogger("parent")
        tagged = parent.tag("TRAIN")
        tagged.info("training step")
        assert len(parent.records) == 1
        assert parent.records[0].tag == "TRAIN"

    def test_level_delegates_to_parent(self):
        parent = ConcreteLogger("parent", level=LogLevel.WARNING)
        tagged = parent.tag("REQ")
        assert tagged.level == LogLevel.WARNING
        tagged.level = LogLevel.DEBUG
        assert parent.level == LogLevel.DEBUG

    def test_context_merged(self):
        parent = ConcreteLogger("parent", context={"a": 1})
        tagged = parent.tag("REQ")
        tagged.info("msg", b=2)
        assert parent.records[0].context["a"] == 1
        assert parent.records[0].context["b"] == 2


# ── ChildLogger ─────────────────────────────────────────────────────────────


class TestChildLogger:

    def test_emit_delegates_to_parent(self):
        parent = ConcreteLogger("parent")
        child = parent.child("sub")
        child.info("from child")
        assert len(parent.records) == 1

    def test_level_delegates_to_parent(self):
        parent = ConcreteLogger("parent", level=LogLevel.ERROR)
        child = parent.child("sub")
        assert child.level == LogLevel.ERROR
        child.level = LogLevel.DEBUG
        assert parent.level == LogLevel.DEBUG

    def test_name_concat(self):
        parent = ConcreteLogger("a.b")
        child = parent.child("c")
        assert child.name == "a.b.c"


# ── CompositeLogger ────────────────────────────────────────────────────────


class TestCompositeLogger:

    def test_emit_to_children(self):
        c1 = ConcreteLogger("c1")
        c2 = ConcreteLogger("c2")
        comp = CompositeLogger("comp", children=[c1, c2])
        comp.info("shared")
        assert len(c1.records) == 1
        assert len(c2.records) == 1

    def test_add(self):
        comp = CompositeLogger("comp")
        c = ConcreteLogger("child")
        result = comp.add(c)
        assert result is comp
        assert c in comp.children

    def test_remove(self):
        c = ConcreteLogger("child")
        comp = CompositeLogger("comp", children=[c])
        comp.remove(c)
        assert c not in comp.children

    def test_children_returns_copy(self):
        c = ConcreteLogger("child")
        comp = CompositeLogger("comp", children=[c])
        kids = comp.children
        kids.clear()
        assert c in comp.children
