"""Tests for domains.logging — LogLevel, LogRecord, Logger ABC, formatters, config, and serialization."""

import io
import json
import logging
import time
from io import StringIO
from unittest.mock import MagicMock

import pytest

from domains.logging.base import (
    LogLevel,
    LogRecord,
    Logger,
    ChildLogger,
    TaggedLogger,
    CompositeLogger,
    ErrorCode,
    LogTag,
)
from domains.logging.console_logger import ConsoleLogger
from domains.logging.cli_logger import CLILogger
from domains.logging.shell_logger import ShellLogger
from domains.logging.web_logger import WebLogger
from domains.logging.bridge import BridgeHandler, record_extra_context
from domains.logging.config import (
    LogFormatter,
    get_request_id,
    set_request_id,
    get_log_context,
    set_log_context,
    clear_log_context,
    _derive_op,
    ClientExtensionFilter,
    _LEGACY_TAG_TO_OP,
)


# ── LogLevel comparisons ─────────────────────────────────────────────────

class TestLogLevel:
    def test_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.CRITICAL

    def test_ge(self):
        assert LogLevel.WARNING >= LogLevel.INFO
        assert LogLevel.WARNING >= LogLevel.WARNING
        assert not (LogLevel.INFO >= LogLevel.WARNING)

    def test_gt(self):
        assert LogLevel.ERROR > LogLevel.WARNING
        assert not (LogLevel.INFO > LogLevel.INFO)

    def test_le(self):
        assert LogLevel.DEBUG <= LogLevel.INFO
        assert LogLevel.INFO <= LogLevel.INFO
        assert not (LogLevel.WARNING <= LogLevel.DEBUG)

    def test_lt(self):
        assert LogLevel.DEBUG < LogLevel.CRITICAL
        assert not (LogLevel.ERROR < LogLevel.WARNING)

    def test_comparison_with_non_loglevel_returns_not_implemented(self):
        assert LogLevel.DEBUG.__ge__("not a log level") is NotImplemented
        assert LogLevel.DEBUG.__gt__(42) is NotImplemented
        assert LogLevel.DEBUG.__le__([]) is NotImplemented
        assert LogLevel.DEBUG.__lt__({}) is NotImplemented

    def test_value_attributes(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"


# ── ErrorCode enum ────────────────────────────────────────────────────────

class TestErrorCode:
    def test_all_members_exist(self):
        assert ErrorCode.E_AUTH_MISSING.value == "E_AUTH_MISSING"
        assert ErrorCode.E_MODEL_LOAD.value == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM.value == "E_MODEL_OOM"
        assert ErrorCode.E_INF_GENERATION.value == "E_INF_GENERATION"
        assert ErrorCode.E_INFRA_STARTUP.value == "E_INFRA_STARTUP"
        assert ErrorCode.E_VAL_REQUEST.value == "E_VAL_REQUEST"
        assert ErrorCode.E_TRAIN_DATA.value == "E_TRAIN_DATA"
        assert ErrorCode.E_DOMAIN.value == "E_DOMAIN"

    def test_is_string_enum(self):
        assert isinstance(ErrorCode.E_AUTH_MISSING, str)
        assert ErrorCode.E_AUTH_MISSING == "E_AUTH_MISSING"

    def test_member_count(self):
        assert len(list(ErrorCode)) >= 20


# ── LogTag enum ───────────────────────────────────────────────────────────

class TestLogTag:
    def test_members(self):
        assert LogTag.REQ.value == "REQ"
        assert LogTag.MODEL.value == "MODEL"
        assert LogTag.TRAIN.value == "TRAIN"
        assert LogTag.ERROR.value == "ERROR"

    def test_is_string_enum(self):
        assert isinstance(LogTag.REQ, str)


# ── LogRecord dataclass ──────────────────────────────────────────────────

class TestLogRecord:
    def test_defaults(self):
        record = LogRecord(level=LogLevel.INFO, message="test")
        assert record.level == LogLevel.INFO
        assert record.message == "test"
        assert record.logger == "slo"
        assert record.context == {}
        assert record.exception is None
        assert record.error_code is None
        assert record.tag is None
        assert record.timestamp > 0

    def test_custom_fields(self):
        record = LogRecord(
            level=LogLevel.ERROR,
            message="fail",
            logger="slo.api",
            timestamp=12345.0,
            context={"key": "val"},
            exception="RuntimeError: oops",
            error_code="E_MODEL_OOM",
            tag="MODEL",
        )
        assert record.logger == "slo.api"
        assert record.timestamp == 12345.0
        assert record.context == {"key": "val"}
        assert record.exception == "RuntimeError: oops"

    def test_frozen(self):
        record = LogRecord(level=LogLevel.INFO, message="test")
        with pytest.raises(AttributeError):
            record.message = "changed"


# ── Concrete Logger subclass for testing ──────────────────────────────────

class _SinkLogger(Logger):
    """Logger that captures emitted records for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record: LogRecord):
        self.records.append(record)


# ── Logger ABC convenience methods ────────────────────────────────────────

class TestLoggerConvenienceMethods:
    def test_debug(self):
        log = _SinkLogger(level=LogLevel.DEBUG)
        log.debug("debug msg", key="val")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.DEBUG
        assert log.records[0].message == "debug msg"
        assert log.records[0].context["key"] == "val"

    def test_info(self):
        log = _SinkLogger(level=LogLevel.INFO)
        log.info("info msg")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.INFO

    def test_warning(self):
        log = _SinkLogger(level=LogLevel.WARNING)
        log.warning("warn msg")
        assert log.records[0].level == LogLevel.WARNING

    def test_error(self):
        log = _SinkLogger(level=LogLevel.ERROR)
        log.error("err msg", exception="RuntimeError: boom")
        assert log.records[0].level == LogLevel.ERROR
        assert log.records[0].exception == "RuntimeError: boom"

    def test_critical(self):
        log = _SinkLogger(level=LogLevel.CRITICAL)
        log.critical("crit msg")
        assert log.records[0].level == LogLevel.CRITICAL

    def test_exception_method(self):
        log = _SinkLogger(level=LogLevel.ERROR)
        try:
            raise ValueError("bad value")
        except ValueError as e:
            log.exception("failed", exc=e)
        assert "ValueError" in log.records[0].exception
        assert "bad value" in log.records[0].exception

    def test_error_code_passed(self):
        log = _SinkLogger(level=LogLevel.ERROR)
        log.error("oom", error_code="E_MODEL_OOM")
        assert log.records[0].error_code == "E_MODEL_OOM"

    def test_level_filtering_debug_not_emitted(self):
        log = _SinkLogger(level=LogLevel.INFO)
        log.debug("should not emit")
        assert len(log.records) == 0

    def test_level_filtering_info_emitted(self):
        log = _SinkLogger(level=LogLevel.INFO)
        log.info("should emit")
        assert len(log.records) == 1

    def test_context_merge(self):
        log = _SinkLogger(level=LogLevel.DEBUG, context={"base": True})
        log.info("msg", extra_key="extra_val")
        assert log.records[0].context["base"] is True
        assert log.records[0].context["extra_key"] == "extra_val"

    def test_set_context(self):
        log = _SinkLogger(level=LogLevel.DEBUG)
        log.set_context(session="abc")
        log.info("msg")
        assert log.records[0].context["session"] == "abc"

    def test_clear_context(self):
        log = _SinkLogger(level=LogLevel.DEBUG, context={"a": 1})
        log.clear_context()
        log.info("msg")
        assert log.records[0].context == {}

    def test_name_property(self):
        log = _SinkLogger(name="slo.test")
        assert log.name == "slo.test"

    def test_level_property(self):
        log = _SinkLogger(level=LogLevel.WARNING)
        assert log.level == LogLevel.WARNING

    def test_level_setter(self):
        log = _SinkLogger(level=LogLevel.WARNING)
        log.level = LogLevel.ERROR
        assert log.level == LogLevel.ERROR

    def test_repr(self):
        log = _SinkLogger(name="slo.x", level=LogLevel.DEBUG)
        r = repr(log)
        assert "slo.x" in r
        assert "debug" in r


# ── TaggedLogger ──────────────────────────────────────────────────────────

class TestTaggedLogger:
    def test_tag_attached(self):
        parent = _SinkLogger(level=LogLevel.DEBUG)
        tagged = parent.tag("MODEL")
        tagged.info("loaded")
        assert parent.records[0].tag == "MODEL"

    def test_tag_does_not_mutate_parent(self):
        parent = _SinkLogger(level=LogLevel.DEBUG)
        tagged = parent.tag("TRAIN")
        tagged.info("msg")
        assert parent.records[0].tag == "TRAIN"

    def test_tagged_level_inherits_parent(self):
        parent = _SinkLogger(level=LogLevel.WARNING)
        tagged = parent.tag("REQ")
        tagged.info("should not emit")
        assert len(parent.records) == 0


# ── ChildLogger ──────────────────────────────────────────────────────────

class TestChildLogger:
    def test_child_name(self):
        parent = _SinkLogger(name="slo.api")
        child = parent.child("inference")
        assert child.name == "slo.api.inference"

    def test_child_emits_through_parent(self):
        parent = _SinkLogger(level=LogLevel.DEBUG)
        child = parent.child("sub")
        child.info("from child")
        assert len(parent.records) == 1
        assert parent.records[0].message == "from child"

    def test_child_inherits_parent_level(self):
        parent = _SinkLogger(level=LogLevel.WARNING)
        child = parent.child("sub")
        child.info("should not emit")
        assert len(parent.records) == 0

    def test_child_level_setter_affects_parent(self):
        parent = _SinkLogger(level=LogLevel.WARNING)
        child = parent.child("sub")
        child.level = LogLevel.DEBUG
        assert parent.level == LogLevel.DEBUG

    def test_child_context_merge(self):
        parent = _SinkLogger(level=LogLevel.DEBUG, context={"a": 1})
        child = parent.child("sub", b=2)
        child.info("msg")
        assert parent.records[0].context["a"] == 1
        assert parent.records[0].context["b"] == 2


# ── CompositeLogger ──────────────────────────────────────────────────────

class TestCompositeLogger:
    def test_emits_to_all_children(self):
        a = _SinkLogger(level=LogLevel.DEBUG)
        b = _SinkLogger(level=LogLevel.DEBUG)
        composite = CompositeLogger(children=[a, b])
        composite.info("msg")
        assert len(a.records) == 1
        assert len(b.records) == 1

    def test_add_child(self):
        a = _SinkLogger(level=LogLevel.DEBUG)
        composite = CompositeLogger()
        composite.add(a)
        composite.info("msg")
        assert len(a.records) == 1

    def test_remove_child(self):
        a = _SinkLogger(level=LogLevel.DEBUG)
        composite = CompositeLogger(children=[a])
        composite.remove(a)
        composite.info("msg")
        assert len(a.records) == 0

    def test_children_property(self):
        a = _SinkLogger()
        b = _SinkLogger()
        composite = CompositeLogger(children=[a, b])
        assert len(composite.children) == 2

    def test_add_chaining(self):
        a = _SinkLogger()
        composite = CompositeLogger()
        result = composite.add(a)
        assert result is composite


# ── ConsoleLogger formatting ──────────────────────────────────────────────

class TestConsoleLogger:
    def test_emit_to_stream(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello")
        output = stream.getvalue()
        assert "hello" in output

    def test_json_format(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, format="json")
        log.info("test msg", key="val")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["msg"] == "test msg"
        assert data["lvl"] == "INFO"
        assert data["logger"] == "slo.test"

    def test_human_format_no_colors(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("test msg")
        output = stream.getvalue()
        assert "INF" in output
        assert "test msg" in output

    def test_exception_formatting(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.error("fail", exception="RuntimeError: OOM")
        output = stream.getvalue()
        assert "RuntimeError" in output
        assert "OOM" in output

    def test_error_code_in_output(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.error("oom", error_code="E_MODEL_OOM")
        output = stream.getvalue()
        assert "E_MODEL_OOM" in output

    def test_tag_in_output(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.tag("MODEL").info("loaded")
        output = stream.getvalue()
        assert "MODEL" in output

    def test_context_in_output(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("msg", port=8000)
        output = stream.getvalue()
        assert "port" in output
        assert "8000" in output

    def test_format_json_record(self):
        stream = StringIO()
        log = ConsoleLogger("slo.test", level=LogLevel.DEBUG, format="json", stream=stream)
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.test")
        log.emit(record)
        output = stream.getvalue()
        data = json.loads(output)
        assert data["lvl"] == "INFO"
        assert data["msg"] == "test"


# ── CLILogger formatting ──────────────────────────────────────────────────

class TestCLILogger:
    def test_emit_to_stream(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello")
        output = stream.getvalue()
        assert "hello" in output

    def test_success(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.success("done")
        output = stream.getvalue()
        assert "done" in output

    def test_step(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.step("processing")
        output = stream.getvalue()
        assert "processing" in output

    def test_header(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.header("My Section")
        output = stream.getvalue()
        assert "My Section" in output

    def test_section(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.section("Part 1")
        output = stream.getvalue()
        assert "Part 1" in output

    def test_table(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.table(["Name", "Value"], [["a", "1"], ["b", "2"]])
        output = stream.getvalue()
        assert "Name" in output
        assert "Value" in output

    def test_table_empty(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.table(["H"], [])
        assert stream.getvalue() == ""

    def test_json_output(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.json({"key": "value"})
        output = stream.getvalue()
        data = json.loads(output)
        assert data["key"] == "value"

    def test_status(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.status("Model", "gpt2", status="ok")
        output = stream.getvalue()
        assert "Model" in output
        assert "gpt2" in output

    def test_key_value(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.key_value("loss", "0.42")
        output = stream.getvalue()
        assert "loss" in output
        assert "0.42" in output

    def test_blank(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.blank(3)
        assert stream.getvalue().count("\n") >= 3

    def test_command(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.command("sloughgpt serve", "Start the server")
        output = stream.getvalue()
        assert "sloughgpt serve" in output

    def test_divider(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.divider()
        output = stream.getvalue()
        assert "-" in output

    def test_terminal_disabled(self):
        import domains.logging.cli_logger as cli_mod
        original = cli_mod._TERMINAL_ENABLED
        try:
            cli_mod._TERMINAL_ENABLED = False
            stream = StringIO()
            log = CLILogger("slo.cli", stream=stream, colors=False)
            log.info("should not appear")
            log.success("should not appear")
            assert stream.getvalue() == ""
        finally:
            cli_mod._TERMINAL_ENABLED = original

    def test_timer(self):
        stream = StringIO()
        log = CLILogger("slo.cli", level=LogLevel.DEBUG, stream=stream, colors=False)
        with log.timer("test elapsed"):
            pass
        output = stream.getvalue()
        assert "elapsed" in output


# ── ShellLogger formatting ───────────────────────────────────────────────

class TestShellLogger:
    def test_emit_to_stream(self):
        stream = StringIO()
        log = ShellLogger("slo.shell", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello")
        output = stream.getvalue()
        assert "hello" in output

    def test_level_in_output(self):
        stream = StringIO()
        log = ShellLogger("slo.shell", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("msg")
        output = stream.getvalue()
        assert "INF" in output

    def test_context_in_output(self):
        stream = StringIO()
        log = ShellLogger("slo.shell", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("msg", model="gpt2")
        output = stream.getvalue()
        assert "model" in output

    def test_exception_in_output(self):
        stream = StringIO()
        log = ShellLogger("slo.shell", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.error("fail", exception="ValueError: bad")
        output = stream.getvalue()
        assert "ValueError" in output

    def test_logger_name_in_output(self):
        stream = StringIO()
        log = ShellLogger("slo.shell.repl", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("msg")
        output = stream.getvalue()
        assert "slo.shell.repl" in output


# ── WebLogger serialization ──────────────────────────────────────────────

class TestWebLogger:
    def test_record_to_dict(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.web", timestamp=100.0)
        d = log._record_to_dict(record)
        assert d["level"] == "info"
        assert d["message"] == "test"
        assert d["timestamp"] == 100.0

    def test_to_json(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.WARNING, message="warn msg", logger="slo.web")
        raw = log.to_json(record)
        data = json.loads(raw)
        assert data["level"] == "warning"
        assert data["message"] == "warn msg"

    def test_from_json_roundtrip(self):
        log = WebLogger("slo.web")
        original = LogRecord(
            level=LogLevel.ERROR,
            message="error msg",
            logger="slo.web.test",
            context={"key": "val"},
            exception="RuntimeError: boom",
        )
        raw = log.to_json(original)
        restored = log.from_json(raw)
        assert restored.level == LogLevel.ERROR
        assert restored.message == "error msg"
        assert restored.logger == "slo.web.test"
        assert restored.context == {"key": "val"}
        assert restored.exception == "RuntimeError: boom"

    def test_from_json_invalid(self):
        log = WebLogger("slo.web")
        record = log.from_json("not valid json {{{")
        assert record.level == LogLevel.WARNING
        assert record.message == "not valid json {{{"

    def test_format_brief(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="hello", logger="slo.web")
        brief = log._format_brief(record)
        assert "[slo.web]" in brief
        assert "hello" in brief

    def test_format_brief_with_context(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="hello", logger="slo.web", context={"a": 1})
        brief = log._format_brief(record)
        assert "a=1" in brief

    def test_format_brief_with_exception(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.ERROR, message="fail", logger="slo.web", exception="err")
        brief = log._format_brief(record)
        assert "err" in brief

    def test_emit_with_console(self):
        console = MagicMock()
        log = WebLogger("slo.web", console=console)
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.web")
        log.emit(record)
        console.log.assert_called_once()

    def test_emit_with_writable(self):
        writable = StringIO()
        log = WebLogger("slo.web", writable=writable)
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.web")
        log.emit(record)
        output = writable.getvalue()
        data = json.loads(output.strip())
        assert data["message"] == "test"

    def test_emit_no_console_no_writable(self):
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.web")
        log.emit(record)

    def test_console_method_mapping(self):
        log = WebLogger("slo.web")
        console = MagicMock()
        log._browser_console = console

        log.emit(LogRecord(level=LogLevel.DEBUG, message="d", logger="slo"))
        console.debug.assert_called_once()

        console.reset_mock()
        log.emit(LogRecord(level=LogLevel.WARNING, message="w", logger="slo"))
        console.warn.assert_called_once()

        console.reset_mock()
        log.emit(LogRecord(level=LogLevel.ERROR, message="e", logger="slo"))
        console.error.assert_called_once()


# ── BridgeHandler ────────────────────────────────────────────────────────

class TestBridgeHandler:
    def test_routes_to_our_logger(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        std_record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="hello %s", args=("world",), exc_info=None,
        )
        handler.emit(std_record)
        assert len(sink.records) == 1
        assert sink.records[0].message == "hello world"
        assert sink.records[0].level == LogLevel.INFO

    def test_level_mapping(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        for py_level, expected in [
            (logging.DEBUG, LogLevel.DEBUG),
            (logging.INFO, LogLevel.INFO),
            (logging.WARNING, LogLevel.WARNING),
            (logging.ERROR, LogLevel.ERROR),
            (logging.CRITICAL, LogLevel.CRITICAL),
        ]:
            std_record = logging.LogRecord(
                name="slo", level=py_level, pathname="", lineno=0,
                msg="test", args=(), exc_info=None,
            )
            handler.emit(std_record)
            assert sink.records[-1].level == expected

    def test_extra_context_captured(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        std_record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        std_record.custom_field = "value"
        handler.emit(std_record)
        assert sink.records[0].context.get("custom_field") == "value"

    def test_explicit_error_code(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        std_record = logging.LogRecord(
            name="slo", level=logging.ERROR, pathname="", lineno=0,
            msg="fail", args=(), exc_info=None,
        )
        std_record.error_code = "E_MODEL_OOM"
        handler.emit(std_record)
        assert sink.records[0].error_code == "E_MODEL_OOM"

    def test_explicit_tag(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        std_record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="load", args=(), exc_info=None,
        )
        std_record.tag = "MODEL"
        handler.emit(std_record)
        assert sink.records[0].tag == "MODEL"

    def test_exception_captured(self):
        sink = _SinkLogger(level=LogLevel.DEBUG)
        handler = BridgeHandler(sink)

        try:
            raise ValueError("bad")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        std_record = logging.LogRecord(
            name="slo", level=logging.ERROR, pathname="", lineno=0,
            msg="fail", args=(), exc_info=exc_info,
        )
        handler.emit(std_record)
        assert "ValueError" in sink.records[0].exception
        assert "bad" in sink.records[0].exception


class TestRecordExtraContext:
    def test_merges_explicit_context(self):
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.context = {"key": "val"}
        ctx = record_extra_context(record)
        assert ctx["key"] == "val"

    def test_captures_non_standard_extras(self):
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        record.custom = "value"
        ctx = record_extra_context(record)
        assert ctx["custom"] == "value"


# ── Config: request_id / log_context ─────────────────────────────────────

class TestConfigContextVars:
    def test_request_id_roundtrip(self):
        set_request_id("test-123")
        assert get_request_id() == "test-123"
        set_request_id(None)
        assert get_request_id() is None

    def test_log_context_roundtrip(self):
        clear_log_context()
        set_log_context(a=1, b=2)
        ctx = get_log_context()
        assert ctx["a"] == 1
        assert ctx["b"] == 2
        clear_log_context()

    def test_log_context_returns_copy(self):
        clear_log_context()
        set_log_context(x=1)
        ctx1 = get_log_context()
        ctx2 = get_log_context()
        ctx1["y"] = 2
        assert "y" not in ctx2
        clear_log_context()

    def test_clear_log_context(self):
        set_log_context(a=1)
        clear_log_context()
        assert get_log_context() == {}

    def test_set_log_context_merges(self):
        clear_log_context()
        set_log_context(a=1)
        set_log_context(b=2)
        ctx = get_log_context()
        assert ctx["a"] == 1
        assert ctx["b"] == 2
        clear_log_context()


# ── LogFormatter ──────────────────────────────────────────────────────────

class TestLogFormatter:
    def test_human_format(self):
        fmt = LogFormatter(colors=False)
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None, created=time.time(),
        )
        output = fmt.format(record)
        assert "INF" in output
        assert "hello" in output

    def test_human_format_with_tag(self):
        fmt = LogFormatter(colors=False)
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="loaded", args=(), exc_info=None, created=time.time(),
        )
        record.tag = "MODEL"
        output = fmt.format(record)
        assert "MODEL" in output

    def test_json_format(self):
        fmt = LogFormatter(fmt="json", colors=False)
        record = logging.LogRecord(
            name="slo.test", level=logging.INFO, pathname="", lineno=0,
            msg="test msg", args=(), exc_info=None, created=time.time(),
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["lvl"] == "INFO"
        assert data["msg"] == "test msg"

    def test_json_format_with_error_code(self):
        fmt = LogFormatter(fmt="json", colors=False)
        record = logging.LogRecord(
            name="slo.test", level=logging.ERROR, pathname="", lineno=0,
            msg="oom", args=(), exc_info=None, created=time.time(),
        )
        record.error_code = "E_MODEL_OOM"
        output = fmt.format(record)
        data = json.loads(output)
        assert data["lvl"] == "ERROR"
        assert data["msg"] == "oom"


# ── ClientExtensionFilter ─────────────────────────────────────────────────

class TestClientExtensionFilter:
    def test_filters_chrome_extension(self):
        f = ClientExtensionFilter()
        record = logging.LogRecord(
            name="slo", level=logging.WARNING, pathname="", lineno=0,
            msg="CLIENT ERROR chrome-extension://abc", args=(), exc_info=None,
        )
        assert f.filter(record) is False

    def test_filters_moz_extension(self):
        f = ClientExtensionFilter()
        record = logging.LogRecord(
            name="slo", level=logging.WARNING, pathname="", lineno=0,
            msg="moz-extension://abc failed", args=(), exc_info=None,
        )
        assert f.filter(record) is False

    def test_passes_normal_message(self):
        f = ClientExtensionFilter()
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="server started", args=(), exc_info=None,
        )
        assert f.filter(record) is True


# ── derive_op ─────────────────────────────────────────────────────────────

class TestDeriveOp:
    def test_explicit_op(self):
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        record.op = "model.load"
        assert _derive_op(record) == "model.load"

    def test_legacy_tag(self):
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        record.tag = "TRAIN"
        assert _derive_op(record) == _LEGACY_TAG_TO_OP["TRAIN"]

    def test_fallback(self):
        record = logging.LogRecord(
            name="slo", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        assert _derive_op(record) == "sys.info"


# ── __init__.py: factory and global logger ────────────────────────────────

from domains.logging import (
    get_logger,
    set_global,
    get_global,
    CompositeLogger as CLFromInit,
)


class TestLoggingFactory:
    def test_get_logger_api(self):
        log = get_logger("api", name="slo.test")
        assert isinstance(log, ConsoleLogger)

    def test_get_logger_cli(self):
        log = get_logger("cli", name="slo.test")
        assert isinstance(log, CLILogger)

    def test_get_logger_shell(self):
        log = get_logger("shell", name="slo.test")
        assert isinstance(log, ShellLogger)

    def test_get_logger_web(self):
        log = get_logger("web", name="slo.test")
        assert isinstance(log, WebLogger)

    def test_get_logger_aliases(self):
        assert isinstance(get_logger("server"), ConsoleLogger)
        assert isinstance(get_logger("console"), ConsoleLogger)
        assert isinstance(get_logger("repl"), ShellLogger)
        assert isinstance(get_logger("browser"), WebLogger)

    def test_get_logger_unknown(self):
        with pytest.raises(ValueError, match="Unknown logger interface"):
            get_logger("nonexistent")

    def test_set_global_get_global(self):
        custom = _SinkLogger(name="slo.custom")
        set_global(custom)
        assert get_global() is custom

    def test_get_global_returns_logger(self):
        log = get_global()
        assert isinstance(log, (ConsoleLogger, Logger))
