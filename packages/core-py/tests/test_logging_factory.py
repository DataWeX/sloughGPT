"""Tests for domains.logging — factory functions, global logger, LogLevel, LogRecord, Logger ABC, ChildLogger, TaggedLogger, CompositeLogger, ErrorCode, LogTag."""

import io
import time
import threading

import pytest

from domains.logging import (
    get_logger, set_global, get_global,
    ConsoleLogger, CLILogger, ShellLogger, WebLogger,
    LogLevel,
)
from domains.logging.base import (
    Logger, LogRecord, ChildLogger, TaggedLogger, CompositeLogger,
    ErrorCode, LogTag,
)
from domains.logging.config import (
    get_request_id, set_request_id, get_log_context, set_log_context, clear_log_context,
)


# ---------------------------------------------------------------------------
# get_logger factory
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_api_returns_console(self):
        log = get_logger("api")
        assert isinstance(log, ConsoleLogger)

    def test_server_returns_console(self):
        log = get_logger("server")
        assert isinstance(log, ConsoleLogger)

    def test_console_returns_console(self):
        log = get_logger("console")
        assert isinstance(log, ConsoleLogger)

    def test_cli_returns_cli(self):
        log = get_logger("cli")
        assert isinstance(log, CLILogger)

    def test_shell_returns_shell(self):
        log = get_logger("shell")
        assert isinstance(log, ShellLogger)

    def test_repl_returns_shell(self):
        log = get_logger("repl")
        assert isinstance(log, ShellLogger)

    def test_web_returns_web(self):
        log = get_logger("web")
        assert isinstance(log, WebLogger)

    def test_browser_returns_web(self):
        log = get_logger("browser")
        assert isinstance(log, WebLogger)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown logger interface"):
            get_logger("nonexistent")

    def test_custom_name(self):
        log = get_logger("api", name="slo.custom")
        assert log.name == "slo.custom"

    def test_custom_level(self):
        log = get_logger("api", level=LogLevel.DEBUG)
        assert log.level == LogLevel.DEBUG

    def test_default_level_is_info(self):
        log = get_logger("api")
        assert log.level == LogLevel.INFO

    def test_default_name_is_slo(self):
        log = get_logger("api")
        assert log.name == "slo"

    def test_case_insensitive_interface(self):
        log = get_logger("API")
        assert isinstance(log, ConsoleLogger)

    def test_case_insensitive_cli(self):
        log = get_logger("CLI")
        assert isinstance(log, CLILogger)

    def test_case_insensitive_web(self):
        log = get_logger("Web")
        assert isinstance(log, WebLogger)

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            get_logger("nonexistent_logger")

    def test_empty_interface_raises(self):
        with pytest.raises(ValueError, match="Unknown logger interface"):
            get_logger("")

    def test_multiple_calls_create_new_instances(self):
        log1 = get_logger("api", name="a")
        log2 = get_logger("api", name="b")
        assert log1 is not log2

    def test_level_passed_through(self):
        log = get_logger("api", level=LogLevel.WARNING)
        assert log.level == LogLevel.WARNING


# ---------------------------------------------------------------------------
# LogLevel
# ---------------------------------------------------------------------------

class TestLogLevel:
    def test_five_members(self):
        assert len(LogLevel) == 5

    def test_values(self):
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_ordering_ge(self):
        assert LogLevel.DEBUG >= LogLevel.DEBUG
        assert LogLevel.INFO >= LogLevel.DEBUG
        assert LogLevel.CRITICAL >= LogLevel.ERROR

    def test_ordering_gt(self):
        assert LogLevel.INFO > LogLevel.DEBUG
        assert LogLevel.WARNING > LogLevel.INFO
        assert not (LogLevel.DEBUG > LogLevel.DEBUG)

    def test_ordering_le(self):
        assert LogLevel.DEBUG <= LogLevel.INFO
        assert LogLevel.ERROR <= LogLevel.CRITICAL

    def test_ordering_lt(self):
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.WARNING < LogLevel.ERROR

    def test_ordering_not_implemented_for_non_loglevel(self):
        assert LogLevel.INFO.__ge__(42) is NotImplemented
        assert LogLevel.INFO.__gt__(42) is NotImplemented
        assert LogLevel.INFO.__le__(42) is NotImplemented
        assert LogLevel.INFO.__lt__(42) is NotImplemented

    def test_comparison_with_str(self):
        assert LogLevel.INFO.__ge__("info") is NotImplemented

    def test_unique_values(self):
        values = [l.value for l in LogLevel]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------

class TestLogRecord:
    def test_creation_minimal(self):
        r = LogRecord(level=LogLevel.INFO, message="test")
        assert r.level == LogLevel.INFO
        assert r.message == "test"
        assert r.logger == "slo"
        assert r.timestamp > 0
        assert r.context == {}
        assert r.exception is None
        assert r.error_code is None
        assert r.tag is None

    def test_creation_full(self):
        r = LogRecord(
            level=LogLevel.ERROR,
            message="boom",
            logger="slo.api",
            timestamp=12345.0,
            context={"key": "val"},
            exception="RuntimeError: oops",
            error_code="E_MODEL_LOAD",
            tag="MODEL",
        )
        assert r.level == LogLevel.ERROR
        assert r.message == "boom"
        assert r.logger == "slo.api"
        assert r.timestamp == 12345.0
        assert r.context == {"key": "val"}
        assert r.exception == "RuntimeError: oops"
        assert r.error_code == "E_MODEL_LOAD"
        assert r.tag == "MODEL"

    def test_frozen(self):
        r = LogRecord(level=LogLevel.INFO, message="test")
        with pytest.raises(AttributeError):
            r.message = "changed"

    def test_default_timestamp_is_recent(self):
        before = time.time()
        r = LogRecord(level=LogLevel.INFO, message="test")
        after = time.time()
        assert before <= r.timestamp <= after

    def test_context_independent_instances(self):
        r1 = LogRecord(level=LogLevel.INFO, message="a", context={"x": 1})
        r2 = LogRecord(level=LogLevel.INFO, message="b", context={"x": 2})
        assert r1.context["x"] == 1
        assert r2.context["x"] == 2


# ---------------------------------------------------------------------------
# ErrorCode
# ---------------------------------------------------------------------------

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
        assert ErrorCode.E_MODEL_CRASH.value == "E_MODEL_CRASH"
        assert ErrorCode.E_MODEL_NOT_FOUND.value == "E_MODEL_NOT_FOUND"
        assert ErrorCode.E_MODEL_WARMUP.value == "E_MODEL_WARMUP"

    def test_inference_codes(self):
        assert ErrorCode.E_INF_TOKENIZER.value == "E_INF_TOKENIZER"
        assert ErrorCode.E_INF_GENERATION.value == "E_INF_GENERATION"
        assert ErrorCode.E_INF_CACHE.value == "E_INF_CACHE"

    def test_infra_codes(self):
        assert ErrorCode.E_INFRA_STARTUP.value == "E_INFRA_STARTUP"
        assert ErrorCode.E_INFRA_TIMEOUT.value == "E_INFRA_TIMEOUT"
        assert ErrorCode.E_INFRA_REGISTRY.value == "E_INFRA_REGISTRY"
        assert ErrorCode.E_INFRA_PROVIDER.value == "E_INFRA_PROVIDER"

    def test_validation_codes(self):
        assert ErrorCode.E_VAL_REQUEST.value == "E_VAL_REQUEST"
        assert ErrorCode.E_VAL_FIELD.value == "E_VAL_FIELD"

    def test_training_codes(self):
        assert ErrorCode.E_TRAIN_DATA.value == "E_TRAIN_DATA"
        assert ErrorCode.E_TRAIN_CRASH.value == "E_TRAIN_CRASH"
        assert ErrorCode.E_TRAIN_CHECKPOINT.value == "E_TRAIN_CHECKPOINT"

    def test_domain_codes(self):
        assert ErrorCode.E_DOMAIN.value == "E_DOMAIN"
        assert ErrorCode.E_NOT_FOUND.value == "E_NOT_FOUND"
        assert ErrorCode.E_CONFLICT.value == "E_CONFLICT"

    def test_is_str(self):
        assert isinstance(ErrorCode.E_MODEL_LOAD, str)
        assert ErrorCode.E_MODEL_LOAD == "E_MODEL_LOAD"

    def test_unique_values(self):
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))

    def test_all_are_pascalcase_prefixed(self):
        for code in ErrorCode:
            assert code.value.startswith("E_")


# ---------------------------------------------------------------------------
# LogTag
# ---------------------------------------------------------------------------

class TestLogTag:
    def test_all_members(self):
        tags = [t.value for t in LogTag]
        assert "REQ" in tags
        assert "AUTH" in tags
        assert "MODEL" in tags
        assert "SOUL" in tags
        assert "TRAIN" in tags
        assert "INFRA" in tags
        assert "START" in tags
        assert "SLOW" in tags
        assert "ERROR" in tags
        assert "WARN" in tags
        assert "OK" in tags

    def test_unique_values(self):
        values = [t.value for t in LogTag]
        assert len(values) == len(set(values))

    def test_is_str(self):
        assert isinstance(LogTag.REQ, str)
        assert LogTag.REQ == "REQ"


# ---------------------------------------------------------------------------
# Global logger
# ---------------------------------------------------------------------------

class TestGlobalLogger:
    def test_set_and_get(self):
        log = ConsoleLogger("slo.test_global")
        set_global(log)
        assert get_global() is log

    def test_get_creates_default(self):
        set_global(None)
        import domains.logging as pkg
        pkg._global_logger = None
        log = get_global()
        assert isinstance(log, ConsoleLogger)
        pkg._global_logger = None

    def test_set_none_gets_default(self):
        set_global(None)
        log = get_global()
        assert isinstance(log, ConsoleLogger)
        set_global(None)

    def test_set_multiple_times(self):
        log1 = ConsoleLogger("slo.a")
        log2 = ConsoleLogger("slo.b")
        set_global(log1)
        assert get_global() is log1
        set_global(log2)
        assert get_global() is log2


# ---------------------------------------------------------------------------
# Logger ABC — concrete subclass for testing
# ---------------------------------------------------------------------------

class _CaptureLogger(Logger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestLoggerABC:
    def test_debug(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.debug("hello")
        assert len(log.records) == 1
        assert log.records[0].level == LogLevel.DEBUG
        assert log.records[0].message == "hello"

    def test_info(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.info("hello")
        assert log.records[0].level == LogLevel.INFO

    def test_warning(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.warning("hello")
        assert log.records[0].level == LogLevel.WARNING

    def test_error(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.error("hello")
        assert log.records[0].level == LogLevel.ERROR

    def test_critical(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.critical("hello")
        assert log.records[0].level == LogLevel.CRITICAL

    def test_level_filtering(self):
        log = _CaptureLogger(name="test", level=LogLevel.WARNING)
        log.debug("a")
        log.info("b")
        log.warning("c")
        log.error("d")
        assert len(log.records) == 2
        assert log.records[0].message == "c"
        assert log.records[1].message == "d"

    def test_exception(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        try:
            raise ValueError("oops")
        except ValueError as e:
            log.exception("failed", exc=e)
        assert len(log.records) == 1
        assert "ValueError" in log.records[0].exception
        assert "oops" in log.records[0].exception

    def test_context_merged(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG, context={"a": 1})
        log.info("msg", b=2)
        ctx = log.records[0].context
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_set_context(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.set_context(x=10)
        log.info("msg")
        assert log.records[0].context["x"] == 10

    def test_clear_context(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG, context={"a": 1})
        log.clear_context()
        log.info("msg")
        assert "a" not in log.records[0].context

    def test_error_code_passthrough(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.error("fail", error_code="E_MODEL_LOAD")
        assert log.records[0].error_code == "E_MODEL_LOAD"

    def test_repr(self):
        log = _CaptureLogger(name="test", level=LogLevel.INFO)
        r = repr(log)
        assert "CaptureLogger" in r
        assert "test" in r


# ---------------------------------------------------------------------------
# ChildLogger
# ---------------------------------------------------------------------------

class TestChildLogger:
    def test_emits_to_parent(self):
        parent = _CaptureLogger(name="slo.parent", level=LogLevel.DEBUG)
        child = parent.child("sub")
        child.info("from child")
        assert len(parent.records) == 1
        assert parent.records[0].message == "from child"
        assert parent.records[0].logger == "slo.parent.sub"

    def test_name_concatenation(self):
        parent = _CaptureLogger(name="slo.api")
        child = parent.child("inference")
        assert child.name == "slo.api.inference"

    def test_level_follows_parent(self):
        parent = _CaptureLogger(name="slo.parent", level=LogLevel.WARNING)
        child = parent.child("sub")
        assert child.level == LogLevel.WARNING
        parent.level = LogLevel.DEBUG
        assert child.level == LogLevel.DEBUG

    def test_context_merged(self):
        parent = _CaptureLogger(name="slo.parent", level=LogLevel.DEBUG, context={"a": 1})
        child = parent.child("sub", b=2)
        child.info("msg")
        ctx = parent.records[0].context
        assert ctx["a"] == 1
        assert ctx["b"] == 2


# ---------------------------------------------------------------------------
# TaggedLogger
# ---------------------------------------------------------------------------

class TestTaggedLogger:
    def test_tag_attached(self):
        parent = _CaptureLogger(name="slo.test", level=LogLevel.DEBUG)
        tagged = parent.tag("REQ")
        tagged.info("handled")
        assert parent.records[0].tag == "REQ"

    def test_emits_to_parent(self):
        parent = _CaptureLogger(name="slo.test", level=LogLevel.DEBUG)
        tagged = parent.tag("MODEL")
        tagged.info("loaded")
        assert len(parent.records) == 1
        assert parent.records[0].message == "loaded"

    def test_level_follows_parent(self):
        parent = _CaptureLogger(name="slo.test", level=LogLevel.WARNING)
        tagged = parent.tag("REQ")
        assert tagged.level == LogLevel.WARNING

    def test_explicit_tag_kwarg_becomes_context(self):
        parent = _CaptureLogger(name="slo.test", level=LogLevel.DEBUG)
        tagged = parent.tag("REQ")
        tagged.info("msg", tag="AUTH")
        assert parent.records[0].tag == "REQ"
        assert parent.records[0].context.get("tag") == "AUTH"

    def test_context_preserved(self):
        parent = _CaptureLogger(name="slo.test", level=LogLevel.DEBUG, context={"x": 1})
        tagged = parent.tag("REQ")
        tagged.info("msg")
        assert parent.records[0].context["x"] == 1


# ---------------------------------------------------------------------------
# CompositeLogger
# ---------------------------------------------------------------------------

class TestCompositeLogger:
    def test_emits_to_all_children(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        c2 = _CaptureLogger(name="b", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo.composite", children=[c1, c2])
        composite.info("hello")
        assert len(c1.records) == 1
        assert len(c2.records) == 1

    def test_add(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo.composite", level=LogLevel.DEBUG)
        composite.add(c1)
        composite.info("hello")
        assert len(c1.records) == 1

    def test_remove(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        c2 = _CaptureLogger(name="b", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo.composite", children=[c1, c2], level=LogLevel.DEBUG)
        composite.remove(c1)
        composite.info("hello")
        assert len(c1.records) == 0
        assert len(c2.records) == 1

    def test_children_property(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo.composite", children=[c1])
        assert len(composite.children) == 1

    def test_empty_composite(self):
        composite = CompositeLogger(name="slo.composite", level=LogLevel.DEBUG)
        composite.info("no children")  # should not raise
        assert len(composite.children) == 0

    def test_add_chaining(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        c2 = _CaptureLogger(name="b", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo.composite", level=LogLevel.DEBUG)
        result = composite.add(c1).add(c2)
        assert result is composite
        assert len(composite.children) == 2


# ---------------------------------------------------------------------------
# ConsoleLogger
# ---------------------------------------------------------------------------

class TestConsoleLogger:
    def test_emit_to_stream(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.info("hello world")
        output = stream.getvalue()
        assert "hello world" in output

    def test_json_format(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.info("structured", key="value")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["msg"] == "structured"
        assert data["level"] == "INFO"

    def test_json_includes_tag(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.tag("REQ").info("handled")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["tag"] == "REQ"

    def test_json_includes_error_code(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.error("fail", error_code="E_MODEL_LOAD")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["code"] == "E_MODEL_LOAD"

    def test_json_includes_context(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.info("msg", port=8000)
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["ctx"]["port"] == 8000

    def test_json_includes_exception(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.error("fail", exception="RuntimeError: oops")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["err"] == "RuntimeError: oops"

    def test_human_format_with_context(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.info("msg", key="val")
        output = stream.getvalue()
        assert "key=val" in output

    def test_human_format_with_tag(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.tag("MODEL").info("loaded")
        output = stream.getvalue()
        assert "[MODEL]" in output

    def test_human_format_with_error_code(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.error("fail", error_code="E_MODEL_OOM")
        output = stream.getvalue()
        assert "E_MODEL_OOM" in output

    def test_colors_disabled(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False)
        log.info("no color")
        output = stream.getvalue()
        assert "\033[" not in output

    def test_colors_enabled(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=True)
        log.info("with color")
        output = stream.getvalue()
        assert "\033[" in output

    def test_repr(self):
        log = ConsoleLogger("slo.test")
        assert "ConsoleLogger" in repr(log)


# ---------------------------------------------------------------------------
# CLILogger
# ---------------------------------------------------------------------------

class TestCLILogger:
    def test_emit_to_stream(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.info("hello")
        output = stream.getvalue()
        assert "hello" in output

    def test_success(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.success("done")
        output = stream.getvalue()
        assert "done" in output

    def test_step(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.step("building")
        output = stream.getvalue()
        assert "building" in output

    def test_header(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.header("Title")
        output = stream.getvalue()
        assert "Title" in output

    def test_section(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.section("Part 1")
        output = stream.getvalue()
        assert "Part 1" in output

    def test_table(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        output = stream.getvalue()
        assert "Alice" in output
        assert "Bob" in output

    def test_table_empty_no_output(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.table(["Name"], [])
        assert stream.getvalue() == ""

    def test_json_output(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.json({"key": "value"})
        output = stream.getvalue()
        assert "key" in output
        assert "value" in output

    def test_status_ok(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.status("Model", "gpt2", status="ok")
        output = stream.getvalue()
        assert "Model" in output
        assert "gpt2" in output

    def test_divider(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.divider()
        output = stream.getvalue()
        assert "-" in output

    def test_key_value(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.key_value("model", "gpt2")
        output = stream.getvalue()
        assert "model" in output
        assert "gpt2" in output

    def test_key_value_empty_key(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.key_value("", "value")
        output = stream.getvalue()
        assert "value" in output

    def test_blank(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.blank(3)
        output = stream.getvalue()
        assert output.count("\n") == 3

    def test_command(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.command("train", "train a model")
        output = stream.getvalue()
        assert "train" in output
        assert "train a model" in output

    def test_timer(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False, level=LogLevel.DEBUG)
        with log.timer("elapsed"):
            pass
        output = stream.getvalue()
        assert "elapsed" in output

    def test_cursor_hide_show(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.hide_cursor()
        log.show_cursor()

    def test_colors_disabled(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.info("no color")
        output = stream.getvalue()
        assert "\033[" not in output

    def test_colors_enabled(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=True)
        log.info("with color")
        output = stream.getvalue()
        assert "\033[" in output


# ---------------------------------------------------------------------------
# ShellLogger
# ---------------------------------------------------------------------------

class TestShellLogger:
    def test_emit_to_stream(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=False)
        log.info("hello")
        output = stream.getvalue()
        assert "hello" in output

    def test_format_includes_logger_name(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell.test", stream=stream, colors=False)
        log.info("msg")
        output = stream.getvalue()
        assert "slo.shell.test" in output

    def test_format_includes_level(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=False)
        log.warning("warn")
        output = stream.getvalue()
        assert "WARNING" in output

    def test_context_shown(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=False)
        log.info("msg", key="val")
        output = stream.getvalue()
        assert "key=val" in output

    def test_exception_shown(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=False)
        log.error("fail", exception="RuntimeError: oops")
        output = stream.getvalue()
        assert "RuntimeError: oops" in output

    def test_colors_disabled(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=False)
        log.info("no color")
        output = stream.getvalue()
        assert "\033[" not in output

    def test_colors_enabled(self):
        stream = io.StringIO()
        log = ShellLogger("slo.shell", stream=stream, colors=True)
        log.info("with color")
        output = stream.getvalue()
        assert "with color" in output
        assert "INFO" in output


# ---------------------------------------------------------------------------
# WebLogger
# ---------------------------------------------------------------------------

class TestWebLogger:
    def test_emit_to_writable(self):
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("hello")
        output = stream.getvalue().strip()
        assert "hello" in output

    def test_emit_json_structure(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("test msg")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "info"
        assert data["message"] == "test msg"
        assert data["logger"] == "slo.web"

    def test_to_json(self):
        import json
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="test", logger="slo.web")
        raw = log.to_json(record)
        data = json.loads(raw)
        assert data["message"] == "test"

    def test_from_json(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "info", "message": "hello", "logger": "slo.web", "timestamp": 1.0})
        record = log.from_json(data)
        assert record.message == "hello"
        assert record.level == LogLevel.INFO

    def test_from_json_invalid(self):
        log = WebLogger("slo.web")
        record = log.from_json("not valid json {{{")
        assert record.level == LogLevel.WARNING
        assert record.message == "not valid json {{{"

    def test_from_json_with_context(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "error", "message": "fail", "logger": "slo.web", "timestamp": 1.0, "context": {"key": "val"}})
        record = log.from_json(data)
        assert record.context["key"] == "val"

    def test_from_json_with_exception(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "error", "message": "fail", "logger": "slo.web", "timestamp": 1.0, "exception": "RuntimeError: oops"})
        record = log.from_json(data)
        assert record.exception == "RuntimeError: oops"

    def test_emit_with_console(self):
        class FakeConsole:
            def __init__(self):
                self.calls = []
            def log(self, *args):
                self.calls.append(("log", args))
        console = FakeConsole()
        log = WebLogger("slo.web", console=console)
        log.info("hello")
        assert len(console.calls) == 1
        assert "hello" in str(console.calls[0])

    def test_emit_with_console_error(self):
        class FakeConsole:
            def __init__(self):
                self.calls = []
            def error(self, *args):
                self.calls.append(("error", args))
        console = FakeConsole()
        log = WebLogger("slo.web", console=console)
        log.error("fail")
        assert len(console.calls) == 1

    def test_emit_with_console_warn(self):
        class FakeConsole:
            def __init__(self):
                self.calls = []
            def warn(self, *args):
                self.calls.append(("warn", args))
        console = FakeConsole()
        log = WebLogger("slo.web", console=console)
        log.warning("warn")
        assert len(console.calls) == 1

    def test_emit_with_console_debug(self):
        class FakeConsole:
            def __init__(self):
                self.calls = []
            def debug(self, *args):
                self.calls.append(("debug", args))
        console = FakeConsole()
        log = WebLogger("slo.web", console=console, level=LogLevel.DEBUG)
        log.debug("dbg")
        assert len(console.calls) == 1


# ---------------------------------------------------------------------------
# Config — request_id and log context
# ---------------------------------------------------------------------------

class TestConfigContext:
    def test_set_get_request_id(self):
        set_request_id("req-123")
        assert get_request_id() == "req-123"

    def test_clear_request_id(self):
        set_request_id("req-456")
        set_request_id(None)
        # None should be accepted
        assert get_request_id() is None

    def test_set_get_log_context(self):
        clear_log_context()
        set_log_context(key="val")
        ctx = get_log_context()
        assert ctx["key"] == "val"

    def test_log_context_merges(self):
        clear_log_context()
        set_log_context(a=1)
        set_log_context(b=2)
        ctx = get_log_context()
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_clear_log_context(self):
        clear_log_context()
        set_log_context(a=1)
        clear_log_context()
        ctx = get_log_context()
        assert ctx == {}

    def test_get_log_context_returns_copy(self):
        clear_log_context()
        set_log_context(a=1)
        ctx1 = get_log_context()
        ctx2 = get_log_context()
        ctx1["b"] = 2
        assert "b" not in get_log_context()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_emit(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, level=LogLevel.DEBUG)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    log.info(f"msg-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        lines = stream.getvalue().strip().split("\n")
        assert len(lines) == 50


# ---------------------------------------------------------------------------
# Extended Logger ABC tests
# ---------------------------------------------------------------------------

class TestLoggerABCExtended:
    def test_level_setter(self):
        log = _CaptureLogger(name="test", level=LogLevel.INFO)
        log.level = LogLevel.WARNING
        assert log.level == LogLevel.WARNING

    def test_context_property(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG, context={"x": 1})
        assert log.context["x"] == 1

    def test_context_mutable(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.set_context(y=2)
        assert log.context["y"] == 2

    def test_debug_not_emitted_when_level_info(self):
        log = _CaptureLogger(name="test", level=LogLevel.INFO)
        log.debug("should not appear")
        assert len(log.records) == 0

    def test_info_emitted_when_level_debug(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.info("visible")
        assert len(log.records) == 1

    def test_warning_emitted_when_level_info(self):
        log = _CaptureLogger(name="test", level=LogLevel.INFO)
        log.warning("visible")
        assert len(log.records) == 1

    def test_error_emitted_when_level_warning(self):
        log = _CaptureLogger(name="test", level=LogLevel.WARNING)
        log.error("visible")
        assert len(log.records) == 1

    def test_critical_emitted_when_level_error(self):
        log = _CaptureLogger(name="test", level=LogLevel.ERROR)
        log.critical("visible")
        assert len(log.records) == 1

    def test_multiple_context_sources_merged(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG, context={"a": 1})
        log.info("msg", b=2, c=3)
        ctx = log.records[0].context
        assert ctx["a"] == 1
        assert ctx["b"] == 2
        assert ctx["c"] == 3

    def test_exception_includes_traceback(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        try:
            raise TypeError("bad type")
        except TypeError as e:
            log.exception("failed", exc=e)
        assert "TypeError" in log.records[0].exception
        assert "bad type" in log.records[0].exception

    def test_error_code_on_debug(self):
        log = _CaptureLogger(name="test", level=LogLevel.DEBUG)
        log.debug("dbg", error_code="E_DEBUG")
        assert log.records[0].error_code == "E_DEBUG"

    def test_repr_contains_name(self):
        log = _CaptureLogger(name="slo.mylogger", level=LogLevel.WARNING)
        assert "slo.mylogger" in repr(log)

    def test_repr_contains_level(self):
        log = _CaptureLogger(name="test", level=LogLevel.ERROR)
        r = repr(log)
        assert "error" in r


# ---------------------------------------------------------------------------
# Extended ChildLogger tests
# ---------------------------------------------------------------------------

class TestChildLoggerExtended:
    def test_nested_children(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        child = parent.child("api")
        grandchild = child.child("inference")
        grandchild.info("nested")
        assert parent.records[0].logger == "slo.api.inference"

    def test_child_context_does_not_mutate_parent(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG, context={"a": 1})
        child = parent.child("sub", b=2)
        assert "b" not in parent.context

    def test_child_debug_respects_parent_level(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.WARNING)
        child = parent.child("sub")
        child.debug("hidden")
        child.info("hidden")
        child.warning("visible")
        assert len(parent.records) == 1

    def test_child_error_code_passthrough(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        child = parent.child("sub")
        child.error("fail", error_code="E_TEST")
        assert parent.records[0].error_code == "E_TEST"

    def test_child_exception_passthrough(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        child = parent.child("sub")
        try:
            raise ValueError("oops")
        except ValueError as e:
            child.exception("fail", exc=e)
        assert "ValueError" in parent.records[0].exception


# ---------------------------------------------------------------------------
# Extended TaggedLogger tests
# ---------------------------------------------------------------------------

class TestTaggedLoggerExtended:
    def test_tag_on_debug(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("MODEL")
        tagged.debug("dbg")
        assert parent.records[0].tag == "MODEL"

    def test_tag_on_error(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("ERROR")
        tagged.error("fail")
        assert parent.records[0].tag == "ERROR"

    def test_tag_on_critical(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("CRIT")
        tagged.critical("bad")
        assert parent.records[0].tag == "CRIT"

    def test_tag_on_warning(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("WARN")
        tagged.warning("warn")
        assert parent.records[0].tag == "WARN"

    def test_tag_context_merged(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG, context={"a": 1})
        tagged = parent.tag("REQ")
        tagged.info("msg", b=2)
        ctx = parent.records[0].context
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_tag_error_code_passthrough(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("REQ")
        tagged.error("fail", error_code="E_AUTH")
        assert parent.records[0].error_code == "E_AUTH"

    def test_tag_exception_passthrough(self):
        parent = _CaptureLogger(name="slo", level=LogLevel.DEBUG)
        tagged = parent.tag("REQ")
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            tagged.exception("fail", exc=e)
        assert "RuntimeError" in parent.records[0].exception


# ---------------------------------------------------------------------------
# Extended CompositeLogger tests
# ---------------------------------------------------------------------------

class TestCompositeLoggerExtended:
    def test_composite_level_filtering(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        c2 = _CaptureLogger(name="b", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo", children=[c1, c2], level=LogLevel.WARNING)
        composite.info("hidden")
        composite.warning("visible")
        assert len(c1.records) == 1
        assert len(c2.records) == 1

    def test_composite_context_merge(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo", children=[c1], level=LogLevel.DEBUG, context={"x": 1})
        composite.info("msg", y=2)
        assert c1.records[0].context["x"] == 1
        assert c1.records[0].context["y"] == 2

    def test_composite_remove_nonexistent_raises(self):
        c1 = _CaptureLogger(name="a", level=LogLevel.DEBUG)
        composite = CompositeLogger(name="slo", children=[c1], level=LogLevel.DEBUG)
        try:
            composite.remove(_CaptureLogger(name="b", level=LogLevel.DEBUG))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_composite_multiple_children(self):
        children = [_CaptureLogger(name=f"c{i}", level=LogLevel.DEBUG) for i in range(5)]
        composite = CompositeLogger(name="slo", children=children, level=LogLevel.DEBUG)
        composite.info("broadcast")
        for c in children:
            assert len(c.records) == 1


# ---------------------------------------------------------------------------
# Extended ConsoleLogger tests
# ---------------------------------------------------------------------------

class TestConsoleLoggerExtended:
    def test_default_name(self):
        log = ConsoleLogger("slo.test")
        assert log.name == "slo.test"

    def test_default_level(self):
        log = ConsoleLogger("slo.test")
        assert log.level == LogLevel.INFO

    def test_human_format_with_exception(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.error("fail", exception="RuntimeError: oops")
        output = stream.getvalue()
        assert "[RuntimeError]" in output
        assert "oops" in output

    def test_json_level_mapping(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.warning("warn")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "WARN"

    def test_json_critical_level(self):
        import json
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, format="json")
        log.critical("crit")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "CRIT"

    def test_human_format_multiple_context(self):
        stream = io.StringIO()
        log = ConsoleLogger("slo.test", stream=stream, colors=False, format="human")
        log.info("msg", a=1, b=2, c=3)
        output = stream.getvalue()
        assert "a=1" in output
        assert "b=2" in output
        assert "c=3" in output


# ---------------------------------------------------------------------------
# Extended CLILogger tests
# ---------------------------------------------------------------------------

class TestCLILoggerExtended:
    def test_json_format(self):
        import json
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.json({"key": "value"})
        data = json.loads(stream.getvalue().strip())
        assert data["key"] == "value"

    def test_json_nested(self):
        import json
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.json({"nested": {"a": 1}})
        data = json.loads(stream.getvalue().strip())
        assert data["nested"]["a"] == 1

    def test_status_error(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.status("Model", "gpt2", status="error")
        output = stream.getvalue()
        assert "Model" in output

    def test_blank_zero(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.blank(0)
        assert stream.getvalue() == ""

    def test_blank_default(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.blank()
        assert stream.getvalue() == "\n"

    def test_command_only_name(self):
        stream = io.StringIO()
        log = CLILogger("slo.cli", stream=stream, colors=False)
        log.command("train")
        output = stream.getvalue()
        assert "train" in output


# ---------------------------------------------------------------------------
# Extended WebLogger tests
# ---------------------------------------------------------------------------

class TestWebLoggerExtended:
    def test_emit_json_includes_timestamp(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("msg")
        data = json.loads(stream.getvalue().strip())
        assert "timestamp" in data

    def test_emit_json_includes_context(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("msg", key="val")
        data = json.loads(stream.getvalue().strip())
        assert data["context"]["key"] == "val"

    def test_emit_json_includes_logger(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("msg")
        data = json.loads(stream.getvalue().strip())
        assert data["logger"] == "slo.web"

    def test_emit_json_includes_timestamp(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.info("msg")
        data = json.loads(stream.getvalue().strip())
        assert "timestamp" in data

    def test_emit_json_error_level(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.error("fail")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "error"

    def test_emit_json_warning_level(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.warning("warn")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "warning"

    def test_emit_json_debug_level(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream, level=LogLevel.DEBUG)
        log.debug("dbg")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "debug"

    def test_emit_json_critical_level(self):
        import json
        stream = io.StringIO()
        log = WebLogger("slo.web", writable=stream)
        log.critical("crit")
        data = json.loads(stream.getvalue().strip())
        assert data["level"] == "critical"

    def test_from_json_with_context(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "info", "message": "hi", "logger": "slo.web", "timestamp": 1.0, "context": {"key": "val"}})
        record = log.from_json(data)
        assert record.context["key"] == "val"

    def test_from_json_with_exception(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "error", "message": "fail", "logger": "slo.web", "timestamp": 1.0, "exception": "RuntimeError: oops"})
        record = log.from_json(data)
        assert record.exception == "RuntimeError: oops"

    def test_from_json_missing_fields(self):
        import json
        log = WebLogger("slo.web")
        data = json.dumps({"level": "info"})
        record = log.from_json(data)
        assert record.message == ""
        assert record.logger == "slo"

    def test_from_json_invalid_json(self):
        log = WebLogger("slo.web")
        record = log.from_json("not valid json {{{")
        assert record.level == LogLevel.WARNING
        assert record.message == "not valid json {{{"

    def test_to_json_includes_exception(self):
        import json
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.ERROR, message="fail", logger="slo.web", exception="RuntimeError: oops")
        raw = log.to_json(record)
        data = json.loads(raw)
        assert data["exception"] == "RuntimeError: oops"

    def test_to_json_includes_context(self):
        import json
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.INFO, message="msg", logger="slo.web", context={"key": "val"})
        raw = log.to_json(record)
        data = json.loads(raw)
        assert data["context"]["key"] == "val"

    def test_to_json_roundtrip(self):
        import json
        log = WebLogger("slo.web")
        record = LogRecord(level=LogLevel.ERROR, message="test", logger="slo.web", context={"a": 1})
        raw = log.to_json(record)
        restored = log.from_json(raw)
        assert restored.message == "test"
        assert restored.level == LogLevel.ERROR
        assert restored.context["a"] == 1


# ---------------------------------------------------------------------------
# Extended Config context tests
# ---------------------------------------------------------------------------

class TestConfigContextExtended:
    def test_set_request_id_overwrites(self):
        set_request_id("req-1")
        set_request_id("req-2")
        assert get_request_id() == "req-2"

    def test_clear_request_id(self):
        set_request_id("req-1")
        clear_log_context()
        assert get_log_context() == {}

    def test_log_context_replaces_key(self):
        clear_log_context()
        set_log_context(a=1)
        set_log_context(a=2)
        assert get_log_context()["a"] == 2

    def test_request_id_in_context(self):
        from domains.logging.base import Logger, LogLevel, LogRecord

        class _TestLogger(Logger):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        set_request_id("req-123")
        log = _TestLogger(name="test", level=LogLevel.DEBUG)
        log.info("msg")
        assert log.records[0].context["request_id"] == "req-123"
        set_request_id(None)

    def test_log_context_in_record(self):
        from domains.logging.base import Logger, LogLevel, LogRecord

        class _TestLogger(Logger):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        clear_log_context()
        set_log_context(env="test")
        log = _TestLogger(name="test", level=LogLevel.DEBUG)
        log.info("msg")
        assert log.records[0].context["env"] == "test"
        clear_log_context()
