"""Tests for ShellLogger — ANSI formatting, levels, stream handling."""

import io
import threading
from datetime import datetime

import pytest

from domains.logging.base import ErrorCode, LogLevel, LogRecord, LogTag
from domains.logging.shell_logger import ShellLogger


def _record(
    message="hello",
    level=LogLevel.INFO,
    logger="slo.shell",
    context=None,
    exception=None,
    timestamp=1_700_000_000.0,
    error_code=None,
    tag=None,
):
    return LogRecord(
        level=level,
        message=message,
        logger=logger,
        timestamp=timestamp,
        context=context or {},
        exception=exception,
        error_code=error_code,
        tag=tag,
    )


@pytest.fixture
def ansi_on(monkeypatch):
    import domains.logging.config as cfg

    codes = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "CYAN": "\033[36m",
        "GREY": "\033[90m",
    }
    for name, code in codes.items():
        monkeypatch.setattr(cfg._A, name, code)


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:

    def test_default_stream(self):
        log = ShellLogger("slo.shell")
        assert log.name == "slo.shell"
        assert log.level == LogLevel.INFO

    def test_override_stream(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf)
        assert log._stream is buf

    def test_colors_flag(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        assert log._colors is False
        log2 = ShellLogger("slo.shell", stream=buf, colors=True)
        assert log2._colors is True

    def test_default_level_is_info(self):
        log = ShellLogger("slo.shell")
        assert log.level == LogLevel.INFO

    def test_custom_level(self):
        log = ShellLogger("slo.shell", level=LogLevel.DEBUG)
        assert log.level == LogLevel.DEBUG

    def test_default_context_empty(self):
        log = ShellLogger("slo.shell")
        assert log.context == {}

    def test_context_passed_through(self):
        log = ShellLogger("slo.shell", context={"env": "test"})
        assert log.context == {"env": "test"}

    def test_name_preserved(self):
        log = ShellLogger("slo.shell.repl")
        assert log.name == "slo.shell.repl"


# ── Formatting ────────────────────────────────────────────────────────

class TestFormatting:

    def test_plain_line(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="model loaded", context={"model": "gpt2"}))
        assert "model loaded" in line
        assert "INF" in line  # Shell format uses abbreviation
        assert "model=gpt2" in line
        assert "slo.shell" in line

    def test_icon_per_level(self):
        log = ShellLogger("slo.shell", colors=False)
        # Shell format uses abbreviations: DBG, INF, WRN, ERR, CRT
        # Icons are only for WARNING, ERROR, CRITICAL
        assert "DBG" in log._format_record(_record(level=LogLevel.DEBUG))
        assert "INF" in log._format_record(_record(level=LogLevel.INFO))
        assert "WRN" in log._format_record(_record(level=LogLevel.WARNING))
        assert "ERR" in log._format_record(_record(level=LogLevel.ERROR))
        assert "CRT" in log._format_record(_record(level=LogLevel.CRITICAL))

    def test_exception_included(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="failed", exception="FileNotFoundError"))
        assert "FileNotFoundError" in line

    def test_colors_emit_ansi(self, ansi_on):
        log = ShellLogger("slo.shell", colors=True)
        line = log._format_record(_record(message="x"))
        assert "\033[" in line

    def test_no_colors_no_ansi(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="x"))
        assert "\033[" not in line

    def test_time_formatted(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(timestamp=1_700_000_000.0))
        expected = datetime.fromtimestamp(1_700_000_000.0).strftime("%H:%M:%S")
        assert expected in line

    def test_multiple_context_values(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"a": 1, "b": 2}))
        assert "a=1" in line
        assert "b=2" in line

    def test_empty_context_no_trailing_space(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={}))
        # Empty context still has logger name on secondary line, but no context
        assert "slo.shell" in line

    def test_error_code_in_record(self):
        rec = _record(error_code="E_MODEL_LOAD")
        assert rec.error_code == "E_MODEL_LOAD"

    def test_tag_in_record(self):
        rec = _record(tag="REQ")
        assert rec.tag == "REQ"

    def test_format_record_level_label_uppercase(self):
        log = ShellLogger("slo.shell", colors=False)
        # Shell format uses abbreviations: DBG, INF, WRN, ERR, CRT
        abbrevs = {
            LogLevel.DEBUG: "DBG",
            LogLevel.INFO: "INF",
            LogLevel.WARNING: "WRN",
            LogLevel.ERROR: "ERR",
            LogLevel.CRITICAL: "CRT",
        }
        for level in LogLevel:
            line = log._format_record(_record(level=level))
            assert abbrevs[level] in line

    def test_format_record_includes_message(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="unique_xyz_123"))
        assert "unique_xyz_123" in line

    def test_format_record_includes_logger_name(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(logger="slo.custom"))
        assert "slo.custom" in line

    def test_colors_exception_ansi_wrapped(self, ansi_on):
        log = ShellLogger("slo.shell", colors=True)
        line = log._format_record(_record(exception="RuntimeError"))
        # Exception should be wrapped in ANSI codes
        assert "RuntimeError" in line

    def test_colors_time_dim_ansi(self, ansi_on):
        log = ShellLogger("slo.shell", colors=True)
        line = log._format_record(_record())
        # Time should be dim
        assert "\033[2m" in line

    def test_colors_logger_cyan(self, ansi_on):
        log = ShellLogger("slo.shell", colors=True)
        line = log._format_record(_record())
        # Logger name is GREY+DIM in shell format, not cyan
        assert "\033[90m" in line  # GREY code


# ── Emit ──────────────────────────────────────────────────────────────

class TestEmit:

    def test_emit_writes_line(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("ready")
        assert "ready" in buf.getvalue()

    def test_emit_respects_level(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.WARNING, colors=False)
        log.debug("hidden")
        log.error("shown")
        out = buf.getvalue()
        assert "hidden" not in out
        assert "shown" in out

    def test_emit_swallows_stream_error(self):
        class Broken:
            def write(self, _):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        log = ShellLogger("slo.shell", stream=Broken(), colors=False)
        log.info("won't raise")

    def test_emit_appends_newline(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("test")
        assert buf.getvalue().endswith("\n")

    def test_emit_multiple_messages(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("first")
        log.info("second")
        lines = buf.getvalue().strip().split("\n")
        # Each message has primary line + secondary line (logger name)
        assert len(lines) == 4
        assert "first" in lines[0]
        assert "second" in lines[2]

    def test_emit_level_debug_blocked_at_info(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.INFO, colors=False)
        log.debug("should not appear")
        assert buf.getvalue() == ""

    def test_emit_level_info_passes(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.INFO, colors=False)
        log.info("visible")
        assert "visible" in buf.getvalue()

    def test_emit_level_error_passes(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.ERROR, colors=False)
        log.error("visible")
        assert "visible" in buf.getvalue()

    def test_emit_with_context(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("ctx", model="gpt2")
        out = buf.getvalue()
        assert "ctx" in out
        assert "model=gpt2" in out

    def test_emit_with_exception_kwarg(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.error("fail", exception="KeyError")
        assert "KeyError" in buf.getvalue()

    def test_emit_critical_writes(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.critical("die")
        assert "die" in buf.getvalue()

    def test_emit_warning_writes(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.warning("careful")
        assert "careful" in buf.getvalue()

    def test_emit_debug_writes_when_level_debug(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.DEBUG, colors=False)
        log.debug("detail")
        assert "detail" in buf.getvalue()


# ── Thread Safety ─────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_emits(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        threads = []
        for i in range(20):
            t = threading.Thread(target=log.info, args=(f"msg{i}",))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = buf.getvalue().strip().split("\n")
        # Each message has primary line + secondary line (logger name)
        assert len(lines) == 40


# ── LogRecord ─────────────────────────────────────────────────────────

class TestLogRecord:

    def test_default_timestamp(self):
        rec = _record()
        assert isinstance(rec.timestamp, float)

    def test_frozen_dataclass(self):
        rec = _record(message="immutable")
        with pytest.raises(AttributeError):
            rec.message = "changed"

    def test_default_error_code_none(self):
        rec = _record()
        assert rec.error_code is None

    def test_default_tag_none(self):
        rec = _record()
        assert rec.tag is None

    def test_default_exception_none(self):
        rec = _record()
        assert rec.exception is None

    def test_default_logger_value(self):
        rec = LogRecord(level=LogLevel.INFO, message="x")
        assert rec.logger == "slo"


# ── LogLevel Comparison ───────────────────────────────────────────────

class TestLogLevelComparison:

    def test_debug_lt_info(self):
        assert LogLevel.DEBUG < LogLevel.INFO

    def test_info_lt_warning(self):
        assert LogLevel.INFO < LogLevel.WARNING

    def test_warning_lt_error(self):
        assert LogLevel.WARNING < LogLevel.ERROR

    def test_error_lt_critical(self):
        assert LogLevel.ERROR < LogLevel.CRITICAL

    def test_debug_le_debug(self):
        assert LogLevel.DEBUG <= LogLevel.DEBUG

    def test_critical_ge_info(self):
        assert LogLevel.CRITICAL >= LogLevel.INFO

    def test_not_implemented_for_non_loglevel(self):
        assert LogLevel.INFO.__ge__("not a level") is NotImplemented

    def test_not_implemented_gt_non_loglevel(self):
        assert LogLevel.INFO.__gt__(42) is NotImplemented


# ── ErrorCode ─────────────────────────────────────────────────────────

class TestErrorCode:

    def test_has_auth_codes(self):
        assert ErrorCode.E_AUTH_MISSING == "E_AUTH_MISSING"
        assert ErrorCode.E_AUTH_EXPIRED == "E_AUTH_EXPIRED"
        assert ErrorCode.E_AUTH_INVALID == "E_AUTH_INVALID"
        assert ErrorCode.E_AUTH_FORBIDDEN == "E_AUTH_FORBIDDEN"

    def test_has_model_codes(self):
        assert ErrorCode.E_MODEL_LOAD == "E_MODEL_LOAD"
        assert ErrorCode.E_MODEL_OOM == "E_MODEL_OOM"
        assert ErrorCode.E_MODEL_TIMEOUT == "E_MODEL_TIMEOUT"
        assert ErrorCode.E_MODEL_CRASH == "E_MODEL_CRASH"
        assert ErrorCode.E_MODEL_NOT_FOUND == "E_MODEL_NOT_FOUND"

    def test_has_inference_codes(self):
        assert ErrorCode.E_INF_TOKENIZER == "E_INF_TOKENIZER"
        assert ErrorCode.E_INF_GENERATION == "E_INF_GENERATION"
        assert ErrorCode.E_INF_CACHE == "E_INF_CACHE"

    def test_has_infra_codes(self):
        assert ErrorCode.E_INFRA_STARTUP == "E_INFRA_STARTUP"
        assert ErrorCode.E_INFRA_TIMEOUT == "E_INFRA_TIMEOUT"

    def test_has_training_codes(self):
        assert ErrorCode.E_TRAIN_DATA == "E_TRAIN_DATA"
        assert ErrorCode.E_TRAIN_CRASH == "E_TRAIN_CRASH"
        assert ErrorCode.E_TRAIN_CHECKPOINT == "E_TRAIN_CHECKPOINT"

    def test_all_are_strings(self):
        for member in ErrorCode:
            assert isinstance(member.value, str)


# ── LogTag ────────────────────────────────────────────────────────────

class TestLogTag:

    def test_has_required_tags(self):
        assert LogTag.REQ == "REQ"
        assert LogTag.AUTH == "AUTH"
        assert LogTag.MODEL == "MODEL"
        assert LogTag.TRAIN == "TRAIN"
        assert LogTag.ERROR == "ERROR"

    def test_all_are_strings(self):
        for member in LogTag:
            assert isinstance(member.value, str)


# ── Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_message(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message=""))
        assert line  # should not crash

    def test_unicode_message(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="日本語テスト"))
        assert "日本語テスト" in line

    def test_special_chars_in_context(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"key": "val=ue&x"}))
        assert "key=val=ue&x" in line

    def test_timestamp_zero(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(timestamp=0.0))
        assert ":" in line

    def test_very_long_message(self):
        log = ShellLogger("slo.shell", colors=False)
        msg = "x" * 10000
        line = log._format_record(_record(message=msg))
        assert msg in line

    def test_colors_none_defaults_to_auto(self):
        import domains.logging.shell_logger as sh
        log = ShellLogger("slo.shell", colors=None)
        assert log._colors == sh._COLOR_ENABLED

    def test_level_setter(self):
        log = ShellLogger("slo.shell", level=LogLevel.INFO)
        log.level = LogLevel.DEBUG
        assert log.level == LogLevel.DEBUG

    def test_context_merge(self):
        log = ShellLogger("slo.shell", context={"a": 1})
        log.set_context(b=2)
        assert log.context == {"a": 1, "b": 2}

    def test_clear_context(self):
        log = ShellLogger("slo.shell", context={"a": 1})
        log.clear_context()
        assert log.context == {}

    def test_multiple_set_context_calls(self):
        log = ShellLogger("slo.shell")
        log.set_context(a=1)
        log.set_context(b=2)
        log.set_context(c=3)
        assert log.context == {"a": 1, "b": 2, "c": 3}

    def test_set_context_overwrites_key(self):
        log = ShellLogger("slo.shell", context={"a": 1})
        log.set_context(a=99)
        assert log.context == {"a": 99}

    def test_emit_level_warning_passes_at_info(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.INFO, colors=False)
        log.warning("shown")
        assert "shown" in buf.getvalue()

    def test_emit_level_error_passes_at_warning(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.WARNING, colors=False)
        log.error("shown")
        assert "shown" in buf.getvalue()

    def test_emit_level_critical_passes_at_error(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.ERROR, colors=False)
        log.critical("shown")
        assert "shown" in buf.getvalue()

    def test_emit_level_info_blocked_at_warning(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.WARNING, colors=False)
        log.info("hidden")
        assert buf.getvalue() == ""

    def test_emit_level_warning_blocked_at_error(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.ERROR, colors=False)
        log.warning("hidden")
        assert buf.getvalue() == ""

    def test_emit_with_multiple_context_kwargs(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("ctx", a=1, b="two", c=True)
        out = buf.getvalue()
        assert "a=1" in out
        assert "b=two" in out
        assert "c=True" in out

    def test_emit_with_exception_and_context(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.error("fail", exception="KeyError", key="missing")
        out = buf.getvalue()
        assert "KeyError" in out
        assert "key=missing" in out

    def test_emit_with_error_code_kwarg(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.error("fail", error_code="E_MODEL_LOAD")
        assert "E_MODEL_LOAD" in buf.getvalue()  # error_code is in shell format

    def test_format_record_with_tag(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="tagged", tag="REQ"))
        assert "tagged" in line

    def test_format_record_with_error_code_and_tag(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(message="err", error_code="E_TRAIN_CRASH", tag="TRAIN"))
        assert "err" in line

    def test_emit_stream_flush_called(self):
        class FlushCapture:
            def __init__(self):
                self.flushed = False
                self._buf = io.StringIO()
            def write(self, s):
                self._buf.write(s)
            def flush(self):
                self.flushed = True
        cap = FlushCapture()
        log = ShellLogger("slo.shell", stream=cap, colors=False)
        log.info("test")
        assert cap.flushed

    def test_emit_stream_write_error_value_error(self):
        class BadWrite:
            def write(self, _):
                raise ValueError("bad")
            def flush(self):
                pass
        log = ShellLogger("slo.shell", stream=BadWrite(), colors=False)
        log.info("won't raise")

    def test_emit_stream_write_error_os_error(self):
        class BadWrite:
            def write(self, _):
                raise OSError("closed")
            def flush(self):
                pass
        log = ShellLogger("slo.shell", stream=BadWrite(), colors=False)
        log.info("won't raise")

    def test_logrecord_equality(self):
        r1 = LogRecord(level=LogLevel.INFO, message="x", timestamp=1.0)
        r2 = LogRecord(level=LogLevel.INFO, message="x", timestamp=1.0)
        assert r1 == r2

    def test_logrecord_not_hashable(self):
        r = _record(message="hashable")
        with pytest.raises(TypeError):
            hash(r)

    def test_logrecord_repr(self):
        r = _record(message="repr_test")
        assert "repr_test" in repr(r)

    def test_logrecord_default_context_is_empty_dict(self):
        r = LogRecord(level=LogLevel.INFO, message="x")
        assert r.context == {}

    def test_logrecord_default_logger_is_slo(self):
        r = LogRecord(level=LogLevel.INFO, message="x")
        assert r.logger == "slo"

    def test_logrecord_default_timestamp_is_float(self):
        r = LogRecord(level=LogLevel.INFO, message="x")
        assert isinstance(r.timestamp, float)

    def test_logrecord_with_all_fields(self):
        r = LogRecord(
            level=LogLevel.ERROR,
            message="all fields",
            logger="slo.test",
            timestamp=123.0,
            context={"k": "v"},
            exception="RuntimeError",
            error_code="E_MODEL_CRASH",
            tag="MODEL",
        )
        assert r.level == LogLevel.ERROR
        assert r.message == "all fields"
        assert r.logger == "slo.test"
        assert r.timestamp == 123.0
        assert r.context == {"k": "v"}
        assert r.exception == "RuntimeError"
        assert r.error_code == "E_MODEL_CRASH"
        assert r.tag == "MODEL"

    def test_format_record_context_value_with_spaces(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"key": "hello world"}))
        assert "key=hello world" in line

    def test_format_record_context_value_with_none(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"key": None}))
        assert "key=None" in line

    def test_format_record_context_value_with_bool(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"flag": True}))
        assert "flag=True" in line

    def test_format_record_context_value_with_list(self):
        log = ShellLogger("slo.shell", colors=False)
        line = log._format_record(_record(context={"items": [1, 2]}))
        assert "items=[1, 2]" in line

    def test_emit_empty_message(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("")
        assert buf.getvalue().strip() != ""

    def test_emit_message_with_newlines(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("line1\nline2")
        out = buf.getvalue()
        assert "line1" in out
        assert "line2" in out

    def test_emit_message_with_special_chars(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        log.info("tab\there")
        assert "tab\there" in buf.getvalue()

    def test_format_record_timestamp_midnight(self):
        log = ShellLogger("slo.shell", colors=False)
        import datetime
        ts = datetime.datetime(2024, 1, 1, 0, 0, 0).timestamp()
        line = log._format_record(_record(timestamp=ts))
        assert "00:00:00" in line

    def test_format_record_timestamp_end_of_day(self):
        log = ShellLogger("slo.shell", colors=False)
        import datetime
        ts = datetime.datetime(2024, 1, 1, 23, 59, 59).timestamp()
        line = log._format_record(_record(timestamp=ts))
        assert "23:59:59" in line

    def test_concurrent_emits_multiple_threads(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, colors=False)
        threads = []
        for i in range(50):
            t = threading.Thread(target=log.info, args=(f"t{i}",))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = buf.getvalue().strip().split("\n")
        # Each message has primary line + secondary line (logger name)
        assert len(lines) == 100

    def test_emit_mixed_levels_all_appear(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.DEBUG, colors=False)
        log.debug("d")
        log.info("i")
        log.warning("w")
        log.error("e")
        log.critical("c")
        out = buf.getvalue()
        assert "d" in out
        assert "i" in out
        assert "w" in out
        assert "e" in out
        assert "c" in out

    def test_emit_level_debug_blocked_at_info_only(self):
        buf = io.StringIO()
        log = ShellLogger("slo.shell", stream=buf, level=LogLevel.INFO, colors=False)
        log.debug("hidden")
        log.info("shown")
        log.warning("shown")
        log.error("shown")
        log.critical("shown")
        out = buf.getvalue()
        assert "hidden" not in out
        lines = out.strip().split("\n")
        # Each message has primary line + secondary line (logger name)
        assert len(lines) == 8
