"""Tests for WebLogger — console delegation, JSON serialization, round-trip."""

import io
import json

from domains.logging.base import LogLevel, LogRecord
from domains.logging.web_logger import WebLogger


def _record(
    message="hello",
    level=LogLevel.INFO,
    logger="slo.web.chat",
    context=None,
    exception=None,
    timestamp=1_700_000_000.0,
):
    return LogRecord(
        level=level,
        message=message,
        logger=logger,
        timestamp=timestamp,
        context=context or {},
        exception=exception,
    )


class TestConstruction:

    def test_defaults(self):
        log = WebLogger("slo.web")
        assert log.name == "slo.web"
        assert log._browser_console is None
        assert log._writable is None


class TestSerialization:

    def test_record_to_dict_basic(self):
        log = WebLogger("slo.web")
        d = log._record_to_dict(_record(message="msg", context={"a": 1}))
        assert d["level"] == "info"
        assert d["logger"] == "slo.web.chat"
        assert d["message"] == "msg"
        assert d["context"] == {"a": 1}
        assert d["timestamp"] == 1_700_000_000.0

    def test_record_to_dict_exception(self):
        log = WebLogger("slo.web")
        d = log._record_to_dict(_record(exception="AbortError: timeout"))
        assert d["exception"] == "AbortError: timeout"

    def test_to_json_round_trip(self):
        log = WebLogger("slo.web")
        record = _record(message="hi", context={"x": 1})
        raw = log.to_json(record)
        parsed = json.loads(raw)
        assert parsed["message"] == "hi"
        assert parsed["context"] == {"x": 1}

    def test_from_json_restores_record(self):
        log = WebLogger("slo.web")
        raw = log.to_json(_record(message="hi", level=LogLevel.ERROR, context={"a": 1}))
        record = log.from_json(raw)
        assert isinstance(record, LogRecord)
        assert record.message == "hi"
        assert record.level == LogLevel.ERROR
        assert record.context == {"a": 1}
        assert record.timestamp == 1_700_000_000.0

    def test_from_json_omitted_fields_default(self):
        log = WebLogger("slo.web")
        record = log.from_json(json.dumps({"level": "debug", "message": "x"}))
        assert record.logger == "slo"
        assert record.exception is None


class TestEmit:

    def test_console_delegation(self):
        calls = []

        class FakeConsole:
            def debug(self, *a):
                calls.append(("debug", a))

            def log(self, *a):
                calls.append(("log", a))

            def warn(self, *a):
                calls.append(("warn", a))

            def error(self, *a):
                calls.append(("error", a))

        log = WebLogger("slo.web", console=FakeConsole())
        log.info("sent", session_id="abc")
        assert calls[0][0] == "log"
        assert "[slo.web] sent" in calls[0][1][0]
        assert calls[0][1][1]["context"] == {"session_id": "abc"}

    def test_console_method_mapping(self):
        calls = []

        class FakeConsole:
            def debug(self, *a):
                calls.append(("debug",))

            def log(self, *a):
                calls.append(("log",))

            def warn(self, *a):
                calls.append(("warn",))

            def error(self, *a):
                calls.append(("error",))

        log = WebLogger("slo.web", console=FakeConsole(), level=LogLevel.DEBUG)
        log.debug("d")
        log.warning("w")
        log.error("e")
        log.critical("c")
        assert [c[0] for c in calls] == ["debug", "warn", "error", "error"]

    def test_writable_fallback_json_line(self):
        buf = io.StringIO()
        log = WebLogger("slo.web", writable=buf)
        log.info("json line", request_id="r1")
        data = json.loads(buf.getvalue().strip())
        assert data["message"] == "json line"
        assert data["context"] == {"request_id": "r1"}

    def test_console_takes_precedence_over_writable(self):
        calls = []

        class FakeConsole:
            def log(self, *a):
                calls.append(a)

        buf = io.StringIO()
        log = WebLogger("slo.web", console=FakeConsole(), writable=buf)
        log.info("only console")
        assert calls
        assert buf.getvalue() == ""

    def test_writable_swallows_error(self):
        class Broken:
            def write(self, _):
                raise OSError("closed")

            def flush(self):
                raise OSError("closed")

        log = WebLogger("slo.web", writable=Broken())
        log.info("won't raise")

    def test_missing_console_method_skips(self):
        class BareConsole:
            pass

        buf = io.StringIO()
        log = WebLogger("slo.web", console=BareConsole(), writable=buf)
        log.info("fallthrough")
        assert "fallthrough" in buf.getvalue()


class TestFormatBrief:

    def test_brief_format(self):
        log = WebLogger("slo.web")
        out = log._format_brief(_record(message="hi", context={"a": 1}))
        assert "[slo.web.chat] (a=1) hi" == out

    def test_brief_with_exception(self):
        log = WebLogger("slo.web")
        out = log._format_brief(_record(message="hi", exception="Err: x"))
        assert "— Err: x" in out
