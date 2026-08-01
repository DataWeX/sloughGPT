"""Tests for logging/bridge.py — BridgeHandler routing stdlib logging to Logger."""

import logging

from domains.logging.base import LogLevel, LogRecord, Logger
from domains.logging.bridge import BridgeHandler, _LEVEL_MAP


class RecordingLogger(Logger):
    """Captures emitted LogRecords for assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record: LogRecord) -> None:
        self.records.append(record)


class TestLevelMap:
    def test_maps_all_standard_levels(self):
        assert _LEVEL_MAP[logging.DEBUG] == LogLevel.DEBUG
        assert _LEVEL_MAP[logging.INFO] == LogLevel.INFO
        assert _LEVEL_MAP[logging.WARNING] == LogLevel.WARNING
        assert _LEVEL_MAP[logging.ERROR] == LogLevel.ERROR
        assert _LEVEL_MAP[logging.CRITICAL] == LogLevel.CRITICAL

    def test_unknown_level_defaults_to_info(self):
        record = logging.LogRecord("x", 0, __file__, 1, "msg", (), None)
        handler = BridgeHandler(RecordingLogger("slo"))
        handler.emit(record)
        assert handler._logger.records[0].level == LogLevel.INFO


class TestBridgeHandler:
    def make_handler(self):
        logger = RecordingLogger("slo.test", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        return handler, logger

    def test_emit_routes_level_message_and_logger_name(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo.api.inference", logging.INFO, "f.py", 10, "hello", (), None)
        handler.emit(record)
        assert len(logger.records) == 1
        out = logger.records[0]
        assert out.level == LogLevel.INFO
        assert out.message == "hello"
        assert out.logger == "slo.api.inference"

    def test_emit_adds_path_and_line_for_debug(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.DEBUG, "src/mod.py", 42, "trace", (), None)
        handler.emit(record)
        assert logger.records[0].context["path"] == "src/mod.py"
        assert logger.records[0].context["line"] == 42

    def test_emit_omits_path_and_line_above_debug(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.INFO, "src/mod.py", 42, "trace", (), None)
        handler.emit(record)
        assert "path" not in logger.records[0].context
        assert "line" not in logger.records[0].context

    def test_emit_merges_extra_context(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.INFO, "f.py", 1, "loaded", (), None)
        record.context = {"model": "gpt2", "device": "cpu"}
        handler.emit(record)
        assert logger.records[0].context["model"] == "gpt2"
        assert logger.records[0].context["device"] == "cpu"

    def test_emit_ignores_non_dict_extra_context(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.INFO, "f.py", 1, "msg", (), None)
        record.context = "not-a-dict"
        handler.emit(record)
        assert logger.records[0].context == {}

    def test_emit_captures_exception_info(self):
        handler, logger = self.make_handler()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord("slo", logging.ERROR, "f.py", 1, "failed", (),
                                       exc_info=logging.sys.exc_info())
        handler.emit(record)
        assert logger.records[0].exception == "ValueError: boom"

    def test_emit_captures_exc_text(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.ERROR, "f.py", 1, "failed", (), None)
        record.exc_text = "manual trace"
        handler.emit(record)
        assert logger.records[0].exception == "manual trace"

    def test_emit_extracts_error_code_and_tag_from_extra(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.ERROR, "f.py", 1, "OOM", (), None)
        record.error_code = "E_MODEL_OOM"
        record.tag = "MODEL"
        handler.emit(record)
        assert logger.records[0].error_code == "E_MODEL_OOM"
        assert logger.records[0].tag == "MODEL"

    def test_emit_without_error_code_and_tag(self):
        handler, logger = self.make_handler()
        record = logging.LogRecord("slo", logging.INFO, "f.py", 1, "ok", (), None)
        handler.emit(record)
        assert logger.records[0].error_code is None
        assert logger.records[0].tag is None


class TestEndToEnd:
    def test_stdlib_logger_routes_through_bridge(self):
        logger = RecordingLogger("slo.e2e", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        root = logging.getLogger("slo.e2e.bridge-test")
        root.handlers = []
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        root.info("via stdlib")
        assert logger.records[-1].message == "via stdlib"
        assert logger.records[-1].level == LogLevel.INFO

    def test_error_with_extra_dict_via_stdlib(self):
        logger = RecordingLogger("slo.e2e", level=LogLevel.DEBUG)
        handler = BridgeHandler(logger)
        root = logging.getLogger("slo.e2e.bridge-extra")
        root.handlers = []
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        root.error("failed", extra={"context": {"model": "gpt2"}})
        rec = logger.records[-1]
        assert rec.message == "failed"
        assert rec.context["model"] == "gpt2"
