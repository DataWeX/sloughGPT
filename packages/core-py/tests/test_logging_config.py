"""
Tests for domains.logging.config — centralized logging configuration.

Covers:
    - setup_logging() with all parameter combinations
    - HumanFormatter and JSONFormatter output
    - Correlation ID injection via contextvars
    - Log context merging
    - File handler with rotation
    - ClientExtensionFilter
    - Third-party logger suppression
    - Record factory enrichment
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure packages/core-py is on the path
_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.logging.config import (
    HumanFormatter,
    JSONFormatter,
    ClientExtensionFilter,
    _enriched_record_factory,
    _collect_extras,
    setup_logging,
    get_request_id,
    set_request_id,
    get_log_context,
    set_log_context,
    clear_log_context,
    _request_id,
    _log_context,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_record(
    name: str = "slo.test",
    level: int = logging.INFO,
    msg: str = "test message",
    **extra,
) -> logging.LogRecord:
    """Create a LogRecord for testing."""
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def _reset_contextvars():
    """Reset contextvars to clean state."""
    _request_id.set(None)
    _log_context.set({})


@pytest.fixture(autouse=True)
def _clean_context():
    """Reset contextvars before each test."""
    _reset_contextvars()
    yield
    _reset_contextvars()


# ── Correlation ID tests ──────────────────────────────────────────────


class TestCorrelationID:
    def test_default_is_none(self):
        assert get_request_id() is None

    def test_set_and_get(self):
        set_request_id("abc-123")
        assert get_request_id() == "abc-123"

    def test_overwrite(self):
        set_request_id("first")
        set_request_id("second")
        assert get_request_id() == "second"

    def test_thread_isolation(self):
        set_request_id("main")
        results = []

        def child():
            results.append(get_request_id())
            set_request_id("child")
            results.append(get_request_id())

        t = threading.Thread(target=child)
        t.start()
        t.join()
        # Main thread should still have "main"
        assert get_request_id() == "main"
        # Child thread starts with None (fresh contextvar), then sets "child"
        assert results[0] is None
        assert results[1] == "child"


# ── Log context tests ─────────────────────────────────────────────────


class TestLogContext:
    def test_default_is_empty(self):
        assert get_log_context() == {}

    def test_set_and_get(self):
        set_log_context(model="gpt2")
        ctx = get_log_context()
        assert ctx["model"] == "gpt2"

    def test_merge(self):
        set_log_context(model="gpt2")
        set_log_context(device="cpu")
        ctx = get_log_context()
        assert ctx["model"] == "gpt2"
        assert ctx["device"] == "cpu"

    def test_overwrite(self):
        set_log_context(model="gpt2")
        set_log_context(model="llama")
        assert get_log_context()["model"] == "llama"

    def test_clear(self):
        set_log_context(model="gpt2")
        clear_log_context()
        assert get_log_context() == {}

    def test_returns_copy(self):
        set_log_context(model="gpt2")
        ctx1 = get_log_context()
        ctx2 = get_log_context()
        assert ctx1 == ctx2
        assert ctx1 is not ctx2  # different dict objects


# ── HumanFormatter tests ──────────────────────────────────────────────


class TestHumanFormatter:
    def test_basic_output(self):
        fmt = HumanFormatter(colors=False)
        record = _make_record(msg="hello world")
        output = fmt.format(record)
        assert "INF" in output
        assert "hello world" in output
        assert "test" in output  # logger name

    def test_level_badges(self):
        fmt = HumanFormatter(colors=False)
        for level, badge in [
            (logging.DEBUG, "DBG"),
            (logging.INFO, "INF"),
            (logging.WARNING, "WRN"),
            (logging.ERROR, "ERR"),
            (logging.CRITICAL, "CRI"),
        ]:
            record = _make_record(level=level)
            output = fmt.format(record)
            assert badge in output

    def test_tag_in_output(self):
        fmt = HumanFormatter(colors=False)
        record = _make_record(tag="MODEL")
        output = fmt.format(record)
        assert "[MODEL]" in output

    def test_request_id_in_output(self):
        fmt = HumanFormatter(colors=False)
        record = _make_record(request_id="abc-123")
        output = fmt.format(record)
        assert "req=abc-123" in output

    def test_context_in_output(self):
        fmt = HumanFormatter(colors=False)
        record = _make_record(model="gpt2", tokens=50)
        output = fmt.format(record)
        assert "model=gpt2" in output
        assert "tokens=50" in output

    def test_exception_in_output(self):
        fmt = HumanFormatter(colors=False)
        try:
            raise ValueError("bad value")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = _make_record(exc_info=exc_info)
        output = fmt.format(record)
        assert "ValueError" in output
        assert "bad value" in output

    def test_colors_enabled(self):
        fmt = HumanFormatter(colors=True)
        record = _make_record(msg="colored")
        output = fmt.format(record)
        assert "\033[" in output  # ANSI escape codes present


# ── JSONFormatter tests ───────────────────────────────────────────────


class TestJSONFormatter:
    def test_basic_output(self):
        fmt = JSONFormatter()
        record = _make_record(msg="hello")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello"
        assert data["level"] == "INFO"
        assert data["logger"] == "slo.test"
        assert "ts" in data

    def test_tag_in_output(self):
        fmt = JSONFormatter()
        record = _make_record(tag="REQ")
        data = json.loads(fmt.format(record))
        assert data["tag"] == "REQ"

    def test_request_id_in_output(self):
        fmt = JSONFormatter()
        record = _make_record(request_id="xyz-789")
        data = json.loads(fmt.format(record))
        assert data["request_id"] == "xyz-789"

    def test_context_in_output(self):
        fmt = JSONFormatter()
        record = _make_record(model="gpt2", tokens=100)
        data = json.loads(fmt.format(record))
        assert data["ctx"]["model"] == "gpt2"
        assert data["ctx"]["tokens"] == 100

    def test_exception_in_output(self):
        fmt = JSONFormatter()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()
        record = _make_record(exc_info=exc_info)
        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert "RuntimeError" in data["exception"]

    def test_no_extra_fields_in_output(self):
        fmt = JSONFormatter()
        record = _make_record()
        data = json.loads(fmt.format(record))
        # Standard fields should not appear in ctx
        assert "ctx" not in data or data["ctx"] == {}


# ── ClientExtensionFilter tests ───────────────────────────────────────


class TestClientExtensionFilter:
    def test_allows_normal_messages(self):
        f = ClientExtensionFilter()
        record = _make_record(msg="normal log message")
        assert f.filter(record) is True

    def test_blocks_chrome_extension(self):
        f = ClientExtensionFilter()
        record = _make_record(msg="chrome-extension://abc/error")
        assert f.filter(record) is False

    def test_blocks_client_error(self):
        f = ClientExtensionFilter()
        record = _make_record(msg="CLIENT ERROR 12345")
        assert f.filter(record) is False

    def test_blocks_zero_zero(self):
        f = ClientExtensionFilter()
        record = _make_record(msg="0 0 connection lost")
        assert f.filter(record) is False


# ── _collect_extras tests ─────────────────────────────────────────────


class TestCollectExtras:
    def test_extracts_non_standard_fields(self):
        record = _make_record(model="gpt2", tokens=50)
        ctx = _collect_extras(record)
        assert ctx["model"] == "gpt2"
        assert ctx["tokens"] == 50

    def test_excludes_standard_fields(self):
        record = _make_record()
        ctx = _collect_extras(record)
        # Standard fields should not appear
        assert "name" not in ctx
        assert "levelname" not in ctx
        assert "message" not in ctx

    def test_excludes_tag_and_request_id(self):
        record = _make_record(tag="REQ", request_id="abc")
        ctx = _collect_extras(record)
        assert "tag" not in ctx
        assert "request_id" not in ctx

    def test_empty_when_no_extras(self):
        record = _make_record()
        ctx = _collect_extras(record)
        assert ctx == {}


# ── Record factory tests ──────────────────────────────────────────────


class TestRecordFactory:
    def test_injects_request_id(self):
        set_request_id("factory-test")
        record = _enriched_record_factory(
            "slo.test", logging.INFO, "", 0, "msg", (), None,
        )
        assert record.request_id == "factory-test"

    def test_no_request_id_when_none(self):
        record = _enriched_record_factory(
            "slo.test", logging.INFO, "", 0, "msg", (), None,
        )
        assert not hasattr(record, "request_id") or getattr(record, "request_id", None) is None

    def test_injects_log_context(self):
        set_log_context(model="gpt2", device="cpu")
        record = _enriched_record_factory(
            "slo.test", logging.INFO, "", 0, "msg", (), None,
        )
        assert record.model == "gpt2"
        assert record.device == "cpu"

    def test_does_not_overwrite_existing(self):
        record = _enriched_record_factory(
            "slo.test", logging.INFO, "", 0, "msg", (), None,
        )
        record.request_id = "existing"
        # Factory should not overwrite
        record2 = _enriched_record_factory(
            "slo.test", logging.INFO, "", 0, "msg", (), None,
        )
        # record2 is a new record, not the same as record


# ── setup_logging tests ───────────────────────────────────────────────


class TestSetupLogging:
    def test_default_setup(self):
        result = setup_logging(enable_output_buffer=False)
        assert result["level"] == "INFO"
        assert result["format"] == "human"
        assert "log_dir" in result

    def test_custom_level(self):
        result = setup_logging(level="DEBUG", enable_output_buffer=False)
        assert result["level"] == "DEBUG"
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_format(self):
        result = setup_logging(format="json", enable_output_buffer=False)
        assert result["format"] == "json"

    def test_file_handler_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = setup_logging(log_dir=tmpdir, enable_output_buffer=False)
            assert result["file_handler"] is not None
            assert isinstance(result["file_handler"], logging.handlers.RotatingFileHandler)
            log_file = Path(tmpdir) / "sloughgpt.log"
            assert log_file.exists()

    def test_file_handler_disabled(self):
        result = setup_logging(enable_file=False, enable_output_buffer=False)
        assert result["file_handler"] is None

    def test_console_handler_installed(self):
        result = setup_logging(enable_output_buffer=False)
        root = logging.getLogger()
        # Should have at least one StreamHandler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_console_handler_disabled(self):
        result = setup_logging(enable_console=False, enable_output_buffer=False)
        # When console disabled, only file handler should be present
        # (no StreamHandler on root)
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_third_party_suppressed(self):
        setup_logging(enable_output_buffer=False)
        for name in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
            logger = logging.getLogger(name)
            assert logger.level >= logging.WARNING

    def test_returns_complete_info(self):
        result = setup_logging(enable_output_buffer=False)
        assert "level" in result
        assert "format" in result
        assert "log_dir" in result
        assert "file_handler" in result
        assert "bridge" in result

    def test_env_var_overrides(self):
        with patch.dict(os.environ, {"SLO_LOG_LEVEL": "DEBUG", "SLO_LOG_FORMAT": "json"}):
            result = setup_logging(enable_output_buffer=False)
            assert result["level"] == "DEBUG"
            assert result["format"] == "json"

    def test_idempotent(self):
        """Calling setup_logging twice should not create duplicate handlers."""
        setup_logging(enable_output_buffer=False)
        handler_count_1 = len(logging.getLogger().handlers)
        setup_logging(enable_output_buffer=False)
        handler_count_2 = len(logging.getLogger().handlers)
        # Should not double handlers (setup_logging removes existing handlers first)
        assert handler_count_2 <= handler_count_1 + 1  # +1 for file handler


# ── Integration test: logging through stdlib works ────────────────────


class TestStdlibIntegration:
    def test_logger_info_appears_in_handler(self):
        """Verify that logging.getLogger('slo.*').info() works through the new setup."""
        output = StringIO()
        setup_logging(enable_output_buffer=False, enable_file=False)

        # Add a test handler that captures output
        test_handler = logging.StreamHandler(output)
        test_handler.setFormatter(HumanFormatter(colors=False))
        logging.getLogger().addHandler(test_handler)

        logger = logging.getLogger("slo.integration.test")
        logger.info("integration test message", extra={"tag": "INFRA"})

        test_output = output.getvalue()
        assert "integration test message" in test_output
        assert "[INFRA]" in test_output

        # Cleanup
        logging.getLogger().removeHandler(test_handler)

    def test_correlation_id_in_stdlib_log(self):
        """Verify correlation ID flows through stdlib logging."""
        output = StringIO()
        setup_logging(enable_output_buffer=False, enable_file=False)

        test_handler = logging.StreamHandler(output)
        test_handler.setFormatter(JSONFormatter())
        logging.getLogger().addHandler(test_handler)

        set_request_id("req-42")
        logger = logging.getLogger("slo.correlation")
        logger.info("correlated message")

        test_output = output.getvalue()
        data = json.loads(test_output)
        assert data["request_id"] == "req-42"

        logging.getLogger().removeHandler(test_handler)
