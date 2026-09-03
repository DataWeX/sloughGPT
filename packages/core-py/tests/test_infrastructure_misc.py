"""Tests for correlation ID, SessionCore, and ConsoleLogger."""
from __future__ import annotations

import asyncio
import io

from domains.infrastructure.correlation import get_correlation_id, set_correlation_id
from domains.infrastructure.session_core import SessionCore
from domains.logging.base import LogLevel
from domains.logging.console_logger import ConsoleLogger, _default_color_enabled


class TestCorrelationId:
    def test_default_is_none(self):
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"

    def test_overwrite(self):
        set_correlation_id("first")
        set_correlation_id("second")
        assert get_correlation_id() == "second"

    def test_context_var_isolation(self):
        set_correlation_id("in-main")

        async def child():
            assert get_correlation_id() == "in-main"
            set_correlation_id("in-child")
            assert get_correlation_id() == "in-child"

        asyncio.run(child())
        assert get_correlation_id() == "in-main"


class TestSessionCore:
    def test_store_and_get(self):
        result = SessionCore.store_context("s1", [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        assert result["status"] == "stored"
        assert result["message_count"] == 2
        msgs = SessionCore.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hello"

    def test_get_empty_session(self):
        msgs = SessionCore.get_messages("nonexistent")
        assert msgs == []

    def test_overwrite_session(self):
        SessionCore.store_context("s2", [{"role": "user", "content": "first"}])
        SessionCore.store_context("s2", [{"role": "user", "content": "second"}])
        msgs = SessionCore.get_messages("s2")
        assert msgs[0]["content"] == "second"

    def test_list_sessions(self):
        SessionCore.store_context("s3", [{"role": "user", "content": "x"}])
        sessions = SessionCore.list_sessions()
        assert isinstance(sessions, list)


class TestConsoleLogger:
    def test_info_emits(self):
        stream = io.StringIO()
        log = ConsoleLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello world")
        output = stream.getvalue()
        assert "hello world" in output

    def test_error_emits(self):
        stream = io.StringIO()
        log = ConsoleLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.error("something failed")
        assert "something failed" in stream.getvalue()

    def test_level_filtering(self):
        stream = io.StringIO()
        log = ConsoleLogger("test", level=LogLevel.WARNING, stream=stream, colors=False)
        log.info("should not appear")
        assert stream.getvalue() == ""

    def test_tag(self):
        stream = io.StringIO()
        log = ConsoleLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.tag("MODEL").info("loaded")
        assert "loaded" in stream.getvalue()

    def test_json_format(self):
        stream = io.StringIO()
        log = ConsoleLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False, format="json")
        log.info("test message")
        output = stream.getvalue()
        assert "test message" in output

    def test_default_color_disabled(self):
        stream = io.StringIO()
        result = _default_color_enabled(stream)
        assert result is False
