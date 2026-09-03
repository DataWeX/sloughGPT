"""Tests for ShellLogger — ANSI logger for REPL."""
from __future__ import annotations

import io

from domains.logging.base import LogLevel
from domains.logging.shell_logger import ShellLogger


class TestShellLogger:
    def test_info_emits(self):
        stream = io.StringIO()
        log = ShellLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.info("hello")
        assert "hello" in stream.getvalue()

    def test_error_emits(self):
        stream = io.StringIO()
        log = ShellLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.error("oops")
        assert "oops" in stream.getvalue()

    def test_level_filtering(self):
        stream = io.StringIO()
        log = ShellLogger("test", level=LogLevel.WARNING, stream=stream, colors=False)
        log.info("silent")
        assert stream.getvalue() == ""

    def test_tag(self):
        stream = io.StringIO()
        log = ShellLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False)
        log.tag("MODEL").info("loaded")
        assert "loaded" in stream.getvalue()

    def test_context(self):
        stream = io.StringIO()
        log = ShellLogger("test", level=LogLevel.DEBUG, stream=stream, colors=False, context={"app": "cli"})
        log.info("started")
        assert "started" in stream.getvalue()
