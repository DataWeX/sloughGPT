"""Tests for domains.shell.audit — structured JSONL audit logger."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from domains.shell.audit import (
    ShellAuditLogger,
    get_shell_audit_logger,
    _audit,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton."""
    import domains.shell.audit as mod
    mod._audit = None
    yield
    mod._audit = None


@pytest.fixture
def audit_log(tmp_path):
    """Create a ShellAuditLogger writing to a temp directory."""
    return ShellAuditLogger(log_dir=tmp_path, log_file="test_audit.jsonl")


def _read_events(log_path: Path) -> list[dict]:
    """Parse all JSONL lines from the audit log."""
    events = []
    for line in log_path.read_text().strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))
    return events


class TestShellAuditLoggerInit:
    def test_creates_log_dir(self, tmp_path):
        target = tmp_path / "sub" / "audit"
        ShellAuditLogger(log_dir=target)
        assert target.is_dir()

    def test_default_log_dir(self):
        logger = ShellAuditLogger()
        assert logger.log_path.name == "shell_audit.jsonl"

    def test_custom_log_file(self, tmp_path):
        logger = ShellAuditLogger(log_dir=tmp_path, log_file="custom.jsonl")
        assert logger.log_path.name == "custom.jsonl"

    def test_session_id_is_timestamp(self, audit_log):
        assert audit_log._session_id.isdigit()
        assert len(audit_log._session_id) >= 13  # millisecond timestamp

    def test_cmd_count_starts_at_zero(self, audit_log):
        assert audit_log._cmd_count == 0


class TestCommand:
    def test_emits_shell_command_event(self, audit_log):
        audit_log.command("ls -la", "ls", "-la", 0)
        events = _read_events(audit_log.log_path)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "shell.command"
        assert e["line"] == "ls -la"
        assert e["cmd"] == "ls"
        assert e["args"] == "-la"
        assert e["exit_code"] == 0
        assert e["cmd_num"] == 1

    def test_increments_cmd_count(self, audit_log):
        audit_log.command("a", "a", "", 0)
        audit_log.command("b", "b", "", 0)
        assert audit_log._cmd_count == 2

    def test_elapsed_ms(self, audit_log):
        audit_log.command("slow", "slow", "", 0, elapsed_ms=123.4)
        events = _read_events(audit_log.log_path)
        assert events[0]["elapsed_ms"] == 123.4

    def test_elapsed_ms_none(self, audit_log):
        audit_log.command("fast", "fast", "", 0)
        events = _read_events(audit_log.log_path)
        assert events[0]["elapsed_ms"] is None

    def test_expanded_field(self, audit_log):
        audit_log.command("echo $X", "echo", "$X", 0, expanded="echo hello")
        events = _read_events(audit_log.log_path)
        assert events[0]["expanded"] == "echo hello"

    def test_is_background(self, audit_log):
        audit_log.command("sleep 10 &", "sleep", "10", 0, is_background=True)
        events = _read_events(audit_log.log_path)
        assert events[0]["is_background"] is True

    def test_is_pipeline(self, audit_log):
        audit_log.command("a | b", "a", "| b", 0, is_pipeline=True)
        events = _read_events(audit_log.log_path)
        assert events[0]["is_pipeline"] is True


class TestEval:
    def test_emits_shell_eval_event(self, audit_log):
        audit_log.eval("1+1", "2", 0)
        events = _read_events(audit_log.log_path)
        e = events[0]
        assert e["event"] == "shell.eval"
        assert e["expression"] == "1+1"
        assert e["result_preview"] == "2"
        assert e["exit_code"] == 0
        assert e["cmd_num"] == 1

    def test_result_preview_truncated(self, audit_log):
        long_result = "x" * 500
        audit_log.eval("test", long_result, 0)
        events = _read_events(audit_log.log_path)
        assert len(events[0]["result_preview"]) == 200


class TestError:
    def test_emits_shell_error_event(self, audit_log):
        audit_log.error("rm -rf /", "permission denied")
        events = _read_events(audit_log.log_path)
        e = events[0]
        assert e["event"] == "shell.error"
        assert e["line"] == "rm -rf /"
        assert e["error"] == "permission denied"

    def test_error_does_not_increment_cmd_count(self, audit_log):
        audit_log.error("bad", "err")
        assert audit_log._cmd_count == 0


class TestUnknown:
    def test_emits_shell_unknown_event(self, audit_log):
        audit_log.unknown("foobarxyz")
        events = _read_events(audit_log.log_path)
        e = events[0]
        assert e["event"] == "shell.unknown"
        assert e["cmd"] == "foobarxyz"
        assert e["cmd_num"] == 1


class TestBackground:
    def test_emits_shell_background_event(self, audit_log):
        audit_log.background("sleep 10 &", 42)
        events = _read_events(audit_log.log_path)
        e = events[0]
        assert e["event"] == "shell.background"
        assert e["line"] == "sleep 10 &"
        assert e["bg_id"] == 42


class TestStartupShutdown:
    def test_startup_event(self, audit_log):
        audit_log.startup()
        events = _read_events(audit_log.log_path)
        e = events[0]
        assert e["event"] == "shell.startup"
        assert e["pid"] == os.getpid()

    def test_shutdown_event(self, audit_log):
        audit_log.command("a", "a", "", 0)
        audit_log.command("b", "b", "", 0)
        audit_log.shutdown()
        events = _read_events(audit_log.log_path)
        e = events[-1]
        assert e["event"] == "shell.shutdown"
        assert e["total_commands"] == 2


class TestTimestamps:
    def test_events_have_iso_timestamp(self, audit_log):
        audit_log.command("x", "x", "", 0)
        events = _read_events(audit_log.log_path)
        assert "ts" in events[0]
        assert "T" in events[0]["ts"]

    def test_events_have_session_id(self, audit_log):
        audit_log.command("x", "x", "", 0)
        events = _read_events(audit_log.log_path)
        assert events[0]["session"] == audit_log._session_id


class TestNoHandler:
    def test_emit_does_not_crash_without_handler(self):
        logger = ShellAuditLogger.__new__(ShellAuditLogger)
        logger._handler = None
        logger._session_id = "test"
        logger._cmd_count = 0
        logger._log_path = Path("/dev/null")
        # Should not raise
        logger._emit("test.event", key="value")
        logger.command("x", "x", "", 0)
        logger.eval("1", "1", 0)
        logger.error("x", "err")
        logger.unknown("x")
        logger.background("x", 1)
        logger.startup()
        logger.shutdown()


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_shell_audit_logger()
        b = get_shell_audit_logger()
        assert a is b
