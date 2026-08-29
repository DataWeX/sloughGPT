"""Tests for domains.shell.log_display — LineModeLogDisplay."""

import time

import pytest
from domains.shell.log_display import LineModeLogDisplay, _LEVEL_LABELS, _LEVEL_COLORS
from domains.shell.log_buffer import LogBuffer, LogEntry


def _make_entry(level="INFO", message="test msg", source="slo.test"):
    return LogEntry(timestamp=time.time(), level=level, source=source, message=message)


class TestLineModeLogDisplayBadge:
    def test_empty_when_no_logs(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        assert d.badge() == ""

    def test_empty_when_only_info(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO"))
        d = LineModeLogDisplay(buf)
        d.poll()
        assert d.badge() == ""

    def test_shows_warning_count(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        badge = d.badge()
        assert "2" in badge

    def test_shows_error_count(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        badge = d.badge()
        assert "1" in badge

    def test_error_overrides_warning(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        badge = d.badge()
        assert "\u2715" in badge or "1" in badge


class TestLineModeLogDisplayPoll:
    def test_poll_increments_counts(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        assert d.unread_warnings == 0
        assert d.unread_errors == 0

        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert d.unread_warnings == 1
        assert d.unread_errors == 1

    def test_poll_only_new_entries(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        assert d.unread_warnings == 1
        # Add more
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        assert d.unread_warnings == 2  # not reset, accumulates


class TestLineModeLogDisplayClearCounts:
    def test_clear_counts(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        d.clear_counts()
        assert d.unread_warnings == 0
        assert d.unread_errors == 0
        assert d.badge() == ""


class TestLineModeLogDisplayRenderRecent:
    def test_render_recent_empty(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        result = d.render_recent()
        assert "No log entries" in result

    def test_render_recent(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "hello"))
        buf._entries.append(_make_entry("ERROR", "fail"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(n=10)
        assert "hello" in result
        assert "fail" in result

    def test_render_recent_limit(self):
        buf = LogBuffer(max_size=100)
        for i in range(5):
            buf._entries.append(_make_entry("INFO", f"msg{i}"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(n=2)
        assert "msg3" in result
        assert "msg4" in result
        assert "msg0" not in result

    def test_render_recent_by_level(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "info msg"))
        buf._entries.append(_make_entry("ERROR", "err msg"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(level="ERROR")
        assert "err msg" in result
        assert "info msg" not in result


class TestLineModeLogDisplayRenderLast:
    def test_render_last_empty(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        result = d.render_last()
        assert "No log entries" in result

    def test_render_last(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "first"))
        buf._entries.append(_make_entry("ERROR", "latest"))
        d = LineModeLogDisplay(buf)
        result = d.render_last()
        assert "latest" in result
        assert "first" not in result


class TestFormatEntry:
    def test_format_entry(self):
        e = _make_entry("WARNING", "something broke", "slo.model_server")
        result = LineModeLogDisplay._format_entry(e)
        assert "something broke" in result
        assert "WRN" in result

    def test_format_truncates_long_message(self):
        e = _make_entry("INFO", "x" * 200)
        result = LineModeLogDisplay._format_entry(e)
        assert len(result) < 200

    def test_level_labels(self):
        assert _LEVEL_LABELS["DEBUG"] == "DBG"
        assert _LEVEL_LABELS["INFO"] == "INF"
        assert _LEVEL_LABELS["WARNING"] == "WRN"
        assert _LEVEL_LABELS["ERROR"] == "ERR"
        assert _LEVEL_LABELS["CRITICAL"] == "CRT"
