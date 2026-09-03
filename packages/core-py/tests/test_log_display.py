"""Tests for domains.shell.log_display — LineModeLogDisplay."""

from __future__ import annotations

import time
import pytest

from domains.shell.log_buffer import LogEntry, LogBuffer
from domains.shell.log_display import LineModeLogDisplay, _LEVEL_LABELS, _LEVEL_COLORS


def _make_entry(level="INFO", source="slo.test", message="test message"):
    return LogEntry(
        timestamp=time.time(),
        level=level,
        source=source,
        message=message,
    )


# ── LineModeLogDisplay ───────────────────────────────────────────────────────

class TestLineModeLogDisplay:
    def test_empty_buffer(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        assert display.badge() == ""
        assert display.unread_warnings == 0
        assert display.unread_errors == 0

    def test_poll_info_no_badge(self):
        buf = LogBuffer()
        buf.append(_make_entry(level="INFO"))
        display = LineModeLogDisplay(buf)
        display.poll()
        assert display.badge() == ""
        assert display.unread_warnings == 0
        assert display.unread_errors == 0

    def test_poll_warning_badge(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        buf.append(_make_entry(level="WARNING"))
        display.poll()
        assert display.unread_warnings == 1
        assert "⚠" in display.badge() or "warning" in display.badge().lower()

    def test_poll_error_badge(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        buf.append(_make_entry(level="ERROR"))
        display.poll()
        assert display.unread_errors == 1
        assert "✕" in display.badge() or "error" in display.badge().lower()

    def test_poll_critical_badge(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        buf.append(_make_entry(level="CRITICAL"))
        display.poll()
        assert display.unread_errors == 1

    def test_error_takes_precedence(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        buf.append(_make_entry(level="WARNING"))
        buf.append(_make_entry(level="ERROR"))
        display.poll()
        badge = display.badge()
        assert "✕" in badge or "error" in badge.lower()

    def test_clear_counts(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        buf.append(_make_entry(level="WARNING"))
        display.poll()
        assert display.unread_warnings == 1
        display.clear_counts()
        assert display.unread_warnings == 0
        assert display.unread_errors == 0
        assert display.badge() == ""

    def test_poll_only_new_entries(self):
        buf = LogBuffer()
        buf.append(_make_entry(level="INFO"))
        display = LineModeLogDisplay(buf)
        display.poll()
        assert display.unread_warnings == 0
        # Add more entries
        buf.append(_make_entry(level="WARNING"))
        display.poll()
        assert display.unread_warnings == 1

    def test_render_recent_empty(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        result = display.render_recent()
        assert "No log entries" in result

    def test_render_recent_with_entries(self):
        buf = LogBuffer()
        buf.append(_make_entry(level="INFO", message="hello"))
        buf.append(_make_entry(level="WARNING", message="world"))
        display = LineModeLogDisplay(buf)
        result = display.render_recent()
        assert "hello" in result
        assert "world" in result

    def test_render_recent_filter_level(self):
        buf = LogBuffer()
        buf.append(_make_entry(level="INFO"))
        buf.append(_make_entry(level="WARNING"))
        display = LineModeLogDisplay(buf)
        result = display.render_recent(level="WARNING")
        assert "WRN" in result

    def test_render_recent_filter_source(self):
        buf = LogBuffer()
        buf.append(_make_entry(source="slo.kernel"))
        buf.append(_make_entry(source="slo.api"))
        display = LineModeLogDisplay(buf)
        result = display.render_recent(source="kernel")
        assert "kernel" in result

    def test_render_last_empty(self):
        buf = LogBuffer()
        display = LineModeLogDisplay(buf)
        result = display.render_last()
        assert "No log entries" in result

    def test_render_last_with_entry(self):
        buf = LogBuffer()
        buf.append(_make_entry(level="ERROR", message="fail"))
        display = LineModeLogDisplay(buf)
        result = display.render_last()
        assert "fail" in result
        assert "ERR" in result

    def test_format_entry(self):
        entry = _make_entry(level="INFO", source="slo.test.mod", message="hello")
        result = LineModeLogDisplay._format_entry(entry)
        assert "INF" in result
        assert "hello" in result
        assert "mod" in result  # last part of source


# ── Level constants ──────────────────────────────────────────────────────────

class TestLevelConstants:
    def test_labels_complete(self):
        assert _LEVEL_LABELS["DEBUG"] == "DBG"
        assert _LEVEL_LABELS["INFO"] == "INF"
        assert _LEVEL_LABELS["WARNING"] == "WRN"
        assert _LEVEL_LABELS["ERROR"] == "ERR"
        assert _LEVEL_LABELS["CRITICAL"] == "CRT"

    def test_colors_complete(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            assert level in _LEVEL_COLORS
