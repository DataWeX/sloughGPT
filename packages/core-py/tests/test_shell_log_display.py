"""Tests for domains.shell.log_display — LineModeLogDisplay."""

import os
import time

import pytest
from domains.shell.log_display import LineModeLogDisplay, _LEVEL_LABELS, _LEVEL_COLORS
from domains.shell.log_buffer import LogBuffer, LogEntry


def _make_entry(level="INFO", message="test msg", source="slo.test"):
    return LogEntry(timestamp=time.time(), level=level, source=source, message=message)


# =============================================================================
# Badge
# =============================================================================

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

    def test_critical_counts_as_error(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("CRITICAL"))
        d.poll()
        assert d.unread_errors == 1
        assert "\u2715" in d.badge()

    def test_multiple_errors_and_warnings(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        for _ in range(3):
            buf._entries.append(_make_entry("WARNING"))
        for _ in range(2):
            buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert d.unread_warnings == 3
        assert d.unread_errors == 2
        # Error badge takes priority
        assert "\u2715" in d.badge()

    def test_debug_does_not_affect_badge(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("DEBUG"))
        d.poll()
        assert d.badge() == ""

    def test_badge_contains_unicode_warning(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        assert "\u26a0" in d.badge()

    def test_badge_contains_unicode_error(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert "\u2715" in d.badge()


# =============================================================================
# Poll
# =============================================================================

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

    def test_poll_no_new_entries(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        d.poll()
        assert d.unread_warnings == 0
        assert d.unread_errors == 0
        # Poll again without adding entries
        d.poll()
        assert d.unread_warnings == 0

    def test_poll_after_clear(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        assert d.unread_warnings == 1
        d.clear_counts()
        # Add more after clear
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert d.unread_errors == 1
        assert d.unread_warnings == 0

    def test_poll_accumulates_multiple_calls(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        for _ in range(5):
            buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert d.unread_errors == 5
        d.poll()  # no new entries
        assert d.unread_errors == 5

    def test_poll_interleaved_levels(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("INFO"))
        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("ERROR"))
        buf._entries.append(_make_entry("INFO"))
        buf._entries.append(_make_entry("CRITICAL"))
        d.poll()
        assert d.unread_warnings == 1
        assert d.unread_errors == 2  # ERROR + CRITICAL

    def test_poll_initial_index(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        assert d._last_index == 0

    def test_poll_initializes_from_existing_entries(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("WARNING"))
        buf._entries.append(_make_entry("ERROR"))
        d = LineModeLogDisplay(buf)
        # Initial index already accounts for existing entries
        assert d._last_index == 2
        # Polling finds no new entries because _last_index already matched len
        d.poll()
        assert d.unread_warnings == 0
        assert d.unread_errors == 0


# =============================================================================
# ClearCounts
# =============================================================================

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

    def test_clear_counts_when_zero(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        d.clear_counts()
        assert d.unread_warnings == 0
        assert d.unread_errors == 0

    def test_clear_and_poll(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        d.clear_counts()
        buf._entries.append(_make_entry("ERROR"))
        d.poll()
        assert d.unread_warnings == 0
        assert d.unread_errors == 1


# =============================================================================
# RenderRecent
# =============================================================================

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

    def test_render_recent_by_source(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "kernel msg", source="slo.kernel"))
        buf._entries.append(_make_entry("INFO", "shell msg", source="slo.shell"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(source="kernel")
        assert "kernel msg" in result
        assert "shell msg" not in result

    def test_render_recent_default_n(self):
        buf = LogBuffer(max_size=100)
        for i in range(25):
            buf._entries.append(_make_entry("INFO", f"msg{i}"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent()
        # Default n=20, should show last 20
        assert "msg24" in result
        assert "msg0" not in result

    def test_render_recent_returns_string(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "test"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent()
        assert isinstance(result, str)

    def test_render_recent_level_and_source_combined(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("WARNING", "k msg", source="slo.kernel"))
        buf._entries.append(_make_entry("WARNING", "s msg", source="slo.shell"))
        buf._entries.append(_make_entry("ERROR", "k err", source="slo.kernel"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(level="WARNING", source="kernel")
        assert "k msg" in result
        assert "s msg" not in result
        assert "k err" not in result

    def test_render_recent_no_match(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "info msg"))
        d = LineModeLogDisplay(buf)
        result = d.render_recent(level="ERROR")
        assert "No log entries" in result


# =============================================================================
# RenderLast
# =============================================================================

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

    def test_render_last_returns_string(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("INFO", "test"))
        d = LineModeLogDisplay(buf)
        result = d.render_last()
        assert isinstance(result, str)

    def test_render_last_single_entry(self):
        buf = LogBuffer(max_size=100)
        buf._entries.append(_make_entry("CRITICAL", "panic!"))
        d = LineModeLogDisplay(buf)
        result = d.render_last()
        assert "panic!" in result


# =============================================================================
# FormatEntry
# =============================================================================

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

    def test_format_entry_debug(self):
        e = _make_entry("DEBUG", "trace")
        result = LineModeLogDisplay._format_entry(e)
        assert "DBG" in result
        assert "trace" in result

    def test_format_entry_info(self):
        e = _make_entry("INFO", "update")
        result = LineModeLogDisplay._format_entry(e)
        assert "INF" in result
        assert "update" in result

    def test_format_entry_error(self):
        e = _make_entry("ERROR", "crash")
        result = LineModeLogDisplay._format_entry(e)
        assert "ERR" in result
        assert "crash" in result

    def test_format_entry_critical(self):
        e = _make_entry("CRITICAL", "fatal")
        result = LineModeLogDisplay._format_entry(e)
        assert "CRT" in result
        assert "fatal" in result

    def test_format_entry_source_strips_prefix(self):
        e = _make_entry("INFO", "msg", source="slo.kernel.driver")
        result = LineModeLogDisplay._format_entry(e)
        assert "driver" in result

    def test_format_entry_empty_source(self):
        e = _make_entry("INFO", "msg", source="")
        result = LineModeLogDisplay._format_entry(e)
        assert "msg" in result

    def test_format_entry_exact_120_chars(self):
        e = _make_entry("INFO", "a" * 120)
        result = LineModeLogDisplay._format_entry(e)
        assert "a" * 120 in result

    def test_format_entry_121_chars_truncated(self):
        e = _make_entry("INFO", "a" * 121)
        result = LineModeLogDisplay._format_entry(e)
        assert len(result) < 200  # truncated at 120

    def test_format_entry_unknown_level_uses_first_3_chars(self):
        e = LogEntry(timestamp=time.time(), level="CUSTOM", source="s", message="m")
        result = LineModeLogDisplay._format_entry(e)
        assert "CUS" in result


# =============================================================================
# Color constants
# =============================================================================

class TestColorConstants:
    def test_level_colors_has_all_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in _LEVEL_COLORS

    def test_level_colors_are_strings(self):
        for level, color in _LEVEL_COLORS.items():
            assert isinstance(color, str)

    def test_level_colors_debug_is_dim(self):
        assert _LEVEL_COLORS["DEBUG"] != ""

    def test_level_colors_error_is_red(self):
        assert "\033[31m" in _LEVEL_COLORS["ERROR"]


# =============================================================================
# Integration
# =============================================================================

class TestIntegration:
    def test_full_workflow(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)

        # Start with some entries
        buf._entries.append(_make_entry("INFO", "startup"))
        buf._entries.append(_make_entry("WARNING", "deprecated api"))
        d.poll()

        # Check badge
        assert d.unread_warnings == 1
        assert "2" not in d.badge()  # only 1 warning

        # Render recent
        recent = d.render_recent()
        assert "startup" in recent
        assert "deprecated api" in recent

        # Clear and add more
        d.clear_counts()
        buf._entries.append(_make_entry("ERROR", "connection failed"))
        d.poll()
        assert d.unread_errors == 1
        assert "\u2715" in d.badge()

    def test_poll_badge_clear_cycle(self):
        buf = LogBuffer(max_size=100)
        d = LineModeLogDisplay(buf)

        # Poll with no entries
        d.poll()
        assert d.badge() == ""

        # Add warnings
        buf._entries.append(_make_entry("WARNING"))
        d.poll()
        assert d.unread_warnings == 1

        # Clear counts
        d.clear_counts()
        assert d.badge() == ""

        # Poll again — no new entries
        d.poll()
        assert d.badge() == ""
