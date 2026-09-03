"""Tests for domains.shell.cmds.dashboard — helper functions."""

from __future__ import annotations

from domains.shell.cmds.dashboard import (
    _format_uptime,
    _format_ts,
    _sparkline,
    _status_icon,
    _progress_bar,
)


# ── _format_uptime ────────────────────────────────────────────────────────────

class TestFormatUptime:
    def test_seconds(self):
        assert _format_uptime(30) == "30s"

    def test_minutes(self):
        assert _format_uptime(125) == "2m 05s"

    def test_hours(self):
        assert _format_uptime(3661) == "1h 01m"

    def test_zero(self):
        assert _format_uptime(0) == "0s"

    def test_exact_minute(self):
        assert _format_uptime(60) == "1m 00s"

    def test_exact_hour(self):
        assert _format_uptime(3600) == "1h 00m"


# ── _format_ts ────────────────────────────────────────────────────────────────

class TestFormatTs:
    def test_returns_string(self):
        result = _format_ts(0.0)
        assert isinstance(result, str)
        assert ":" in result

    def test_known_timestamp(self):
        # 2024-01-01 00:00:00 UTC
        result = _format_ts(1704067200.0)
        assert result == "00:00:00"


# ── _sparkline ────────────────────────────────────────────────────────────────

class TestSparkline:
    def test_empty(self):
        assert _sparkline([]) == ""

    def test_single_value(self):
        result = _sparkline([5.0])
        assert len(result) == 1

    def test_constant_values(self):
        result = _sparkline([5.0, 5.0, 5.0])
        assert len(result) == 3

    def test_increasing(self):
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5

    def test_sampling(self):
        values = list(range(20))
        result = _sparkline(values, width=10)
        assert len(result) == 10


# ── _status_icon ──────────────────────────────────────────────────────────────

class TestStatusIcon:
    def test_running(self):
        icon = _status_icon("running")
        assert "▶" in icon

    def test_complete(self):
        icon = _status_icon("complete")
        assert "✓" in icon

    def test_error(self):
        icon = _status_icon("error")
        assert "✗" in icon

    def test_unknown(self):
        icon = _status_icon("unknown")
        assert icon == "?"


# ── _progress_bar ─────────────────────────────────────────────────────────────

class TestProgressBar:
    def test_zero(self):
        bar = _progress_bar(0)
        assert "░" in bar
        assert "█" not in bar.replace("\033[32m", "").replace("\033[90m", "").replace("\033[0m", "")

    def test_full(self):
        bar = _progress_bar(100)
        assert "█" in bar

    def test_half(self):
        bar = _progress_bar(50)
        assert "█" in bar
        assert "░" in bar
