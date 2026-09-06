"""Tests for shell.error — user-friendly error formatting."""

from __future__ import annotations

import pytest

from domains.shell.error import format_error


# ── format_error ──────────────────────────────────────────────────────────


class TestFormatError:

    def test_connection_error_with_hint(self):
        e = ConnectionError("refused")
        result = format_error(e, color=False)
        assert "Connection failed" in result
        assert "Is the API server running?" in result

    def test_requests_connection_error(self):
        import requests
        e = requests.ConnectionError("refused")
        result = format_error(e, color=False)
        assert "Connection failed" in result
        assert "Is the API server running?" in result

    def test_timeout_error(self):
        e = TimeoutError("timed out")
        result = format_error(e, color=False)
        assert "Request timed out" in result

    def test_requests_timeout(self):
        import requests
        e = requests.Timeout("timed out")
        result = format_error(e, color=False)
        assert "Request timed out" in result

    def test_permission_error(self):
        e = PermissionError("denied")
        result = format_error(e, color=False)
        assert "Permission denied" in result

    def test_file_not_found_error(self):
        e = FileNotFoundError("missing.txt")
        result = format_error(e, color=False)
        assert "File not found" in result

    def test_generic_error_without_cmd(self):
        e = ValueError("bad value")
        result = format_error(e, color=False)
        assert "ValueError: bad value" in result
        assert result.startswith("  ")

    def test_generic_error_with_cmd(self):
        e = RuntimeError("boom")
        result = format_error(e, cmd="deploy", color=False)
        assert "[deploy]" in result
        assert "RuntimeError: boom" in result

    def test_empty_cmd_no_prefix(self):
        e = KeyError("missing")
        result = format_error(e, cmd="", color=False)
        assert result.startswith("  ")
        assert "[" not in result.split(":")[0]

    def test_color_flag_does_not_affect_output(self):
        e = ValueError("test")
        no_color = format_error(e, color=False)
        with_color = format_error(e, color=True)
        # Neither should contain ANSI codes for ValueError fallback
        assert no_color == with_color
