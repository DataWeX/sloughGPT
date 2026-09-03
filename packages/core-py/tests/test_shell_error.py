"""Tests for shell error formatting — user-friendly error messages."""
from __future__ import annotations

import pytest

from domains.shell.error import format_error


class TestFormatError:
    def test_generic_error(self):
        e = ValueError("bad value")
        result = format_error(e, color=False)
        assert "ValueError" in result
        assert "bad value" in result

    def test_error_with_cmd(self):
        e = RuntimeError("oops")
        result = format_error(e, cmd="train", color=False)
        assert "[train]" in result
        assert "oops" in result

    def test_permission_error(self):
        e = PermissionError("denied")
        result = format_error(e, color=False)
        assert "Permission denied" in result

    def test_file_not_found(self):
        e = FileNotFoundError("no file")
        result = format_error(e, color=False)
        assert "File not found" in result

    def test_connection_error(self):
        e = ConnectionError("refused")
        result = format_error(e, color=False)
        assert "Connection failed" in result
        assert "api start" in result

    def test_timeout_error(self):
        e = TimeoutError("timed out")
        result = format_error(e, color=False)
        assert "timed out" in result

    def test_color_flag(self):
        e = ValueError("test")
        result_colored = format_error(e, color=True)
        result_plain = format_error(e, color=False)
        # Both should contain the error message
        assert "test" in result_colored
        assert "test" in result_plain

    def test_fallback_format(self):
        e = KeyError("missing_key")
        result = format_error(e, cmd="load", color=False)
        assert "KeyError" in result
        assert "missing_key" in result
