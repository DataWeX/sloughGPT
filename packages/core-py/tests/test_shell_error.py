"""Tests for domains.shell.error — format_error."""

from __future__ import annotations

import pytest
from domains.shell.error import format_error


class TestFormatError:
    def test_connection_error(self):
        e = ConnectionError("refused")
        result = format_error(e)
        assert "Connection failed" in result
        assert "Is the API server running" in result

    def test_timeout_error(self):
        e = TimeoutError("timed out")
        result = format_error(e)
        assert "timed out" in result.lower()

    def test_permission_error(self):
        e = PermissionError("/secret")
        result = format_error(e)
        assert "Permission denied" in result

    def test_file_not_found(self):
        e = FileNotFoundError("/missing")
        result = format_error(e)
        assert "File not found" in result

    def test_generic_error(self):
        e = ValueError("bad value")
        result = format_error(e)
        assert "ValueError" in result
        assert "bad value" in result

    def test_with_command_prefix(self):
        e = ValueError("bad")
        result = format_error(e, cmd="test_cmd")
        assert "[test_cmd]" in result

    def test_without_command_prefix(self):
        e = ValueError("bad")
        result = format_error(e)
        assert "  ValueError: bad" == result

    def test_requests_connection_error(self):
        try:
            import requests
            e = requests.ConnectionError("refused")
            result = format_error(e)
            assert "Connection failed" in result
        except ImportError:
            pytest.skip("requests not installed")

    def test_requests_timeout_error(self):
        try:
            import requests
            e = requests.Timeout("timed out")
            result = format_error(e)
            assert "timed out" in result.lower()
        except ImportError:
            pytest.skip("requests not installed")
