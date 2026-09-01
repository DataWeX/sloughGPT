from unittest.mock import patch, MagicMock
import pytest

from domains.shell.error import format_error


class FakeRequests:
    ConnectionError = type("ConnectionError", (ConnectionError,), {})
    Timeout = type("Timeout", (TimeoutError,), {})


@pytest.fixture(autouse=True)
def _patch_requests():
    with patch.dict("sys.modules", {"requests": FakeRequests}):
        yield


class TestFormatError:
    def test_connection_error(self):
        e = ConnectionError("refused")
        result = format_error(e, color=False)
        assert "Connection failed" in result
        assert "Is the API server running" in result

    def test_timeout_error(self):
        e = TimeoutError("timed out")
        result = format_error(e)
        assert "timed out" in result

    def test_permission_error(self):
        e = PermissionError("/etc/shadow")
        result = format_error(e)
        assert "Permission denied" in result
        assert "/etc/shadow" in result

    def test_file_not_found(self):
        e = FileNotFoundError("/tmp/nope.txt")
        result = format_error(e)
        assert "File not found" in result
        assert "/tmp/nope.txt" in result

    def test_generic_error(self):
        e = ValueError("bad value")
        result = format_error(e)
        assert "ValueError" in result
        assert "bad value" in result

    def test_generic_error_with_cmd(self):
        e = RuntimeError("oops")
        result = format_error(e, cmd="train")
        assert "[train]" in result
        assert "RuntimeError" in result

    def test_empty_cmd_no_prefix(self):
        e = KeyError("missing")
        result = format_error(e, cmd="")
        assert result.startswith("  ")
        assert "[train]" not in result

    def test_color_param_accepted(self):
        e = TypeError("nope")
        result = format_error(e, color=True)
        assert "TypeError" in result
        result2 = format_error(e, color=False)
        assert "TypeError" in result2

    def test_requests_connection_error(self):
        e = FakeRequests.ConnectionError("net fail")
        result = format_error(e)
        assert "Connection failed" in result

    def test_requests_timeout(self):
        e = FakeRequests.Timeout("slow")
        result = format_error(e)
        assert "timed out" in result
