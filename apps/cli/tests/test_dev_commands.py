"""Tests for apps/cli/src/commands/dev.py — dev server and health commands."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.dev as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestHelpers:
    def test_repo_root_returns_path(self):
        from commands.dev import _repo_root
        root = _repo_root()
        assert root.exists()
        assert (root / "apps").is_dir()

    def test_check_port_closed(self):
        from commands.dev import _check_port
        assert _check_port(1) is False

    def test_check_api_ready_closed(self):
        from commands.dev import _check_api_ready
        assert _check_api_ready(1) is False

    def test_get_startup_progress_none(self):
        from commands.dev import _get_startup_progress
        assert _get_startup_progress(1) is None


class TestCmdHealth:
    def test_server_down_logs_error(self, mock_log, monkeypatch):
        from commands.dev import cmd_health
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        mock_log.header.assert_called_with("API Health Check")
        mock_log.error.assert_called_with("API not reachable")

    def test_server_up_logs_success(self, mock_log, monkeypatch):
        from commands.dev import cmd_health
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        mock_log.success.assert_called()
        assert "Healthy" in mock_log.success.call_args[0][0]

    def test_server_up_displays_keys(self, mock_log, monkeypatch):
        from commands.dev import cmd_health
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model": "gpt2"}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_health(args)
        kv_keys = [c[0][0] for c in mock_log.key_value.call_args_list]
        assert "Endpoint" in kv_keys


class TestCmdApiStatus:
    def test_server_down_logs_not_reachable(self, mock_log, monkeypatch):
        from commands.dev import cmd_api_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_api_status(args)
        mock_log.header.assert_called_with("SloughGPT API Status")
        mock_log.status.assert_called()

    def test_endpoints_checked(self, mock_log, monkeypatch):
        from commands.dev import cmd_api_status
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        cmd_api_status(args)
        assert mock_log.status.call_count >= 4


class TestCmdApiTest:
    def test_server_down_logs_error(self, mock_log, monkeypatch):
        from commands.dev import cmd_api_test
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        monkeypatch.setattr(requests, "post", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.host = "localhost"
        args.port = 8000
        args.endpoint = "/health"
        cmd_api_test(args)
        mock_log.header.assert_called_with("API Endpoint Tests")


class TestStatusBlock:
    """Tests for the StatusBlock in-place update class."""

    def test_update_writes_to_tty(self):
        import io
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__('threading').Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        assert block._is_tty is True

        block.update("  SloughGPT", "  API: starting")

        assert logger.info.call_count == 2
        calls = [c[0][0] for c in logger.info.call_args_list]
        assert "  SloughGPT" in calls
        assert "  API: starting" in calls

    def test_update_clears_previous_on_tty(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__('threading').Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        block.update("  Line 1", "  Line 2")
        assert len(block._lines) == 2

        block.update("  New Line 1")

        # Should have called cursor_up and clear_line to clear previous lines
        assert logger.cursor_up.called
        assert logger.clear_line.called
        assert len(block._lines) == 1

    def test_first_update_no_clear(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._lock = __import__('threading').Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream

        block = StatusBlock(logger)
        block.update("  Only line")

        # First update should not call cursor_up or clear_line
        assert not logger.cursor_up.called
        assert not logger.clear_line.called
        assert len(block._lines) == 1

    def test_non_tty_uses_info(self):
        import threading
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._stream = MagicMock()
        logger._stream.isatty.return_value = False
        logger._lock = threading.Lock()
        logger._colors = False

        block = StatusBlock(logger)
        assert block._is_tty is False

        block.update("  Line 1", "  Line 2")
        assert logger.info.call_count == 2

    def test_non_tty_prints_only_once(self):
        from commands.dev import StatusBlock

        logger = MagicMock()
        logger._stream = MagicMock()
        logger._stream.isatty.return_value = False
        logger._lock = __import__('threading').Lock()
        logger._colors = False

        block = StatusBlock(logger)
        block.update("  Line 1")
        assert logger.info.call_count == 1

        # Second update should NOT print again
        block.update("  Line 2")
        assert logger.info.call_count == 1

    def test_line_count_tracking(self):
        import threading
        from commands.dev import StatusBlock

        logger = MagicMock()
        stream = MagicMock()
        stream.isatty.return_value = True
        logger._stream = stream
        logger._lock = threading.Lock()
        logger._colors = False
        logger.cursor_up = MagicMock()
        logger.clear_line = MagicMock()

        block = StatusBlock(logger)
        block.update("a", "b", "c")
        assert len(block._lines) == 3
        block.update("x")
        assert len(block._lines) == 1


class TestPortHelpers:
    """Tests for port utility functions."""

    def test_is_port_bound_free(self):
        from commands.dev import _is_port_bound
        assert _is_port_bound(49999) is False

    def test_find_free_port_same_if_free(self):
        from commands.dev import _find_free_port
        assert _find_free_port(49999) == 49999

    def test_find_free_port_skips_bound(self):
        import socket
        from commands.dev import _find_free_port, _is_port_bound

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", 49998))
            sock.listen(1)
            assert _is_port_bound(49998) is True
            result = _find_free_port(49998)
            assert result != 49998
            assert result >= 49998
        finally:
            sock.close()

    def test_check_web_ready_nonexistent(self):
        from commands.dev import _check_web_ready
        assert _check_web_ready(49997) is False
