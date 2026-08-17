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
