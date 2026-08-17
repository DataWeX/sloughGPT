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
    def test_repo_root(self):
        from commands.dev import _repo_root
        root = _repo_root()
        assert root.exists()
        assert (root / "apps").exists()

    def test_check_port_closed(self):
        from commands.dev import _check_port
        result = _check_port(1)
        assert isinstance(result, bool)

    def test_check_api_ready_closed(self):
        from commands.dev import _check_api_ready
        result = _check_api_ready(1)
        assert result is False

    def test_get_startup_progress_none(self):
        from commands.dev import _get_startup_progress
        result = _get_startup_progress(1)
        assert result is None


class TestCmdHealth:
    def test_health_server_down(self, monkeypatch):
        from commands.dev import cmd_health
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        cmd_health(args)

    def test_health_server_up(self, monkeypatch):
        from commands.dev import cmd_health
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.port = 8000
        cmd_health(args)


class TestCmdApiStatus:
    def test_api_status_server_down(self, monkeypatch):
        from commands.dev import cmd_api_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        cmd_api_status(args)

    def test_api_status_server_up(self, monkeypatch):
        from commands.dev import cmd_api_status
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True, "version": "1.0"}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.port = 8000
        cmd_api_status(args)


class TestCmdApiTest:
    def test_api_test_server_down(self, monkeypatch):
        from commands.dev import cmd_api_test
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        args.endpoint = "/health"
        cmd_api_test(args)
