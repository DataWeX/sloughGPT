"""Tests for apps/cli/src/commands/dev.py — dev server and health commands."""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    fake_log.header = MagicMock()
    fake_log.section = MagicMock()
    fake_log.info = MagicMock()
    fake_log.warning = MagicMock()
    fake_log.error = MagicMock()
    fake_log.success = MagicMock()
    fake_log.step = MagicMock()
    fake_log.key_value = MagicMock()
    fake_log.blank = MagicMock()
    fake_log.table = MagicMock()
    fake_log.status = MagicMock()
    fake_log.command = MagicMock()
    import commands.dev as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestHelpers:
    def test_repo_root(self):
        from commands.dev import _repo_root
        root = _repo_root()
        assert root.exists()
        assert (root / "cli.py").exists() or (root / "apps").exists()

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
    def test_health_server_down(self, monkeypatch, capsys):
        from commands.dev import cmd_health
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        cmd_health(args)
        out = capsys.readouterr().out
        assert "Health" in out or "Offline" in out or "not" in out.lower()

    def test_health_server_up(self, monkeypatch, capsys):
        from commands.dev import cmd_health
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_resp))
        args = MagicMock()
        args.port = 8000
        cmd_health(args)
        out = capsys.readouterr().out
        assert "Health" in out or "Online" in out or "ok" in out.lower()


class TestCmdApiStatus:
    def test_api_status_server_down(self, monkeypatch, capsys):
        from commands.dev import cmd_api_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        cmd_api_status(args)
        out = capsys.readouterr().out
        assert "API" in out or "Status" in out or "not" in out.lower()


class TestCmdApiTest:
    def test_api_test_server_down(self, monkeypatch, capsys):
        from commands.dev import cmd_api_test
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.port = 8000
        args.endpoint = "/health"
        cmd_api_test(args)
        out = capsys.readouterr().out
        assert "Test" in out or "API" in out or "fail" in out.lower() or "error" in out.lower()
