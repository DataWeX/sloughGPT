"""Tests for apps/cli/src/commands/system.py — system info and config commands."""
import sys
import os
import secrets
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.system as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdSystem:
    def test_header_called(self, mock_log):
        from commands.system import cmd_system
        args = MagicMock()
        cmd_system(args)
        mock_log.header.assert_called_with("System Information")

    def test_platform_section(self, mock_log):
        from commands.system import cmd_system
        args = MagicMock()
        cmd_system(args)
        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "Platform" in sections

    def test_platform_key_values(self, mock_log):
        from commands.system import cmd_system
        args = MagicMock()
        cmd_system(args)
        kv_keys = [c[0][0] for c in mock_log.key_value.call_args_list]
        assert "Platform" in kv_keys
        assert "Python" in kv_keys
        assert "Machine" in kv_keys


class TestCmdStatus:
    def test_server_down_shows_error(self, mock_log, monkeypatch):
        from commands.system import cmd_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.watch = False
        args.interval = 3
        cmd_status(args)
        mock_log.header.assert_called_with("SloughGPT Status")
        mock_log.status.assert_called()

    def test_shows_info_hint(self, mock_log, monkeypatch):
        from commands.system import cmd_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.watch = False
        args.interval = 3
        cmd_status(args)
        mock_log.info.assert_called()
        assert "watch" in mock_log.info.call_args[0][0].lower()


class TestCmdOptimize:
    def test_header_called(self, mock_log):
        from commands.system import cmd_optimize
        args = MagicMock()
        args.optimize = False
        cmd_optimize(args)
        mock_log.header.assert_called()

    def test_accelerator_info_shown(self, mock_log):
        from commands.system import cmd_optimize
        args = MagicMock()
        args.optimize = False
        cmd_optimize(args)
        kv_keys = [c[0][0] for c in mock_log.key_value.call_args_list]
        assert any("ccelerator" in k or "Backend" in k or "Device" in k for k in kv_keys)


class TestCmdConfigCheck:
    def test_runs_doctor(self, mock_log):
        from commands.system import cmd_config_check
        args = MagicMock()
        cmd_config_check(args)
        mock_log.header.assert_called()


class TestCmdConfigValidate:
    def test_missing_env_file(self, mock_log):
        from commands.system import cmd_config_validate
        args = MagicMock()
        args.env = "/nonexistent/.env"
        cmd_config_validate(args)
        mock_log.warning.assert_called()
        assert "not found" in mock_log.warning.call_args[0][0].lower()

    def test_valid_env_file(self, mock_log, tmp_path):
        from commands.system import cmd_config_validate
        env_file = tmp_path / ".env"
        env_file.write_text("SLO_API_KEY=abcdefghijklmnopqrstuvwxyz123456\n")
        args = MagicMock()
        args.env = str(env_file)
        cmd_config_validate(args)
        mock_log.header.assert_called()


class TestCmdConfigGenerate:
    def test_generates_api_key(self, mock_log):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "api-key"
        cmd_config_generate(args)
        commands = [c[0][0] for c in mock_log.command.call_args_list]
        assert any("SLO_API_KEY=" in c for c in commands)
        key_line = [c for c in commands if "SLO_API_KEY=" in c][0]
        key = key_line.split("=", 1)[1]
        assert len(key) >= 32  # secrets.token_urlsafe(32) produces 43 chars

    def test_generates_jwt_secret(self, mock_log):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "jwt-secret"
        cmd_config_generate(args)
        commands = [c[0][0] for c in mock_log.command.call_args_list]
        assert any("SLO_JWT_SECRET=" in c for c in commands)
        key_line = [c for c in commands if "SLO_JWT_SECRET=" in c][0]
        key = key_line.split("=", 1)[1]
        assert len(key) >= 64  # secrets.token_urlsafe(64) produces 86 chars

    def test_generates_all(self, mock_log):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "all"
        cmd_config_generate(args)
        commands = [c[0][0] for c in mock_log.command.call_args_list]
        assert any("SLO_API_KEY=" in c for c in commands)
        assert any("SLO_JWT_SECRET=" in c for c in commands)
        assert any("ENCRYPTION_KEY=" in c for c in commands)


class TestCmdStats:
    def test_header_called(self, mock_log, tmp_path, monkeypatch):
        from commands.system import cmd_stats
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "datasets").mkdir()
        (tmp_path / "checkpoints").mkdir()
        args = MagicMock()
        cmd_stats(args)
        mock_log.header.assert_called_with("SloughGPT Statistics")
