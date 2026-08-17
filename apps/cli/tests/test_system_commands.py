"""Tests for apps/cli/src/commands/system.py — system info and config commands."""
import sys
import os
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
    def test_runs_without_error(self):
        from commands.system import cmd_system
        args = MagicMock()
        cmd_system(args)


class TestCmdStatus:
    def test_server_down(self, monkeypatch):
        from commands.system import cmd_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError))
        args = MagicMock()
        args.watch = False
        args.interval = 3
        cmd_status(args)


class TestCmdOptimize:
    def test_runs_without_error(self):
        from commands.system import cmd_optimize
        args = MagicMock()
        args.optimize = False
        cmd_optimize(args)


class TestCmdConfigCheck:
    def test_runs_without_error(self):
        from commands.system import cmd_config_check
        args = MagicMock()
        cmd_config_check(args)


class TestCmdConfigValidate:
    def test_missing_env_file(self):
        from commands.system import cmd_config_validate
        args = MagicMock()
        args.env = "/nonexistent/.env"
        cmd_config_validate(args)

    def test_valid_env_file(self, tmp_path):
        from commands.system import cmd_config_validate
        env_file = tmp_path / ".env"
        env_file.write_text("SLO_API_KEY=abcdefghijklmnopqrstuvwxyz123456\n")
        args = MagicMock()
        args.env = str(env_file)
        cmd_config_validate(args)


class TestCmdConfigGenerate:
    def test_generates_api_key(self):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "api-key"
        cmd_config_generate(args)

    def test_generates_jwt_secret(self):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "jwt-secret"
        cmd_config_generate(args)

    def test_generates_all(self):
        from commands.system import cmd_config_generate
        args = MagicMock()
        args.type = "all"
        cmd_config_generate(args)


class TestCmdStats:
    def test_runs_without_error(self, tmp_path, monkeypatch):
        from commands.system import cmd_stats
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "datasets").mkdir()
        (tmp_path / "checkpoints").mkdir()
        args = MagicMock()
        cmd_stats(args)
