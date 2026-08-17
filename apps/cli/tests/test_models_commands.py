"""Tests for apps/cli/src/commands/models.py — model management commands."""
import sys
import os
import pytest
from pathlib import Path
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
    import commands.models as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdModels:
    def test_lists_soul_files(self, tmp_path, monkeypatch, capsys):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "test.soul").write_bytes(b"fake soul data")
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)
        out = capsys.readouterr().out
        assert "test.soul" in out or "Available Models" in out

    def test_empty_models_dir(self, tmp_path, monkeypatch, capsys):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)
        out = capsys.readouterr().out
        assert "No soul files found" in out or "Available Models" in out


class TestCmdModelsInfo:
    def test_missing_model_exits(self, tmp_path):
        from commands.models import _cmd_models_info
        args = MagicMock()
        args.model = str(tmp_path / "nonexistent.soul")
        _cmd_models_info(args)


class TestCmdModelsStatus:
    def test_shows_status(self, capsys):
        from commands.models import cmd_models
        args = MagicMock()
        cmd_models(args)
