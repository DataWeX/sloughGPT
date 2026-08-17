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
    fake_log.table = MagicMock()
    fake_log.header = MagicMock()
    fake_log.section = MagicMock()
    fake_log.info = MagicMock()
    fake_log.key_value = MagicMock()
    fake_log.blank = MagicMock()
    fake_log.warning = MagicMock()
    fake_log.error = MagicMock()
    fake_log.success = MagicMock()
    fake_log.step = MagicMock()
    import commands.models as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdModels:
    def test_lists_soul_files(self, tmp_path, monkeypatch):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "test.soul").write_bytes(b"fake soul data")
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

    def test_empty_models_dir(self, tmp_path, monkeypatch):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

    def test_no_models_dir(self, tmp_path, monkeypatch):
        from commands.models import cmd_models
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)


class TestCmdModelsInfo:
    def test_missing_model_exits(self):
        from commands.models import _cmd_models_info
        args = MagicMock()
        args.model = "/nonexistent/path.soul"
        _cmd_models_info(args)


class TestCmdModelsCompare:
    def test_runs_without_error(self, tmp_path, monkeypatch):
        from commands.models import _cmd_models_compare
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        _cmd_models_compare(args)


class TestCmdModelsStatus:
    def test_no_cache_dir(self, monkeypatch):
        from commands.models import _cmd_models_status
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))
        args = MagicMock()
        _cmd_models_status(args)
