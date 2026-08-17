"""Tests for apps/cli/src/commands/data.py — dataset management commands."""
import sys
import os
import json
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
    import commands.data as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdDatasets:
    def test_lists_datasets(self, tmp_path, monkeypatch, capsys):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        (ds_dir / "shakespeare").mkdir()
        (ds_dir / "wiki").mkdir()
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)
        out = capsys.readouterr().out
        assert "Datasets" in out

    def test_empty_datasets_dir(self, tmp_path, monkeypatch, capsys):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)
        out = capsys.readouterr().out
        assert "Datasets" in out


class TestCmdDatasetImport:
    def test_import_url(self, tmp_path, monkeypatch, capsys):
        from commands.data import cmd_dataset_import
        ds_dir = tmp_path / "datasets"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.text = "test content"
        mock_response.raise_for_status = MagicMock()

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_response))

        args = MagicMock()
        args.url = "http://example.com/data.txt"
        args.name = "test_import"
        cmd_dataset_import(args, source="url")
        out = capsys.readouterr().out
        assert "Import" in out or "Downloaded" in out


class TestCmdDataTool:
    def test_stats_file(self, tmp_path, capsys):
        from commands.data import cmd_data_tool
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")
        args = MagicMock()
        args.path = str(test_file)
        cmd_data_tool(args, subcmd="stats")
        out = capsys.readouterr().out
        assert "Statistics" in out or "Lines" in out

    def test_missing_path(self, capsys):
        from commands.data import cmd_data_tool
        args = MagicMock()
        args.path = "/nonexistent/path"
        cmd_data_tool(args, subcmd="stats")
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "Path not found" in out

    def test_stats_directory(self, tmp_path, capsys):
        from commands.data import cmd_data_tool
        ds_dir = tmp_path / "mydata"
        ds_dir.mkdir()
        (ds_dir / "file1.txt").write_text("hello\n")
        (ds_dir / "file2.txt").write_text("world\n")
        args = MagicMock()
        args.path = str(ds_dir)
        cmd_data_tool(args, subcmd="stats")
        out = capsys.readouterr().out
        assert "Statistics" in out or "Files" in out
