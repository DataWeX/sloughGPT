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
    import commands.data as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdDatasets:
    def test_header_called(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)
        mock_log.header.assert_called_with("Datasets")

    def test_lists_datasets_as_rows(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        (ds_dir / "shakespeare").mkdir()
        (ds_dir / "shakespeare" / "input.txt").write_text("hello\nworld\n")
        (ds_dir / "wiki").mkdir()
        (ds_dir / "wiki" / "input.txt").write_text("a\n")
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)

        table_call = mock_log.table.call_args
        rows = table_call[0][1]
        names = [r[0] for r in rows]
        assert "shakespeare" in names
        assert "wiki" in names

    def test_total_size_accumulated(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        (ds_dir / "ds1").mkdir()
        (ds_dir / "ds1" / "file.txt").write_text("x" * 1000)
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)

        kv_calls = {c[0][0]: c[0][1] for c in mock_log.key_value.call_args_list}
        assert "Total" in kv_calls
        assert kv_calls["Total"] != "0 B"

    def test_with_registry_vocab_info(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_datasets
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        (ds_dir / "test").mkdir()
        (ds_dir / "test" / "data.txt").write_text("x")
        registry = {"test": {"meta": {"vocab_size": 5000}}}
        (ds_dir / "registry.json").write_text(json.dumps(registry))
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_datasets(args)

        table_call = mock_log.table.call_args
        rows = table_call[0][1]
        vocab_row = [r for r in rows if r[0] == "test"]
        assert len(vocab_row) == 1
        assert vocab_row[0][2] == "5000"


class TestCmdDatasetImportUrl:
    def test_import_url_creates_file(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_dataset_import
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.text = "test content here"
        mock_response.raise_for_status = MagicMock()

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_response))

        args = MagicMock()
        args.url = "http://example.com/data.txt"
        args.name = "test_import"
        cmd_dataset_import(args, source="url")

        output_file = ds_dir / "test_import" / "input.txt"
        assert output_file.exists()
        assert output_file.read_text() == "test content here"

    def test_import_jsonl_writes_corpus(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_dataset_import
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.text = '{"text": "hello"}\n'
        mock_response.raise_for_status = MagicMock()

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_response))

        args = MagicMock()
        args.url = "http://example.com/data.jsonl"
        args.name = "jsonl_test"
        cmd_dataset_import(args, source="url")

        output_file = ds_dir / "jsonl_test" / "corpus.jsonl"
        assert output_file.exists()

    def test_import_url_success_logged(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_dataset_import
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.raise_for_status = MagicMock()

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(return_value=mock_response))

        args = MagicMock()
        args.url = "http://example.com/data.txt"
        args.name = "test"
        cmd_dataset_import(args, source="url")

        mock_log.success.assert_called()
        assert "Downloaded" in mock_log.success.call_args[0][0]

    def test_import_url_error_logged(self, mock_log, tmp_path, monkeypatch):
        from commands.data import cmd_dataset_import
        ds_dir = tmp_path / "data"
        ds_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=requests.ConnectionError("timeout")))

        args = MagicMock()
        args.url = "http://example.com/data.txt"
        args.name = "test"
        cmd_dataset_import(args, source="url")

        mock_log.error.assert_called()


class TestCmdDataToolStats:
    def test_file_stats_lines_and_chars(self, mock_log, tmp_path):
        from commands.data import cmd_data_tool
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")
        args = MagicMock()
        args.path = str(test_file)
        cmd_data_tool(args, subcmd="stats")

        mock_log.header.assert_called_with("File Statistics")
        kv_calls = {c[0][0]: c[0][1] for c in mock_log.key_value.call_args_list}
        assert kv_calls["Lines"] == "3"

    def test_missing_path_logs_error(self, mock_log):
        from commands.data import cmd_data_tool
        args = MagicMock()
        args.path = "/nonexistent/path"
        cmd_data_tool(args, subcmd="stats")
        mock_log.error.assert_called_with("Path not found: /nonexistent/path")

    def test_directory_stats(self, mock_log, tmp_path):
        from commands.data import cmd_data_tool
        ds_dir = tmp_path / "mydata"
        ds_dir.mkdir()
        (ds_dir / "file1.txt").write_text("hello\n")
        (ds_dir / "file2.txt").write_text("world\n")
        args = MagicMock()
        args.path = str(ds_dir)
        cmd_data_tool(args, subcmd="stats")

        mock_log.header.assert_called_with("Directory Statistics")
        kv_calls = {c[0][0]: c[0][1] for c in mock_log.key_value.call_args_list}
        assert kv_calls["Files"] == "2"
