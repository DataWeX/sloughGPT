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
    import commands.models as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdModels:
    def test_header_called(self, mock_log, monkeypatch):
        from commands.models import cmd_models
        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        cmd_models(args)
        mock_log.header.assert_called_with("Available Models")

    def test_soul_files_listed(self, mock_log, tmp_path, monkeypatch):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        soul = models_dir / "test.soul"
        soul.write_bytes(b"fake soul data")

        import utils.helpers
        monkeypatch.setattr(
            utils.helpers, "local_soul_candidate_paths",
            lambda x: [soul]
        )
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

        # First table call is soul files; second is architectures
        assert mock_log.table.call_count >= 1
        first_table_args = mock_log.table.call_args_list[0][0]
        rows = first_table_args[1]
        assert any("test.soul" in r[0] for r in rows)

    def test_no_soul_files_info(self, mock_log, monkeypatch):
        from commands.models import cmd_models
        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        cmd_models(args)
        mock_log.info.assert_any_call("No soul files found")

    def test_architectures_section(self, mock_log, monkeypatch):
        from commands.models import cmd_models
        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        cmd_models(args)
        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "Available Architectures" in sections

    def test_slnc_section(self, mock_log, tmp_path, monkeypatch):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        slnc = models_dir / "model.slnc"
        slnc.write_bytes(b"fake slnc data")

        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "Compiled Models (.slnc)" in sections

    def test_safetensors_section(self, mock_log, tmp_path, monkeypatch):
        from commands.models import cmd_models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        st = models_dir / "model.safetensors"
        st.write_bytes(b"fake st data")

        import utils.helpers
        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "SafeTensors (.safetensors)" in sections


class TestCmdModelsInfo:
    def test_missing_model_logs_error(self, mock_log):
        from commands.models import _cmd_models_info
        args = MagicMock()
        args.model = "/nonexistent/path.soul"
        _cmd_models_info(args)
        mock_log.error.assert_called()
        assert "not found" in mock_log.error.call_args[0][0].lower()

    def test_error_message_contains_path(self, mock_log):
        from commands.models import _cmd_models_info
        args = MagicMock()
        args.model = "/tmp/fake_model.soul"
        _cmd_models_info(args)
        assert "/tmp/fake_model.soul" in mock_log.error.call_args[0][0]


class TestCmdModelsCompare:
    def test_header_called(self, mock_log, monkeypatch):
        from commands.models import _cmd_models_compare
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        _cmd_models_compare(args)
        mock_log.header.assert_called_with("Model Comparison")

    def test_model_specs_section(self, mock_log, monkeypatch):
        from commands.models import _cmd_models_compare
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        _cmd_models_compare(args)
        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "Model Specifications" in sections

    def test_benchmark_results_if_exist(self, mock_log, tmp_path, monkeypatch):
        from commands.models import _cmd_models_compare
        bench_dir = tmp_path / "data" / "experiments" / "benchmarks"
        bench_dir.mkdir(parents=True)
        (bench_dir / "test.json").write_text('{"model": "gpt2", "tokens_per_second": 10.5}')
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        _cmd_models_compare(args)
        sections = [c[0][0] for c in mock_log.section.call_args_list]
        assert "Benchmark Results" in sections


class TestCmdModelsStatus:
    def test_no_cache_dir(self, mock_log, monkeypatch):
        from commands.models import _cmd_models_status
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))
        args = MagicMock()
        _cmd_models_status(args)
        mock_log.info.assert_called()
        assert "HuggingFace cache" in mock_log.info.call_args[0][0]
