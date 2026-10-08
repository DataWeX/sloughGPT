"""Tests for apps/cli/src/commands/models.py — model management commands."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "apps", "cli", "src"))


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.models as mod

    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


class TestCmdModels:
    def test_header_called(self, mock_log, monkeypatch):
        import utils.helpers
        from commands.models import cmd_models

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

        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [soul])
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        cmd_models(args)

        # First table call is soul files; second is architectures
        assert mock_log.table.call_count >= 1
        first_table_args = mock_log.table.call_args_list[0][0]
        rows = first_table_args[1]
        assert any("test.soul" in r[0] for r in rows)

    def test_provenance_column_is_rendered(self, mock_log, tmp_path, monkeypatch):
        """Provenance is an axis of identity, so it gets SHOWN, not suppressed.

        Declared at save time by ``save_soul(record=...)``, so a checkpoint
        predating the field reads ``-`` (unknown) rather than an inferred
        value, while a declared one comes through verbatim.
        """
        from commands.models import cmd_models

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        declared = models_dir / "declared.soul"
        declared.write_bytes(b"fake soul data")
        declared.with_name("declared.soul.meta.json").write_text(
            '{"provenance": "training"}', encoding="utf-8"
        )
        unknown = models_dir / "unknown.soul"
        unknown.write_bytes(b"fake soul data")

        import utils.helpers

        monkeypatch.setattr(
            utils.helpers, "local_soul_candidate_paths", lambda x: [declared, unknown]
        )
        monkeypatch.chdir(tmp_path)
        cmd_models(MagicMock())

        headers, rows = mock_log.table.call_args_list[0][0][:2]
        assert headers[-1] == "Provenance"
        idx = headers.index("Provenance")
        by_name = {r[0]: r for r in rows}
        assert by_name["declared.soul"][idx] == "training"
        assert by_name["unknown.soul"][idx] == "-"

    def test_format_and_hash_come_from_bytes_not_the_name(self, mock_log, tmp_path, monkeypatch):
        """The two columns that cannot be read off the filename are not.

        This table exists because name and size cannot tell same-stem
        checkpoints apart — two ``journey_select_trained`` siblings sit 1KB
        apart with losses 4.10 and 3.97. So Format must come from the header
        (a ``.soul`` whose bytes disagree says ``not-soul`` rather than
        inheriting the claim) and Hash from the sidecar's content digest,
        truncated, reading ``-`` when no digest was recorded.

        Both are silent-failure columns: drop the classify call and every
        Format quietly follows the name again; drop the sidecar field and
        every Hash reads ``-``. Nothing else in the suite would notice.
        """
        from commands.models import cmd_models

        from domain.inference._internal.slo_format import SOU_MAGIC

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # The name claims soul; the header disagrees.
        stray = models_dir / "claimed.soul"
        stray.write_bytes(b"not actually a soul")
        stray.with_name("claimed.soul.meta.json").write_text(
            '{"soul_name": "claimed"}', encoding="utf-8"
        )
        # Real magic, with the digest save_soul stamped beside it.
        kept = models_dir / "kept.soul"
        kept.write_bytes(SOU_MAGIC + b"\x00" * 32)
        kept.with_name("kept.soul.meta.json").write_text(
            '{"soul_name": "kept", "integrity_hash": "deadbeefcafe1234"}', encoding="utf-8"
        )

        import utils.helpers

        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [stray, kept])
        monkeypatch.chdir(tmp_path)
        cmd_models(MagicMock())

        headers, rows = mock_log.table.call_args_list[0][0][:2]
        assert headers == ["Name", "Size", "Format", "Hash", "Loss", "Born", "Provenance"]
        by_name = {r[0]: r for r in rows}
        fmt_idx, hash_idx = headers.index("Format"), headers.index("Hash")
        assert by_name["claimed.soul"][fmt_idx] == "not-soul"
        assert by_name["claimed.soul"][hash_idx] == "-"
        assert by_name["kept.soul"][fmt_idx] == "soul"
        assert by_name["kept.soul"][hash_idx] == "deadbeefca"

    def test_no_soul_files_info(self, mock_log, monkeypatch):
        import utils.helpers
        from commands.models import cmd_models

        monkeypatch.setattr(utils.helpers, "local_soul_candidate_paths", lambda x: [])
        monkeypatch.chdir(Path("/tmp"))
        args = MagicMock()
        cmd_models(args)
        mock_log.info.assert_any_call("No soul files found")

    def test_architectures_section(self, mock_log, monkeypatch):
        import utils.helpers
        from commands.models import cmd_models

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

    def test_identity_is_logged_for_a_real_checkpoint(self, mock_log, tmp_path):
        """The detail view owes the same identity the list shows.

        Tier reaches ``canonical`` through the suffix policy and provenance
        through ``record=``; a real file is required because ``classify_soul``
        only applies the suffix fallback when the content is a soul (a
        renamed non-soul reports unknown rather than inheriting a tier it
        cannot corroborate).
        """
        from commands.models import _cmd_models_info

        from domain.inference import save_soul

        soul = tmp_path / "identified.soul"
        save_soul(None, str(soul), weights_only=True, record="training")

        args = MagicMock()
        args.model = str(soul)
        _cmd_models_info(args)

        logged = {c[0][0]: c[0][1] for c in mock_log.key_value.call_args_list}
        assert logged["Format"]
        assert logged["Tier"] == "canonical"
        assert logged["Provenance"] == "training"
        assert logged["Hash"]

    def test_identity_is_shown_even_when_the_load_fails(self, mock_log, tmp_path, monkeypatch):
        """Identity must not depend on parsing weights.

        A file too broken to load is precisely the one you need to identify,
        so the block renders from header + sidecar before the load is
        attempted — not as a consequence of it succeeding.
        """
        from commands.models import _cmd_models_info

        import domain.training._internal.slonet as slonet
        from domain.inference import save_soul

        def unreadable(*_args, **_kwargs):
            raise RuntimeError("weights unreadable")

        monkeypatch.setattr(slonet, "import_from_sou", unreadable)

        soul = tmp_path / "broken.soul"
        save_soul(None, str(soul), weights_only=True, record="export")

        args = MagicMock()
        args.model = str(soul)
        _cmd_models_info(args)

        logged = {c[0][0]: c[0][1] for c in mock_log.key_value.call_args_list}
        assert logged["Provenance"] == "export"
        assert logged["Tier"] == "canonical"
        mock_log.error.assert_called()


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
