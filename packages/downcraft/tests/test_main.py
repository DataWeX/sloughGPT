"""Tests for the downcraft CLI (``downcraft.__main__``).

Verifies the HF-agnostic command surface: ``url``, ``status``, ``list``.
The ``hf`` and ``verify`` subcommands were removed when downcraft became
HuggingFace-agnostic.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from downcraft import __main__ as cli
from conftest import RangeHandler, _range_url


class TestHelp:
    def test_help_lists_only_generic_commands(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for cmd in ("url", "status", "list"):
            assert cmd in out
        # HF-specific commands must be gone.
        for removed in ("hf ", "verify", "--hf-home", "model_id"):
            assert removed not in out


class TestStatus:
    def test_unknown_key_reports_not_found(self, capsys):
        cli.cmd_status(type("A", (), {"key": "https://example.com/nope"}))
        assert "not found in state" in capsys.readouterr().out

    def test_tracked_key_shows_status(self, capsys):
        from downcraft.state import PersistentState
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch("downcraft.state.get_state") as mock_state:
                st = PersistentState(state_dir=Path(td) / "state")
                key = "https://example.com/f.bin"
                st.create(key, cache_dir="")
                st.update_file_progress(
                    key,
                    file_path="f.bin",
                    url=key,
                    bytes_downloaded=5 * 1024 * 1024,
                    total_bytes=10 * 1024 * 1024,
                )
                mock_state.return_value = st
                cli.cmd_status(type("A", (), {"key": key}))
                out = capsys.readouterr().out
                assert "downloading" in out
                assert "5 / 10 MB (50.0%)" in out


class TestList:
    def test_empty_state(self, capsys):
        cli.cmd_list(type("A", (), {}))
        assert "No downloads tracked" in capsys.readouterr().out

    def test_lists_tracked_downloads(self, capsys):
        from downcraft.state import PersistentState
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with patch("downcraft.state.get_state") as mock_state:
                st = PersistentState(state_dir=Path(td) / "state")
                st.create("https://example.com/a.bin", cache_dir="")
                st.create("https://example.com/b.bin", cache_dir="")
                mock_state.return_value = st
                cli.cmd_list(type("A", (), {}))
                out = capsys.readouterr().out
                assert "a.bin" in out
                assert "b.bin" in out


class TestUrl:
    def test_downloads_url(self, range_server, capsys, tmp_path):
        content = b"cli download test payload"
        RangeHandler.payloads["/cli.bin"] = content
        dest = tmp_path / "cli.bin"
        cli.cmd_url(
            type("A", (), {"url": _range_url(range_server, "/cli.bin"), "dest": str(dest)})
        )
        out = capsys.readouterr().out
        assert "Done" in out
        assert dest.read_bytes() == content

    def test_bad_url_exits(self, range_server):
        import io
        dest = str(Path("/tmp/nonexistent_cli_out.bin"))
        with pytest.raises(SystemExit) as exc:
            cli.cmd_url(
                type("A", (), {"url": "http://127.0.0.1:1/missing", "dest": dest})
            )
        assert exc.value.code == 1
