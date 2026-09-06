"""Tests for domains.infrastructure.gpu.compile_shaders — pure logic coverage.

Covers path constants, shader list, naga discovery logic, subprocess compile
paths, and main() argument parsing. No real naga binary or GPU hardware needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.gpu.compile_shaders import (
    SHADERS_DIR,
    OUTPUT_DIR,
    COMPUTE_SHADERS,
    find_naga,
    compile_spirv,
    compile_hlsl,
    compile_msl,
    main,
)


# ── Constants ───────────────────────────────────────────────────────────


class TestShaderConstants:
    def test_shaders_dir_is_path(self):
        assert isinstance(SHADERS_DIR, Path)

    def test_output_dir_is_path(self):
        assert isinstance(OUTPUT_DIR, Path)

    def test_shaders_dir_points_to_shaders_subdir(self):
        assert SHADERS_DIR.name == "shaders"

    def test_output_dir_points_to_shaders_subdir(self):
        assert OUTPUT_DIR.name == "shaders"

    def test_shaders_and_output_same_directory(self):
        assert SHADERS_DIR == OUTPUT_DIR

    def test_dirs_are_under_gpu_module(self):
        gpu_dir = Path(__file__).resolve().parents[1] / "domains" / "infrastructure" / "gpu"
        assert SHADERS_DIR.parent == gpu_dir

    def test_compute_shaders_is_list(self):
        assert isinstance(COMPUTE_SHADERS, list)

    def test_compute_shaders_non_empty(self):
        assert len(COMPUTE_SHADERS) > 0

    def test_compute_shaders_contains_expected_names(self):
        expected = {"matmul", "softmax", "rmsnorm", "rope", "silu", "gelu"}
        assert set(COMPUTE_SHADERS) == expected

    def test_compute_shaders_are_strings(self):
        for name in COMPUTE_SHADERS:
            assert isinstance(name, str)

    def test_compute_shaders_no_duplicates(self):
        assert len(COMPUTE_SHADERS) == len(set(COMPUTE_SHADERS))


# ── find_naga ──────────────────────────────────────────────────────────


class TestFindNaga:
    def test_finds_naga_in_path(self, tmp_path):
        naga_bin = tmp_path / "naga"
        naga_bin.write_text("#!/bin/sh\n")
        naga_bin.chmod(0o755)

        env = {"PATH": str(tmp_path), "CARGO_HOME": "/nonexistent"}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == str(naga_bin)

    def test_finds_naga_cli_in_path(self, tmp_path):
        naga_bin = tmp_path / "naga-cli"
        naga_bin.write_text("#!/bin/sh\n")
        naga_bin.chmod(0o755)

        env = {"PATH": str(tmp_path), "CARGO_HOME": "/nonexistent"}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == str(naga_bin)

    def test_finds_naga_in_cargo_home(self, tmp_path):
        cargo_bin = tmp_path / "bin"
        cargo_bin.mkdir()
        naga_bin = cargo_bin / "naga"
        naga_bin.write_text("#!/bin/sh\n")
        naga_bin.chmod(0o755)

        env = {"PATH": "/nonexistent", "CARGO_HOME": str(tmp_path)}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == str(naga_bin)

    def test_defaults_to_naga_string(self):
        env = {"PATH": "/nonexistent", "CARGO_HOME": "/nonexistent"}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == "naga"

    def test_prefers_path_over_cargo(self, tmp_path):
        path_naga = tmp_path / "path_dir" / "naga"
        path_naga.parent.mkdir()
        path_naga.write_text("#!/bin/sh\n")
        path_naga.chmod(0o755)

        cargo_naga = tmp_path / "cargo_dir" / "bin" / "naga"
        cargo_naga.parent.mkdir(parents=True)
        cargo_naga.write_text("#!/bin/sh\n")
        cargo_naga.chmod(0o755)

        env = {"PATH": str(path_naga.parent), "CARGO_HOME": str(tmp_path / "cargo_dir")}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == str(path_naga)

    def test_prefers_naga_over_naga_cli(self, tmp_path):
        (tmp_path / "naga-cli").write_text("")
        (tmp_path / "naga").write_text("")

        env = {"PATH": str(tmp_path), "CARGO_HOME": "/nonexistent"}
        with patch.dict(os.environ, env, clear=False):
            result = find_naga()

        assert result == str(tmp_path / "naga")

    def test_cargo_default_path(self):
        env = {"PATH": "/nonexistent", "CARGO_HOME": ""}
        with patch.dict(os.environ, env, clear=False):
            with patch("pathlib.Path.home", return_value=Path("/fakehome")):
                with patch("pathlib.Path.exists", return_value=False):
                    result = find_naga()

        assert result == "naga"


# ── compile_spirv ──────────────────────────────────────────────────────


class TestCompileSpirv:
    def test_success(self, tmp_path):
        wgsl = tmp_path / "test.wgsl"
        out = tmp_path / "test.spv"
        wgsl.write_text("fn main() {}")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=256)
                result = compile_spirv("/usr/bin/naga", wgsl, out)

        assert result is True

    def test_failure(self, tmp_path, caplog):
        import logging
        wgsl = tmp_path / "bad.wgsl"
        out = tmp_path / "bad.spv"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "parse error at line 3"

        with caplog.at_level(logging.ERROR, logger="slo.gpu.compile_shaders"):
            with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
                result = compile_spirv("/usr/bin/naga", wgsl, out)

        assert result is False
        assert "SPIR-V FAILED" in caplog.text

    def test_passes_correct_command(self, tmp_path):
        wgsl = tmp_path / "shader.wgsl"
        out = tmp_path / "shader.spv"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result) as mock_run:
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=128)
                compile_spirv("/custom/naga", wgsl, out)

        mock_run.assert_called_once_with(
            ["/custom/naga", str(wgsl), str(out)],
            capture_output=True,
            text=True,
        )

    def test_capture_output_enabled(self, tmp_path):
        wgsl = tmp_path / "test.wgsl"
        out = tmp_path / "test.spv"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result) as mock_run:
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=64)
                compile_spirv("naga", wgsl, out)

        _, kwargs = mock_run.call_args
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True


# ── compile_hlsl ───────────────────────────────────────────────────────


class TestCompileHlsl:
    def test_success(self, tmp_path):
        wgsl = tmp_path / "test.wgsl"
        out = tmp_path / "test.hlsl"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=512)
                result = compile_hlsl("naga", wgsl, out)

        assert result is True

    def test_failure(self, tmp_path, capsys):
        wgsl = tmp_path / "bad.wgsl"
        out = tmp_path / "bad.hlsl"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "unsupported feature"

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
            result = compile_hlsl("naga", wgsl, out)

        assert result is False
        captured = capsys.readouterr()
        assert "HLSL FAILED" in captured.out

    def test_passes_correct_command(self, tmp_path):
        wgsl = tmp_path / "compute.wgsl"
        out = tmp_path / "compute.hlsl"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result) as mock_run:
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=200)
                compile_hlsl("naga", wgsl, out)

        mock_run.assert_called_once_with(
            ["naga", str(wgsl), str(out)],
            capture_output=True,
            text=True,
        )


# ── compile_msl ────────────────────────────────────────────────────────


class TestCompileMsl:
    def test_success(self, tmp_path):
        wgsl = tmp_path / "test.wgsl"
        out = tmp_path / "test.metal"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=384)
                result = compile_msl("naga", wgsl, out)

        assert result is True

    def test_failure(self, tmp_path, capsys):
        wgsl = tmp_path / "bad.wgsl"
        out = tmp_path / "bad.metal"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "metal: unknown intrinsic"

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result):
            result = compile_msl("naga", wgsl, out)

        assert result is False
        captured = capsys.readouterr()
        assert "MSL FAILED" in captured.out

    def test_passes_correct_command(self, tmp_path):
        wgsl = tmp_path / "compute.wgsl"
        out = tmp_path / "compute.metal"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("domains.infrastructure.gpu.compile_shaders.subprocess.run", return_value=mock_result) as mock_run:
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=300)
                compile_msl("naga", wgsl, out)

        mock_run.assert_called_once_with(
            ["naga", str(wgsl), str(out)],
            capture_output=True,
            text=True,
        )


# ── main() ─────────────────────────────────────────────────────────────


class TestMain:
    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_compiles_all_formats_when_no_flags(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        for name in COMPUTE_SHADERS:
            (shaders_dir / f"{name}.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])

        main()

        assert mock_spv.call_count == len(COMPUTE_SHADERS)
        assert mock_hlsl.call_count == len(COMPUTE_SHADERS)
        assert mock_msl.call_count == len(COMPUTE_SHADERS)

    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_spirv_only_flag(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "--spirv-only"])

        main()

        mock_spv.assert_called_once()
        mock_hlsl.assert_not_called()
        mock_msl.assert_not_called()

    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_hlsl_only_flag(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "--hlsl-only"])

        main()

        mock_spv.assert_not_called()
        mock_hlsl.assert_called_once()
        mock_msl.assert_not_called()

    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_msl_only_flag(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "--msl-only"])

        main()

        mock_spv.assert_not_called()
        mock_hlsl.assert_not_called()
        mock_msl.assert_called_once()

    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="/custom/naga")
    def test_naga_path_override(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "--naga", "/custom/naga"])

        main()

        mock_naga.assert_not_called()
        wgsl_path = shaders_dir / "matmul.wgsl"
        mock_spv.assert_called_with("/custom/naga", wgsl_path, shaders_dir / "matmul.spv")

    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=False)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_exits_on_failure(self, mock_naga, mock_spv, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])

        with pytest.raises(SystemExit, match="1"):
            main()

    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_skips_missing_wgsl_files(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        # Only create one shader file
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])

        main()

        # Only matmul should be compiled (1 call), not all 6
        assert mock_spv.call_count == 1
        assert mock_hlsl.call_count == 1
        assert mock_msl.call_count == 1

    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_creates_output_dir(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])

        main()

        assert shaders_dir.exists()

    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_compiles_correct_shader_files(self, mock_naga, mock_spv, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")
        (shaders_dir / "softmax.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py", "--spirv-only"])

        main()

        call_args_list = [call[0] for call in mock_spv.call_args_list]
        compiled_names = sorted([Path(a[1]).stem for a in call_args_list])
        assert compiled_names == ["matmul", "softmax"]

    @patch("domains.infrastructure.gpu.compile_shaders.compile_msl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_hlsl", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.compile_spirv", return_value=True)
    @patch("domains.infrastructure.gpu.compile_shaders.find_naga", return_value="naga")
    def test_output_paths_correct(self, mock_naga, mock_spv, mock_hlsl, mock_msl, tmp_path, monkeypatch):
        shaders_dir = tmp_path / "shaders"
        shaders_dir.mkdir()
        (shaders_dir / "matmul.wgsl").write_text("fn main() {}")

        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.SHADERS_DIR", shaders_dir)
        monkeypatch.setattr("domains.infrastructure.gpu.compile_shaders.OUTPUT_DIR", shaders_dir)
        monkeypatch.setattr(sys, "argv", ["compile_shaders.py"])

        main()

        wgsl_path = shaders_dir / "matmul.wgsl"
        mock_spv.assert_called_with("naga", wgsl_path, shaders_dir / "matmul.spv")
        mock_hlsl.assert_called_with("naga", wgsl_path, shaders_dir / "matmul.hlsl")
        mock_msl.assert_called_with("naga", wgsl_path, shaders_dir / "matmul.metal")
