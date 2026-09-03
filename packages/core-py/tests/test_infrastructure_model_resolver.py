"""Tests for model_resolver — find models in HuggingFace cache."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from domains.infrastructure.model_resolver import find_safetensors, get_model_dir


class TestGetModelDir:
    def test_returns_path(self):
        result = get_model_dir("bert-base-uncased")
        assert isinstance(result, Path)

    def test_cache_id_format(self):
        result = get_model_dir("bert-base-uncased")
        assert "models--bert-base-uncased" in str(result)

    def test_slash_replaced(self):
        result = get_model_dir("org/model-name")
        assert "models--org--model-name" in str(result)


class TestFindSafetensors:
    def test_returns_none_when_no_file(self, tmp_path: Path):
        result = find_safetensors(tmp_path)
        assert result is None

    def test_finds_in_snapshots(self, tmp_path: Path):
        snap_dir = tmp_path / "snapshots" / "abc"
        snap_dir.mkdir(parents=True)
        st_file = snap_dir / "model.safetensors"
        st_file.write_text("fake")
        result = find_safetensors(tmp_path)
        assert result == st_file

    def test_finds_in_root(self, tmp_path: Path):
        st_file = tmp_path / "model.safetensors"
        st_file.write_text("fake")
        result = find_safetensors(tmp_path)
        assert result == st_file
