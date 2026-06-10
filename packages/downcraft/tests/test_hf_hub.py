"""Tests for downcraft.hf_hub — HuggingFace Hub integration."""

import os
import tempfile
from pathlib import Path

import pytest

from downcraft.hf_hub import (
    _matches_ignore,
    get_cache_dir,
    is_download_complete,
    list_model_files,
)


class TestMatchesIgnore:
    def test_h5_ignored(self):
        assert _matches_ignore("model.h5") is True

    def test_onnx_ignored(self):
        assert _matches_ignore("model.onnx") is True

    def test_gguf_ignored(self):
        assert _matches_ignore("model.gguf") is True

    def test_msgpack_ignored(self):
        assert _matches_ignore("model.msgpack") is True

    def test_tflite_ignored(self):
        assert _matches_ignore("model.tflite") is True

    def test_ot_ignored(self):
        assert _matches_ignore("model.ot") is True

    def test_safetensors_not_ignored(self):
        assert _matches_ignore("model.safetensors") is False

    def test_bin_not_ignored(self):
        assert _matches_ignore("pytorch_model.bin") is False

    def test_json_not_ignored(self):
        assert _matches_ignore("config.json") is False

    def test_onnx_subdirectory_ignored(self):
        assert _matches_ignore("onnx/model.onnx") is True

    def test_tf_subdirectory_ignored(self):
        assert _matches_ignore("tf/variables") is True

    def test_regular_subdirectory_not_ignored(self):
        assert _matches_ignore("not-onnx/file.bin") is False

    def test_case_sensitive(self):
        assert _matches_ignore("Model.ONNX") is False  # fnmatch is case-sensitive on Linux/Mac


class TestGetCacheDir:
    def test_default_path(self):
        cache = get_cache_dir("gpt2")
        assert "models--gpt2" in str(cache)

    def test_with_org(self):
        cache = get_cache_dir("Qwen/Qwen2.5-0.5B-Instruct")
        assert "models--Qwen--Qwen2.5-0.5B-Instruct" in str(cache)

    def test_respects_hf_home(self):
        cache = get_cache_dir("gpt2", hf_home="/custom/hf")
        # When hf_home is provided, it is used directly as the base
        assert str(cache) == os.path.join("/custom/hf", "models--gpt2")

    def test_respects_hf_home_no_trailing(self):
        cache = get_cache_dir("gpt2", hf_home="/custom")
        assert str(cache) == os.path.join("/custom", "models--gpt2")


class TestIsDownloadComplete:
    def test_nonexistent_cache(self):
        assert is_download_complete("fake-model-nonexistent", hf_home="/tmp/nonexistent_hf_cache_xyz") is False

    def test_empty_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_no_refs_main(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            (cache_dir / "snapshots" / "abc123").mkdir(parents=True)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_no_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_complete_with_safetensors(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_complete_with_bin(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "pytorch_model.bin").write_bytes(b"x" * 2000)
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is True

    def test_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / ".incomplete").write_text("incomplete")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_lock_file_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 2000)
            (cache_dir / "some.lock").write_text("locked")
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False

    def test_small_file_less_than_1kb_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "hub" / "models--test-model"
            cache_dir.mkdir(parents=True)
            refs = cache_dir / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123")
            snap = cache_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"x" * 500)  # < 1KB
            assert is_download_complete("test-model", hf_home=str(Path(td) / "hub")) is False


class TestListModelFiles:
    def test_real_model_returns_files(self):
        files = list_model_files("gpt2")
        assert len(files) > 0

        # Should include config, tokenizer, and model weights
        paths = [f.path for f in files]
        assert "config.json" in paths
        assert any("model" in p for p in paths)

        # Should exclude ignored formats
        assert not any(f.is_ignored for f in files if "model.safetensors" in f.path)

    def test_real_model_has_sizes(self):
        files = list_model_files("gpt2")
        safetensors = [f for f in files if f.path.endswith(".safetensors")]
        if safetensors:
            assert safetensors[0].size > 0
            assert safetensors[0].checksum != ""

    def test_real_model_has_download_urls(self):
        files = list_model_files("gpt2")
        non_ignored = [f for f in files if not f.is_ignored]
        assert len(non_ignored) > 0
        for f in non_ignored:
            assert f.download_url.startswith("https://")
            assert "gpt2" in f.download_url

    def test_ignored_files_have_empty_url(self):
        files = list_model_files("gpt2")
        ignored = [f for f in files if f.is_ignored]
        for f in ignored:
            assert f.download_url == ""
