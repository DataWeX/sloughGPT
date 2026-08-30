"""Tests for domains.infrastructure.hf_hub — HFFile, ResumeInfo, _matches_ignore, _strip_incomplete_suffix, _derive_model_id, _match_repo_file, IGNORED_PATTERNS."""

from pathlib import Path

import pytest

from domains.infrastructure.hf_hub import (
    HFFile,
    ResumeInfo,
    _matches_ignore,
    _strip_incomplete_suffix,
    _derive_model_id,
    _match_repo_file,
    IGNORED_PATTERNS,
)


# ── IGNORED_PATTERNS ────────────────────────────────────────────────────────

class TestIgnoredPatterns:
    def test_contains_h5(self):
        assert "*.h5" in IGNORED_PATTERNS

    def test_contains_ot(self):
        assert "*.ot" in IGNORED_PATTERNS

    def test_contains_onnx(self):
        assert "*.onnx" in IGNORED_PATTERNS

    def test_contains_msgpack(self):
        assert "*.msgpack" in IGNORED_PATTERNS

    def test_contains_gguf(self):
        assert "*.gguf" in IGNORED_PATTERNS

    def test_contains_tflite(self):
        assert "*.tflite" in IGNORED_PATTERNS

    def test_length_at_least_six(self):
        assert len(IGNORED_PATTERNS) >= 6


# ── HFFile ───────────────────────────────────────────────────────────────────

class TestHFFile:
    def test_fields(self):
        hf = HFFile(path="model.bin", size=1024, checksum="abc", download_url="http://x")
        assert hf.path == "model.bin"
        assert hf.size == 1024
        assert hf.checksum == "abc"
        assert hf.download_url == "http://x"
        assert hf.is_ignored is False

    def test_is_ignored_default_false(self):
        hf = HFFile(path="a", size=0, checksum="", download_url="")
        assert hf.is_ignored is False

    def test_is_ignored_set_true(self):
        hf = HFFile(path="a", size=0, checksum="", download_url="", is_ignored=True)
        assert hf.is_ignored is True

    def test_zero_size(self):
        hf = HFFile(path="empty", size=0, checksum="", download_url="")
        assert hf.size == 0

    def test_large_size(self):
        hf = HFFile(path="big", size=10**12, checksum="", download_url="")
        assert hf.size == 10**12

    def test_empty_checksum(self):
        hf = HFFile(path="p", size=100, checksum="", download_url="")
        assert hf.checksum == ""

    def test_long_checksum(self):
        long = "a" * 64
        hf = HFFile(path="p", size=100, checksum=long, download_url="")
        assert hf.checksum == long


# ── ResumeInfo ───────────────────────────────────────────────────────────────

class TestResumeInfo:
    def test_fields(self):
        ri = ResumeInfo(
            model_id="gpt2", repo_path="model.safetensors",
            partial_path=Path("/tmp/p"), final_path=Path("/tmp/f"),
            resume_offset=100, total_bytes=1000,
            download_url="http://x", checksum="abc", complete=False,
        )
        assert ri.model_id == "gpt2"
        assert ri.repo_path == "model.safetensors"
        assert ri.partial_path == Path("/tmp/p")
        assert ri.final_path == Path("/tmp/f")
        assert ri.resume_offset == 100
        assert ri.total_bytes == 1000
        assert ri.download_url == "http://x"
        assert ri.checksum == "abc"
        assert ri.complete is False

    def test_complete_true(self):
        ri = ResumeInfo(
            model_id="m", repo_path="r", partial_path=Path("/p"),
            final_path=Path("/f"), resume_offset=0, total_bytes=0,
            download_url="", checksum="", complete=True,
        )
        assert ri.complete is True

    def test_zero_offset(self):
        ri = ResumeInfo(
            model_id="m", repo_path="r", partial_path=Path("/p"),
            final_path=Path("/f"), resume_offset=0, total_bytes=100,
            download_url="", checksum="", complete=False,
        )
        assert ri.resume_offset == 0

    def test_large_offset(self):
        ri = ResumeInfo(
            model_id="m", repo_path="r", partial_path=Path("/p"),
            final_path=Path("/f"), resume_offset=10**9, total_bytes=10**10,
            download_url="", checksum="", complete=False,
        )
        assert ri.resume_offset == 10**9


# ── _matches_ignore ─────────────────────────────────────────────────────────

class TestMatchesIgnore:
    def test_h5(self):
        assert _matches_ignore("model.h5") is True

    def test_ot(self):
        assert _matches_ignore("model.ot") is True

    def test_onnx(self):
        assert _matches_ignore("model.onnx") is True

    def test_msgpack(self):
        assert _matches_ignore("model.msgpack") is True

    def test_gguf(self):
        assert _matches_ignore("model.gguf") is True

    def test_tflite(self):
        assert _matches_ignore("model.tflite") is True

    def test_normal_bin(self):
        assert _matches_ignore("model.bin") is False

    def test_normal_safetensors(self):
        assert _matches_ignore("model.safetensors") is False

    def test_onnx_dir(self):
        assert _matches_ignore("onnx/model.bin") is True

    def test_tf_dir(self):
        assert _matches_ignore("tf/weights.bin") is True

    def test_config_json(self):
        assert _matches_ignore("config.json") is False

    def test_tokenizer(self):
        assert _matches_ignore("tokenizer.json") is False

    def test_h5_in_subdir(self):
        assert _matches_ignore("subdir/model.h5") is True

    def test_empty_string(self):
        assert _matches_ignore("") is False

    def test_dot_prefix_not_ignored(self):
        assert _matches_ignore(".gitignore") is False


# ── _strip_incomplete_suffix ────────────────────────────────────────────────

class TestStripIncompleteSuffix:
    def test_sgpart(self):
        assert _strip_incomplete_suffix("model.bin.sgpart") == "model.bin"

    def test_incomplete(self):
        assert _strip_incomplete_suffix("model.bin.incomplete") == "model.bin"

    def test_no_suffix(self):
        assert _strip_incomplete_suffix("model.bin") == "model.bin"

    def test_safetensors_sgpart(self):
        assert _strip_incomplete_suffix("model.safetensors.sgpart") == "model.safetensors"

    def test_double_suffix(self):
        assert _strip_incomplete_suffix("model.bin.sgpart.incomplete") == "model.bin.sgpart"

    def test_empty_string(self):
        assert _strip_incomplete_suffix("") == ""

    def test_only_suffix(self):
        assert _strip_incomplete_suffix(".sgpart") == ""

    def test_suffix_in_middle(self):
        assert _strip_incomplete_suffix("model.sgpart.bin") == "model.sgpart.bin"

    def test_multiple_dots(self):
        assert _strip_incomplete_suffix("my.model.v2.bin.sgpart") == "my.model.v2.bin"


# ── _derive_model_id ────────────────────────────────────────────────────────

class TestDeriveModelId:
    def test_single_slash_model(self):
        result = _derive_model_id("/cache/models--gpt2/snapshots/abc/file.bin")
        assert result == "gpt2"

    def test_org_model(self):
        result = _derive_model_id("/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/abc/file.bin")
        assert result == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_no_models_prefix(self):
        result = _derive_model_id("/some/random/path/file.bin")
        assert result is None

    def test_root_path(self):
        result = _derive_model_id("/file.bin")
        assert result is None

    def test_deeply_nested(self):
        result = _derive_model_id("/a/b/c/models--meta-llama--Llama-3/d/snapshots/x/w.bin")
        assert result == "meta-llama/Llama-3"


# ── _match_repo_file ────────────────────────────────────────────────────────

class TestMatchRepoFile:
    def test_exact_match(self):
        files = [HFFile(path="model.bin", size=100, checksum="", download_url="")]
        match = _match_repo_file("model.bin", files)
        assert match is not None
        assert match.path == "model.bin"

    def test_basename_match(self):
        files = [HFFile(path="shard-00001-of-00003.safetensors", size=100, checksum="", download_url="")]
        match = _match_repo_file("shard-00001-of-00003.safetensors", files)
        assert match is not None

    def test_no_match(self):
        files = [HFFile(path="model.bin", size=100, checksum="", download_url="")]
        match = _match_repo_file("nonexistent.bin", files)
        assert match is None

    def test_empty_files(self):
        match = _match_repo_file("anything", [])
        assert match is None

    def test_prefers_exact_over_basename(self):
        files = [
            HFFile(path="dir/model.bin", size=100, checksum="", download_url=""),
            HFFile(path="model.bin", size=200, checksum="", download_url=""),
        ]
        match = _match_repo_file("model.bin", files)
        assert match.path == "model.bin"
        assert match.size == 200

    def test_partial_name_match(self):
        files = [
            HFFile(path="model-00001-of-00003.safetensors", size=100, checksum="", download_url=""),
            HFFile(path="model-00002-of-00003.safetensors", size=100, checksum="", download_url=""),
        ]
        match = _match_repo_file("model-00001-of-00003.safetensors", files)
        assert match is not None
        assert "00001" in match.path
