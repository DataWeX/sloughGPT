"""Tests for domains.infrastructure.hf_hub — HFFile, ResumeInfo, _matches_ignore, _strip_incomplete_suffix."""

from pathlib import Path
from domains.infrastructure.hf_hub import (
    HFFile, ResumeInfo, _matches_ignore, _strip_incomplete_suffix, IGNORED_PATTERNS,
)


class TestHFFile:
    def test_fields(self):
        hf = HFFile(path="model.bin", size=1024, checksum="abc", download_url="http://x")
        assert hf.path == "model.bin"
        assert hf.size == 1024
        assert hf.is_ignored is False


class TestResumeInfo:
    def test_fields(self):
        ri = ResumeInfo(
            model_id="gpt2", repo_path="model.safetensors",
            partial_path=Path("/tmp/p"), final_path=Path("/tmp/f"),
            resume_offset=100, total_bytes=1000,
            download_url="http://x", checksum="abc", complete=False,
        )
        assert ri.model_id == "gpt2"
        assert ri.resume_offset == 100
        assert ri.complete is False


class TestMatchesIgnore:
    def test_h5(self):
        assert _matches_ignore("model.h5") is True
    def test_ot(self):
        assert _matches_ignore("model.ot") is True
    def test_onnx(self):
        assert _matches_ignore("model.onnx") is True
    def test_normal(self):
        assert _matches_ignore("model.bin") is False
    def test_onnx_dir(self):
        assert _matches_ignore("onnx/model.bin") is True
    def test_tf_dir(self):
        assert _matches_ignore("tf/weights.bin") is True
    def test_safetensors(self):
        assert _matches_ignore("model.safetensors") is False


class TestStripIncompleteSuffix:
    def test_sgpart(self):
        assert _strip_incomplete_suffix("model.bin.sgpart") == "model.bin"
    def test_incomplete(self):
        assert _strip_incomplete_suffix("model.bin.incomplete") == "model.bin"
    def test_no_suffix(self):
        assert _strip_incomplete_suffix("model.bin") == "model.bin"
