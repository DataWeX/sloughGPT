"""Tests for downcraft.verify — SHA-256 integrity verification."""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from downcraft.verify import _sha256_of, verify_file, list_missing_files


class TestSha256Of:
    def test_small_file(self, sample_file):
        expected = hashlib.sha256(sample_file.read_bytes()).hexdigest()
        assert _sha256_of(sample_file) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_text("")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_of(f) == expected

    def test_large_file(self, tmp_path):
        data = os.urandom(1024 * 1024)  # 1 MB
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_of(f) == expected

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _sha256_of(Path("/tmp/nonexistent_sha256_test_file_xyz"))


class TestVerifyFile:
    def test_valid_checksum(self, sample_file):
        expected = hashlib.sha256(sample_file.read_bytes()).hexdigest()
        assert verify_file(sample_file, expected) is True

    def test_invalid_checksum(self, sample_file):
        assert verify_file(sample_file, "wrongchecksum" * 8) is False

    def test_nonexistent_file(self):
        assert verify_file(Path("/tmp/nonexistent_verify_test"), "abc") is False


class TestListMissingFiles:
    def test_nonexistent_model_returns_all_files(self):
        with patch("downcraft.hf_hub.list_model_files") as mock_list:
            from downcraft.hf_hub import HFFile
            mock_list.return_value = [
                HFFile(path="config.json", size=100, checksum="abc", download_url="https://x.com/cfg"),
                HFFile(path="model.safetensors", size=1000, checksum="def", download_url="https://x.com/model"),
            ]
            missing = list_missing_files("test-model", hf_home="/tmp/fake_hf_home")
            assert len(missing) == 2
            assert "config.json" in missing

    def test_missing_files_when_snap_missing(self):
        """When snapshot doesn't exist, all files should be reported missing."""
        with patch("downcraft.hf_hub.list_model_files") as mock_list:
            from downcraft.hf_hub import HFFile
            mock_list.return_value = [
                HFFile(path="weights.bin", size=500, checksum="abc", download_url="https://x.com/w"),
            ]
            missing = list_missing_files("no-snap-model", hf_home="/tmp/fake_hf_home")
            assert len(missing) == 1
            assert "weights.bin" in missing
