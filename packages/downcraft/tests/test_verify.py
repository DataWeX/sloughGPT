"""Tests for downcraft.verify — SHA-256 integrity verification.

Model-level ``list_missing_files`` moved to
``domains.infrastructure.hf_hub`` (see core-py tests).
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from downcraft.verify import _sha256_of, verify_file


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
