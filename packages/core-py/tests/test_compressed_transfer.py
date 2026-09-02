"""Tests for domains.infrastructure.compressed_transfer — streaming compression, integrity, roundtrip."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import struct
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from domains.infrastructure.compressed_transfer import (
    MAGIC,
    HEADER_SIZE,
    CHUNK_SIZE,
    CompressionResult,
    DownloadResult,
    compress_stream,
    decompress_stream,
    compress_file,
    decompress_file,
    compress_bytes,
    decompress_bytes,
    compressed_file_iterator,
    peek_compressed_header,
)


# ── CompressionResult ─────────────────────────────────────────────────────────

class TestCompressionResult:
    def test_ratio(self):
        r = CompressionResult(bytes_uncompressed=1000, bytes_compressed=500)
        assert r.ratio == 0.5

    def test_ratio_zero_uncompressed(self):
        r = CompressionResult(bytes_uncompressed=0, bytes_compressed=0)
        assert r.ratio == 0.0

    def test_savings_pct(self):
        r = CompressionResult(bytes_uncompressed=1000, bytes_compressed=200)
        assert r.savings_pct == pytest.approx(80.0)

    def test_savings_pct_no_compression(self):
        r = CompressionResult(bytes_uncompressed=1000, bytes_compressed=1000)
        assert r.savings_pct == pytest.approx(0.0)

    def test_defaults(self):
        r = CompressionResult()
        assert r.bytes_uncompressed == 0
        assert r.bytes_compressed == 0
        assert r.sha256 == ""
        assert r.elapsed_seconds == 0.0


# ── DownloadResult ────────────────────────────────────────────────────────────

class TestDownloadResult:
    def test_defaults(self):
        r = DownloadResult()
        assert r.success is False
        assert r.dest_path == ""
        assert r.sha256_match is False
        assert r.error == ""


# ── compress_bytes / decompress_bytes ─────────────────────────────────────────

class TestCompressDecompressBytes:
    def test_roundtrip(self):
        data = b"Hello, compressed world! " * 100
        compressed, result = compress_bytes(data)
        assert result.bytes_uncompressed == len(data)
        assert result.bytes_compressed < len(data)
        assert len(result.sha256) == 64

        decompressed, result2 = decompress_bytes(compressed)
        assert decompressed == data
        assert result2.sha256 == result.sha256

    def test_small_data(self):
        data = b"tiny"
        compressed, _ = compress_bytes(data)
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == data

    def test_empty_data(self):
        data = b""
        compressed, _ = compress_bytes(data)
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == data

    def test_binary_data(self):
        data = bytes(range(256)) * 100
        compressed, r1 = compress_bytes(data)
        decompressed, r2 = decompress_bytes(compressed)
        assert decompressed == data
        assert r1.sha256 == r2.sha256

    def test_repetitive_data_compresses_well(self):
        data = b"A" * 10000
        compressed, result = compress_bytes(data)
        assert result.ratio < 0.01  # >99% compression

    def test_random_data_compresses_less(self):
        import random
        random.seed(42)
        data = bytes(random.getrandbits(8) for _ in range(10000))
        compressed, result = compress_bytes(data)
        assert result.ratio > 0.5  # random data doesn't compress well


# ── compress_stream / decompress_stream ───────────────────────────────────────

class TestCompressDecompressStream:
    def test_roundtrip_no_header(self):
        data = b"Stream test data " * 50
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        r1 = compress_stream(source, compressed, include_header=False)

        compressed.seek(0)
        decompressed = io.BytesIO()
        r2 = decompress_stream(compressed, decompressed, verify_header=False)

        assert decompressed.getvalue() == data
        assert r1.sha256 == r2.sha256

    def test_roundtrip_with_header(self):
        data = b"Header test data " * 50
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        r1 = compress_stream(source, compressed, include_header=True)

        compressed.seek(0)
        decompressed = io.BytesIO()
        r2 = decompress_stream(compressed, decompressed, verify_header=True)

        assert decompressed.getvalue() == data
        assert r1.sha256 == r2.sha256

    def test_header_format(self):
        data = b"test data for header"
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        compress_stream(source, compressed, include_header=True)

        compressed.seek(0)
        header = compressed.read(HEADER_SIZE)
        assert header[0:4] == MAGIC
        uncompressed_size = struct.unpack(">Q", header[4:12])[0]
        assert uncompressed_size == len(data)
        expected_sha = hashlib.sha256(data).digest()
        assert header[12:44] == expected_sha

    def test_compression_ratio(self):
        data = b"compressible " * 1000
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        result = compress_stream(source, compressed, include_header=False)
        assert result.ratio < 0.5

    def test_on_progress_callback(self):
        data = b"progress test " * 100
        progress_calls = []

        def on_progress(bytes_read, total):
            progress_calls.append((bytes_read, total))

        source = io.BytesIO(data)
        compressed = io.BytesIO()
        compress_stream(source, compressed, include_header=False, on_progress=on_progress)
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == len(data)

    def test_no_header_no_seek(self):
        data = b"non-seekable test"
        source = io.BytesIO(data)
        # Wrap to remove seek
        class NonSeekable:
            def __init__(self, data):
                self._data = data
                self._pos = 0
            def read(self, n=-1):
                chunk = self._data[self._pos:self._pos + n]
                self._pos += len(chunk)
                return chunk

        non_seek = NonSeekable(data)
        compressed = io.BytesIO()
        result = compress_stream(non_seek, compressed, include_header=True)
        assert result.bytes_uncompressed == len(data)


# ── compress_file / decompress_file ───────────────────────────────────────────

class TestCompressDecompressFile:
    def test_roundtrip(self, tmp_path):
        src = tmp_path / "source.bin"
        src.write_bytes(b"File content " * 200)
        compressed_path = tmp_path / "source.bin.gz"
        decompressed_path = tmp_path / "source_restored.bin"

        r1 = compress_file(src, compressed_path)
        assert compressed_path.exists()
        assert r1.bytes_uncompressed == src.stat().st_size

        r2 = decompress_file(compressed_path, decompressed_path)
        assert decompressed_path.exists()
        assert decompressed_path.read_bytes() == src.read_bytes()
        assert r1.sha256 == r2.sha256

    def test_compressed_file_smaller(self, tmp_path):
        data = b"compressible data " * 500
        src = tmp_path / "big.bin"
        src.write_bytes(data)
        compressed_path = tmp_path / "big.bin.gz"

        compress_file(src, compressed_path)
        assert compressed_path.stat().st_size < src.stat().st_size

    def test_verify_integrity(self, tmp_path):
        data = b"integrity check " * 100
        src = tmp_path / "src.bin"
        src.write_bytes(data)
        compressed_path = tmp_path / "src.bin.gz"
        decompressed_path = tmp_path / "dst.bin"

        compress_file(src, compressed_path)
        decompress_file(compressed_path, decompressed_path)
        assert decompressed_path.read_bytes() == data


# ── peek_compressed_header ────────────────────────────────────────────────────

class TestPeekCompressedHeader:
    def test_valid_header(self):
        data = b"peek test data"
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        compress_stream(source, compressed, include_header=True)
        compressed.seek(0)
        info = peek_compressed_header(compressed)
        assert info is not None
        assert info["magic"] == MAGIC
        assert info["uncompressed_size"] == len(data)
        assert len(info["sha256"]) == 64

    def test_no_header(self):
        data = b"no header here"
        compressed = io.BytesIO()
        compress_stream(io.BytesIO(data), compressed, include_header=False)
        compressed.seek(0)
        info = peek_compressed_header(compressed)
        assert info is None

    def test_invalid_magic(self):
        source = io.BytesIO(b"bad magic")
        bad = io.BytesIO(b"XXXX" + b"\x00" * 40)
        info = peek_compressed_header(bad)
        assert info is None

    def test_empty_stream(self):
        info = peek_compressed_header(io.BytesIO(b""))
        assert info is None

    def test_short_stream(self):
        info = peek_compressed_header(io.BytesIO(b"SGZ"))
        assert info is None


# ── compressed_file_iterator ──────────────────────────────────────────────────

class TestCompressedFileIterator:
    def test_iterate_compressed(self, tmp_path):
        data = b"iterator test " * 100
        src = tmp_path / "iter.bin"
        src.write_bytes(data)

        chunks = list(compressed_file_iterator(src))
        result = b"".join(chunks)
        # Result is compressed with header, so verify by decompressing
        assert result[:4] == MAGIC
        decompressed = io.BytesIO()
        source = io.BytesIO(result)
        decompress_stream(source, decompressed, verify_header=True)
        assert decompressed.getvalue() == data

    def test_iterate_small_chunks(self, tmp_path):
        data = b"chunk test"
        src = tmp_path / "chunk.bin"
        src.write_bytes(data)

        chunks = list(compressed_file_iterator(src, chunk_size=4))
        result = b"".join(chunks)
        assert result[:4] == MAGIC
        decompressed = io.BytesIO()
        decompress_stream(io.BytesIO(result), decompressed, verify_header=True)
        assert decompressed.getvalue() == data

    def test_iterate_empty_file(self, tmp_path):
        src = tmp_path / "empty.bin"
        src.write_bytes(b"")
        chunks = list(compressed_file_iterator(src))
        result = b"".join(chunks)
        assert result[:4] == MAGIC
        decompressed = io.BytesIO()
        decompress_stream(io.BytesIO(result), decompressed, verify_header=True)
        assert decompressed.getvalue() == b""


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_corrupt_gzip_data(self):
        corrupt = io.BytesIO(b"SGZ1" + b"\x00" * 40 + b"NOT_VALID_GZIP")
        decompressed = io.BytesIO()
        with pytest.raises((ValueError, OSError)):
            decompress_stream(corrupt, decompressed, verify_header=True)

    def test_header_checksum_mismatch(self, tmp_path):
        data = b"checksum mismatch test"
        source = io.BytesIO(data)
        compressed = io.BytesIO()
        compress_stream(source, compressed, include_header=True)

        # Corrupt the SHA-256 in the header
        compressed.seek(12)
        compressed.write(b"\xff" * 32)
        compressed.seek(0)

        decompressed = io.BytesIO()
        with pytest.raises(ValueError, match="SHA-256"):
            decompress_stream(compressed, decompressed, verify_header=True)

    def test_large_data_roundtrip(self):
        data = os.urandom(1024 * 1024)  # 1MB random
        compressed, r1 = compress_bytes(data)
        decompressed, r2 = decompress_bytes(compressed)
        assert decompressed == data
        assert r1.sha256 == r2.sha256
