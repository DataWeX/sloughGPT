"""Tests for downcraft.download.compress — LZ4 compression/decompression."""

import io
import sys
from pathlib import Path

import pytest

# Ensure local conftest is importable
sys.path.insert(0, str(Path(__file__).parent))

from downcraft.download.compress import (
    MAGIC,
    HEADER_SIZE,
    compress_bytes,
    compress_file,
    compress_stream,
    decompress_bytes,
    decompress_file,
    decompress_stream,
    peek_compressed_header,
    is_compressed_file,
    auto_decompress,
    CompressionResult,
)


# ---------------------------------------------------------------------------
# compress_bytes / decompress_bytes
# ---------------------------------------------------------------------------


class TestCompressDecompressBytes:
    def test_roundtrip_small(self):
        data = b"hello world"
        compressed, c_result = compress_bytes(data)
        assert len(compressed) > 0
        assert c_result.bytes_uncompressed == len(data)

        decompressed, d_result = decompress_bytes(compressed)
        assert decompressed == data
        assert d_result.bytes_uncompressed == len(data)

    def test_roundtrip_large(self):
        data = b"x" * 100_000
        compressed, c_result = compress_bytes(data)
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == data
        assert c_result.savings_pct > 90  # highly compressible

    def test_roundtrip_random(self):
        import os
        data = os.urandom(10_000)
        compressed, _ = compress_bytes(data)
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == data

    def test_empty_bytes(self):
        data = b""
        compressed, _ = compress_bytes(data)
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == data

    def test_compression_level(self):
        data = b"test data " * 1000
        low, _ = compress_bytes(data, compression_level=1)
        high, _ = compress_bytes(data, compression_level=16)
        # Higher compression should be smaller or equal
        assert len(high) <= len(low)


# ---------------------------------------------------------------------------
# compress_file / decompress_file
# ---------------------------------------------------------------------------


class TestCompressDecompressFile:
    def test_roundtrip(self, tmp_path):
        src = tmp_path / "input.bin"
        compressed = tmp_path / "output.lz4"
        decompressed = tmp_path / "restored.bin"

        data = b"file content " * 500
        src.write_bytes(data)

        c_result = compress_file(src, compressed)
        assert compressed.exists()
        assert c_result.bytes_uncompressed == len(data)

        d_result = decompress_file(compressed, decompressed)
        assert decompressed.exists()
        assert decompressed.read_bytes() == data

    def test_compressed_is_smaller(self, tmp_path):
        src = tmp_path / "input.bin"
        compressed = tmp_path / "output.lz4"

        data = b"compressible data " * 1000
        src.write_bytes(data)
        compress_file(src, compressed)

        assert compressed.stat().st_size < src.stat().st_size

    def test_nested_directories(self, tmp_path):
        src = tmp_path / "input.bin"
        compressed = tmp_path / "sub" / "dir" / "output.lz4"
        decompressed = tmp_path / "other" / "dir" / "restored.bin"

        data = b"nested path test"
        src.write_bytes(data)

        compress_file(src, compressed)
        assert compressed.exists()

        decompress_file(compressed, decompressed)
        assert decompressed.read_bytes() == data


# ---------------------------------------------------------------------------
# compress_stream / decompress_stream
# ---------------------------------------------------------------------------


class TestCompressDecompressStream:
    def test_roundtrip_with_header(self):
        data = b"stream test data " * 200
        src = io.BytesIO(data)
        dst = io.BytesIO()

        c_result = compress_stream(src, dst, include_header=True)
        assert c_result.bytes_uncompressed == len(data)

        compressed_data = dst.getvalue()
        assert compressed_data[:4] == MAGIC

        src2 = io.BytesIO(compressed_data)
        dst2 = io.BytesIO()
        d_result = decompress_stream(src2, dst2, verify_header=True)
        assert dst2.getvalue() == data

    def test_roundtrip_without_header(self):
        data = b"no header test"
        src = io.BytesIO(data)
        dst = io.BytesIO()

        compress_stream(src, dst, include_header=False)
        compressed_data = dst.getvalue()
        assert compressed_data[:4] != MAGIC

        src2 = io.BytesIO(compressed_data)
        dst2 = io.BytesIO()
        decompress_stream(src2, dst2, verify_header=False)
        assert dst2.getvalue() == data

    def test_invalid_magic_raises(self):
        # Data that's too short to have a valid header
        data = b"short"
        src = io.BytesIO(data)
        dst = io.BytesIO()

        with pytest.raises(ValueError, match="Header too short"):
            decompress_stream(src, dst, verify_header=True)


# ---------------------------------------------------------------------------
# peek_compressed_header
# ---------------------------------------------------------------------------


class TestPeekCompressedHeader:
    def test_valid_header(self):
        data = b"header test " * 100
        src = io.BytesIO(data)
        dst = io.BytesIO()
        compress_stream(src, dst, include_header=True)

        src2 = io.BytesIO(dst.getvalue())
        info = peek_compressed_header(src2)
        assert info is not None
        assert info["magic"] == MAGIC
        assert info["uncompressed_size"] == len(data)
        assert len(info["sha256"]) == 64

    def test_not_compressed(self):
        src = io.BytesIO(b"not compressed data")
        info = peek_compressed_header(src)
        assert info is None

    def test_too_short(self):
        src = io.BytesIO(b"short")
        info = peek_compressed_header(src)
        assert info is None


# ---------------------------------------------------------------------------
# is_compressed_file
# ---------------------------------------------------------------------------


class TestIsCompressedFile:
    def test_compressed_file(self, tmp_path):
        fpath = tmp_path / "test.lz4"
        data = b"test data"
        src = io.BytesIO(data)
        dst = io.BytesIO()
        compress_stream(src, dst, include_header=True)
        fpath.write_bytes(dst.getvalue())

        assert is_compressed_file(fpath) is True

    def test_not_compressed_file(self, tmp_path):
        fpath = tmp_path / "test.bin"
        fpath.write_bytes(b"not compressed")
        assert is_compressed_file(fpath) is False

    def test_nonexistent_file(self, tmp_path):
        fpath = tmp_path / "nonexistent.lz4"
        assert is_compressed_file(fpath) is False


# ---------------------------------------------------------------------------
# auto_decompress
# ---------------------------------------------------------------------------


class TestAutoDecompress:
    def test_compressed_file(self, tmp_path):
        src = tmp_path / "input.bin"
        compressed = tmp_path / "compressed.lz4"
        decompressed = tmp_path / "output.bin"

        data = b"auto decompress test"
        src.write_bytes(data)

        # Create compressed file
        compress_file(src, compressed)

        # Auto-decompress
        result = auto_decompress(compressed, decompressed)
        assert decompressed.read_bytes() == data
        assert result.bytes_uncompressed == len(data)

    def test_uncompressed_file(self, tmp_path):
        src = tmp_path / "input.bin"
        decompressed = tmp_path / "output.bin"

        data = b"already uncompressed"
        src.write_bytes(data)

        result = auto_decompress(src, decompressed)
        assert decompressed.read_bytes() == data
        assert result.bytes_uncompressed == len(data)


# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------


class TestCompressionResult:
    def test_ratio(self):
        result = CompressionResult(bytes_uncompressed=1000, bytes_compressed=500)
        assert result.ratio == 0.5

    def test_savings_pct(self):
        result = CompressionResult(bytes_uncompressed=1000, bytes_compressed=500)
        assert result.savings_pct == 50.0

    def test_zero_uncompressed(self):
        result = CompressionResult(bytes_uncompressed=0, bytes_compressed=0)
        assert result.ratio == 0.0
        assert result.savings_pct == 100.0
