"""Tests for compressed_transfer module — streaming compression/decompression."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import struct
import time
from pathlib import Path

import pytest

from domains.infrastructure.compressed_transfer import (
    MAGIC,
    HEADER_SIZE,
    CHUNK_SIZE,
    CompressedDownloader,
    CompressedFileServer,
    CompressionResult,
    DownloadResult,
    compress_bytes,
    compress_file,
    compress_stream,
    decompress_bytes,
    decompress_file,
    decompress_stream,
    peek_compressed_header,
    compressed_file_iterator,
)


# ── Test data ──


SMALL_TEXT = b"Hello, world! This is a test of streaming compression."
MEDIUM_TEXT = b"A" * 100_000  # 100 KB of repeated chars (high compressibility)
RANDOM_DATA = os.urandom(10_000)  # 10 KB random (low compressibility)


# ── Header tests ──


class TestHeader:
    def test_magic_bytes(self):
        assert MAGIC == b"SGZ1"

    def test_header_size(self):
        assert HEADER_SIZE == 44  # 4 + 8 + 32


# ── compress_bytes / decompress_bytes (in-memory roundtrip) ──


class TestInMemoryRoundtrip:
    def test_small_text(self):
        compressed, c_result = compress_bytes(SMALL_TEXT)
        assert c_result.bytes_uncompressed == len(SMALL_TEXT)
        assert c_result.sha256 == hashlib.sha256(SMALL_TEXT).hexdigest()

        decompressed, d_result = decompress_bytes(compressed)
        assert decompressed == SMALL_TEXT
        assert d_result.sha256 == c_result.sha256

    def test_medium_text(self):
        compressed, c_result = compress_bytes(MEDIUM_TEXT)
        decompressed, d_result = decompress_bytes(compressed)
        assert decompressed == MEDIUM_TEXT
        assert len(compressed) < len(MEDIUM_TEXT)  # highly compressible

    def test_random_data(self):
        compressed, c_result = compress_bytes(RANDOM_DATA)
        decompressed, d_result = decompress_bytes(compressed)
        assert decompressed == RANDOM_DATA

    def test_empty_data(self):
        compressed, _ = compress_bytes(b"")
        decompressed, _ = decompress_bytes(compressed)
        assert decompressed == b""

    def test_compression_ratio_reported(self):
        _, result = compress_bytes(MEDIUM_TEXT)
        assert result.ratio < 0.1  # should compress very well
        assert result.savings_pct > 90


# ── compress_stream / decompress_stream (streaming roundtrip) ──


class TestStreamingRoundtrip:
    def test_stream_roundtrip_with_header(self):
        src = io.BytesIO(SMALL_TEXT)
        dst = io.BytesIO()
        c_result = compress_stream(src, dst, include_header=True)
        assert c_result.bytes_uncompressed == len(SMALL_TEXT)

        dst.seek(0)
        out = io.BytesIO()
        d_result = decompress_stream(dst, out, verify_header=True)
        assert out.getvalue() == SMALL_TEXT
        assert d_result.sha256 == c_result.sha256

    def test_stream_roundtrip_without_header(self):
        src = io.BytesIO(MEDIUM_TEXT)
        dst = io.BytesIO()
        c_result = compress_stream(src, dst, include_header=False)

        dst.seek(0)
        out = io.BytesIO()
        d_result = decompress_stream(dst, out, verify_header=False)
        assert out.getvalue() == MEDIUM_TEXT

    def test_progress_callback(self):
        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int):
            progress_calls.append((done, total))

        src = io.BytesIO(MEDIUM_TEXT)
        dst = io.BytesIO()
        compress_stream(src, dst, on_progress=on_progress)
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == len(MEDIUM_TEXT)

    def test_streaming_decompress_progress(self):
        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int):
            progress_calls.append((done, total))

        src = io.BytesIO(SMALL_TEXT)
        dst = io.BytesIO()
        compress_stream(src, dst, include_header=True)

        dst.seek(0)
        out = io.BytesIO()
        decompress_stream(dst, out, verify_header=True, on_progress=on_progress)
        assert len(progress_calls) > 0


# ── compress_file / decompress_file (file-based roundtrip) ──


class TestFileRoundtrip:
    def test_file_roundtrip(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        compressed_path = tmp_path / "output.sgz"
        decompressed_path = tmp_path / "output.bin"

        src_path.write_bytes(MEDIUM_TEXT)
        c_result = compress_file(src_path, compressed_path)
        assert compressed_path.exists()
        assert c_result.bytes_uncompressed == len(MEDIUM_TEXT)
        assert c_result.sha256 == hashlib.sha256(MEDIUM_TEXT).hexdigest()

        d_result = decompress_file(compressed_path, decompressed_path)
        assert decompressed_path.read_bytes() == MEDIUM_TEXT
        assert d_result.sha256 == c_result.sha256

    def test_compressed_file_smaller(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        compressed_path = tmp_path / "output.sgz"

        src_path.write_bytes(MEDIUM_TEXT)
        compress_file(src_path, compressed_path)
        assert compressed_path.stat().st_size < src_path.stat().st_size

    def test_creates_parent_dirs(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        compressed_path = tmp_path / "sub" / "dir" / "output.sgz"
        decompressed_path = tmp_path / "sub2" / "dir2" / "output.bin"

        src_path.write_bytes(SMALL_TEXT)
        compress_file(src_path, compressed_path)
        assert compressed_path.exists()

        decompress_file(compressed_path, decompressed_path)
        assert decompressed_path.read_bytes() == SMALL_TEXT

    def test_progress_callback_file(self, tmp_path: Path):
        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int):
            progress_calls.append((done, total))

        src_path = tmp_path / "input.bin"
        src_path.write_bytes(MEDIUM_TEXT)
        compress_file(src_path, tmp_path / "out.sgz", on_progress=on_progress)
        assert len(progress_calls) > 0


# ── SGZ1 header format tests ──


class TestSGZ1Header:
    def test_header_written_correctly(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        compressed_path = tmp_path / "output.sgz"

        src_path.write_bytes(SMALL_TEXT)
        compress_file(src_path, compressed_path)

        data = compressed_path.read_bytes()
        assert data[0:4] == MAGIC
        uncompressed_size = struct.unpack(">Q", data[4:12])[0]
        assert uncompressed_size == len(SMALL_TEXT)
        expected_sha = hashlib.sha256(SMALL_TEXT).hexdigest()
        actual_sha = data[12:44].hex()
        assert actual_sha == expected_sha

    def test_header_with_seekable_stream(self):
        src = io.BytesIO(MEDIUM_TEXT)
        dst = io.BytesIO()
        compress_stream(src, dst, include_header=True)

        dst.seek(0)
        header = dst.read(HEADER_SIZE)
        assert header[0:4] == MAGIC
        size = struct.unpack(">Q", header[4:12])[0]
        assert size == len(MEDIUM_TEXT)


# ── peek_compressed_header ──


class TestPeekHeader:
    def test_peek_sgz1_header(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        compressed_path = tmp_path / "output.sgz"

        src_path.write_bytes(SMALL_TEXT)
        compress_file(src_path, compressed_path)

        with open(compressed_path, "rb") as f:
            info = peek_compressed_header(f)
            assert info is not None
            assert info["magic"] == MAGIC
            assert info["uncompressed_size"] == len(SMALL_TEXT)
            assert info["sha256"] == hashlib.sha256(SMALL_TEXT).hexdigest()

            # File position should be restored
            assert f.tell() == 0

    def test_peek_non_sgz_returns_none(self):
        src = io.BytesIO(b"not a sgz file")
        info = peek_compressed_header(src)
        assert info is None

    def test_peek_empty_returns_none(self):
        src = io.BytesIO(b"")
        info = peek_compressed_header(src)
        assert info is None


# ── compressed_file_iterator (streaming server helper) ──


class TestCompressedFileIterator:
    def test_iterator_yields_valid_sgz1(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        src_path.write_bytes(SMALL_TEXT)

        chunks = list(compressed_file_iterator(src_path))
        assert len(chunks) > 0

        # First chunk should start with SGZ1 header
        data = b"".join(chunks)
        assert data[0:4] == MAGIC

    def test_iterator_decompressible(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        src_path.write_bytes(MEDIUM_TEXT)

        chunks = list(compressed_file_iterator(src_path))
        data = b"".join(chunks)

        # Should decompress correctly
        decompressed, _ = decompress_bytes(data[HEADER_SIZE:])
        assert decompressed == MEDIUM_TEXT

    def test_iterator_chunk_size(self, tmp_path: Path):
        src_path = tmp_path / "input.bin"
        src_path.write_bytes(MEDIUM_TEXT)

        chunks = list(compressed_file_iterator(src_path, chunk_size=1024))
        # All chunks except last should be <= 1024
        for chunk in chunks[:-1]:
            assert len(chunk) <= 1024


# ── CompressedDownloader (client-side) ──


class TestCompressedDownloader:
    def test_download_from_stream(self, tmp_path: Path):
        # Create compressed data in memory
        src = io.BytesIO(MEDIUM_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        dest = tmp_path / "downloaded.bin"
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(compressed, dest)

        assert result.success is True
        assert dest.read_bytes() == MEDIUM_TEXT
        assert result.sha256_match is True

    def test_download_with_expected_sha256(self, tmp_path: Path):
        src = io.BytesIO(SMALL_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        expected_sha = hashlib.sha256(SMALL_TEXT).hexdigest()
        dest = tmp_path / "downloaded.bin"
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(compressed, dest, expected_sha256=expected_sha)

        assert result.success is True
        assert result.sha256_match is True

    def test_download_bad_sha256_fails(self, tmp_path: Path):
        src = io.BytesIO(SMALL_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        dest = tmp_path / "downloaded.bin"
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(
            compressed, dest, expected_sha256="0000000000000000000000000000000000000000000000000000000000000000"
        )

        # Download succeeds but SHA-256 doesn't match
        assert result.success is True
        assert result.sha256_match is False

    def test_download_creates_parent_dirs(self, tmp_path: Path):
        src = io.BytesIO(SMALL_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        dest = tmp_path / "a" / "b" / "c" / "file.bin"
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(compressed, dest)

        assert result.success is True
        assert dest.exists()


# ── CompressedFileServer (server-side) ──


class TestCompressedFileServer:
    def test_serve_returns_metadata(self, tmp_path: Path):
        src_path = tmp_path / "model.bin"
        src_path.write_bytes(MEDIUM_TEXT)

        server = CompressedFileServer()
        info = server.serve(src_path)

        assert info["size"] == len(MEDIUM_TEXT)
        assert info["content_type"] == "application/x-sgzs"
        assert "gzip" in info["headers"]["Content-Encoding"]

    def test_serve_iterator_decompresses(self, tmp_path: Path):
        src_path = tmp_path / "model.bin"
        src_path.write_bytes(MEDIUM_TEXT)

        server = CompressedFileServer()
        info = server.serve(src_path)

        # Collect compressed data from iterator
        compressed = b"".join(info["iterator"])

        # Decompress and verify
        decompressed, _ = decompress_bytes(compressed[HEADER_SIZE:])
        assert decompressed == MEDIUM_TEXT

    def test_serve_nonexistent_raises(self, tmp_path: Path):
        server = CompressedFileServer()
        with pytest.raises(FileNotFoundError):
            server.serve(tmp_path / "nonexistent.bin")

    def test_serve_range(self, tmp_path: Path):
        src_path = tmp_path / "model.bin"
        src_path.write_bytes(MEDIUM_TEXT)

        # Compress the file first
        compressed_path = tmp_path / "model.sgz"
        compress_file(src_path, compressed_path)

        server = CompressedFileServer()
        range_result = server.serve_range(compressed_path, 0, 5)

        assert len(range_result["data"]) == 6
        assert range_result["data"] == MEDIUM_TEXT[:6]


# ── Integrity verification ──


class TestIntegrity:
    def test_corrupted_compressed_data_detected(self, tmp_path: Path):
        src = io.BytesIO(SMALL_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        # Corrupt the compressed data (after header)
        data = bytearray(compressed.read())
        # Flip some bits in the middle of the compressed payload
        mid = len(data) // 2
        data[mid : mid + 10] = b"\x00" * 10

        corrupted = io.BytesIO(bytes(data))
        dest = tmp_path / "output.bin"
        downloader = CompressedDownloader()

        # Should either fail or produce different checksum
        result = downloader.download_from_stream(corrupted, dest)
        if result.success:
            # If it decompressed (unlikely with corruption), checksum won't match
            assert result.sha256 != hashlib.sha256(SMALL_TEXT).hexdigest()

    def test_truncated_header_detected(self):
        src = io.BytesIO(SMALL_TEXT)
        compressed = io.BytesIO()
        compress_stream(src, compressed, include_header=True)
        compressed.seek(0)

        # Read only part of header
        truncated = io.BytesIO(compressed.read(20))
        dest = Path("/tmp/test_output.bin")
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(truncated, dest)
        assert result.success is False
        assert "Header too short" in result.error

    def test_wrong_magic_detected(self):
        # Write fake header
        fake_data = b"FAKE" + b"\x00" * 40 + b"compressed stuff"
        src = io.BytesIO(fake_data)
        dest = Path("/tmp/test_output.bin")
        downloader = CompressedDownloader()
        result = downloader.download_from_stream(src, dest)
        assert result.success is False
        assert "Invalid magic" in result.error


# ── Performance characteristics ──


class TestPerformance:
    def test_compressibility_ratio(self):
        _, result = compress_bytes(MEDIUM_TEXT)
        assert result.savings_pct > 90  # repeated chars compress extremely well

    def test_random_data_ratio(self):
        _, result = compress_bytes(RANDOM_DATA)
        # Random data shouldn't compress much
        assert result.savings_pct < 50

    def test_large_file_streaming(self, tmp_path: Path):
        """Verify streaming works for larger data without excessive memory."""
        large_data = b"x" * 1_000_000  # 1 MB
        src_path = tmp_path / "large.bin"
        src_path.write_bytes(large_data)

        compressed_path = tmp_path / "large.sgz"
        decompressed_path = tmp_path / "large_out.bin"

        compress_file(src_path, compressed_path)
        decompress_file(compressed_path, decompressed_path)

        assert decompressed_path.read_bytes() == large_data
        # Should compress well
        assert compressed_path.stat().st_size < 100_000
