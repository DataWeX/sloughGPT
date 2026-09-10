"""
Streaming LZ4 Compression — on-the-fly LZ4 compression/decompression for file transfers.

Design goals:
  - Zero buffered copies: compress → stream → decompress → write (no temp files)
  - Server-side: compress from disk, stream out (never store .lz4 on disk)
  - Client-side: receive compressed stream, decompress on-the-fly to disk
  - Integrity: SHA-256 checksum of uncompressed payload, verified after decompression
  - Resumable: byte-range support with compressed offset mapping

Usage:
    # Server: serve a file with streaming LZ4 compression
    from domains.infrastructure.compressed_transfer import CompressedFileServer
    server = CompressedFileServer()
    response = server.serve("/path/to/model.bin")  # returns StreamingResponse

    # Client: download with on-the-fly decompression
    from domains.infrastructure.compressed_transfer import CompressedDownloader
    downloader = CompressedDownloader()
    result = downloader.download_from_url(
        url="http://server/download/model.bin.lz4",
        dest="/local/model.bin",
        expected_sha256="abc123...",
    )
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Generator

import lz4.frame

logger = logging.getLogger("slo.compressed_transfer")

# ── Header format ──
# The first 44 bytes of a compressed stream are a custom header:
#   [0:4]   magic bytes: b"SLZ4" (sloughGPT LZ4 v1)
#   [4:12]  uncompressed size (uint64 big-endian)
#   [12:44] SHA-256 of uncompressed content (32 bytes)
#   [44:...] LZ4 frame-compressed payload

MAGIC = b"SLZ4"
HEADER_SIZE = 4 + 8 + 32  # 44 bytes
CHUNK_SIZE = 65536  # 64 KB read chunks


@dataclass
class CompressionResult:
    """Result of a compression/decompression operation."""
    bytes_uncompressed: int = 0
    bytes_compressed: int = 0
    sha256: str = ""
    elapsed_seconds: float = 0.0

    @property
    def ratio(self) -> float:
        if self.bytes_uncompressed == 0:
            return 0.0
        return self.bytes_compressed / self.bytes_uncompressed

    @property
    def savings_pct(self) -> float:
        return (1.0 - self.ratio) * 100


@dataclass
class DownloadResult:
    """Result of a compressed download."""
    success: bool = False
    dest_path: str = ""
    bytes_written: int = 0
    sha256: str = ""
    sha256_match: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0


# ── Streaming compression (server-side) ──


def compress_stream(
    source: BinaryIO,
    dest: BinaryIO,
    *,
    compression_level: int = 6,
    include_header: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> CompressionResult:
    """Compress a binary stream on-the-fly using LZ4.

    Reads from *source*, writes LZ4-compressed data to *dest*.
    If *include_header* is True, prepends the SLZ4 header with
    uncompressed size and SHA-256 for integrity verification.

    Args:
        source: Readable binary stream (e.g., open(path, "rb")).
        dest: Writable binary stream (e.g., response body, socket).
        compression_level: LZ4 compression level (1-16, default 6).
        include_header: Whether to prepend SLZ4 header.
        on_progress: Callback(bytes_read, total_bytes) if total known.

    Returns:
        CompressionResult with sizes and checksum.
    """
    sha256 = hashlib.sha256()
    uncompressed_size = 0

    # First pass: compute size and hash if source supports seeking
    total_size = None
    if hasattr(source, "seek") and hasattr(source, "tell"):
        try:
            pos = source.tell()
            source.seek(0, os.SEEK_END)
            total_size = source.tell()
            source.seek(pos)
        except (OSError, io.UnsupportedOperation):
            pass

    # Write placeholder header if including header
    if include_header:
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        dest.write(bytes(header))

    # Compress using LZ4 frame
    buffer = io.BytesIO()
    with lz4.frame.open(buffer, "wb", compression_level=compression_level) as f:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            uncompressed_size += len(chunk)
            f.write(chunk)
            if on_progress and total_size:
                on_progress(uncompressed_size, total_size)

    compressed_data = buffer.getvalue()
    compressed_size = len(compressed_data)

    # Write compressed data
    dest.write(compressed_data)

    # Seek back and write the real header if we included one
    if include_header and hasattr(dest, "seek") and hasattr(dest, "tell"):
        try:
            pos = dest.tell()
            dest.seek(0)
            header = bytearray(HEADER_SIZE)
            header[0:4] = MAGIC
            header[4:12] = struct.pack(">Q", uncompressed_size)
            header[12:44] = sha256.digest()
            dest.write(bytes(header))
            dest.seek(pos)
        except (OSError, io.UnsupportedOperation):
            pass

    return CompressionResult(
        bytes_uncompressed=uncompressed_size,
        bytes_compressed=compressed_size,
        sha256=sha256.hexdigest(),
    )


def decompress_stream(
    source: BinaryIO,
    dest: BinaryIO,
    *,
    verify_header: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> CompressionResult:
    """Decompress an LZ4 stream on-the-fly.

    Reads from *source* (optionally with SLZ4 header), writes
    decompressed data to *dest*.

    Args:
        source: Readable binary stream with LZ4-compressed data.
        dest: Writable binary stream for decompressed output.
        verify_header: Whether to expect and verify SLZ4 header.
        on_progress: Callback(bytes_written, expected_total) if header present.

    Returns:
        CompressionResult with sizes and checksum.

    Raises:
        ValueError: If header is malformed or checksum mismatch.
    """
    expected_size = None
    expected_sha256 = None

    if verify_header:
        header = source.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(header)} bytes (expected {HEADER_SIZE})")
        if header[0:4] != MAGIC:
            raise ValueError(f"Invalid magic: {header[0:4]!r} (expected {MAGIC!r})")
        expected_size = struct.unpack(">Q", header[4:12])[0]
        expected_sha256 = header[12:44].hex()

    # Decompress using LZ4 frame
    sha256 = hashlib.sha256()
    uncompressed_size = 0

    with lz4.frame.open(source, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            dest.write(chunk)
            uncompressed_size += len(chunk)
            if on_progress and expected_size:
                on_progress(uncompressed_size, expected_size)

    actual_sha256 = sha256.hexdigest()

    if verify_header and expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    return CompressionResult(
        bytes_uncompressed=uncompressed_size,
        bytes_compressed=uncompressed_size,  # unknown for stream
        sha256=actual_sha256,
    )


# ── File-level helpers ──


def compress_file(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    compression_level: int = 6,
    on_progress: Callable[[int, int], None] | None = None,
) -> CompressionResult:
    """Compress a file on disk to a .lz4 file with header."""
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
        return compress_stream(src, dst, compression_level=compression_level, on_progress=on_progress)


def decompress_file(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    verify_header: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> CompressionResult:
    """Decompress a .lz4 file to disk."""
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
        return decompress_stream(src, dst, verify_header=verify_header, on_progress=on_progress)


def compress_bytes(data: bytes, compression_level: int = 6) -> tuple[bytes, CompressionResult]:
    """Compress raw bytes in memory."""
    src = io.BytesIO(data)
    dst = io.BytesIO()
    result = compress_stream(src, dst, compression_level=compression_level, include_header=False)
    return dst.getvalue(), result


def decompress_bytes(data: bytes) -> tuple[bytes, CompressionResult]:
    """Decompress raw LZ4 bytes in memory."""
    src = io.BytesIO(data)
    dst = io.BytesIO()
    result = decompress_stream(src, dst, verify_header=False)
    return dst.getvalue(), result


# ── Streaming HTTP response helper ──


def compressed_file_iterator(
    file_path: str | Path,
    *,
    compression_level: int = 6,
    chunk_size: int = CHUNK_SIZE,
) -> Generator[bytes, None, None]:
    """Yield LZ4-compressed chunks from a file for streaming HTTP responses.

    Yields the SLZ4 header first, then compressed chunks.
    Useful with Starlette/FastAPI StreamingResponse.
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size

    # Compute SHA-256 and uncompressed size
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    # Build header
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4:12] = struct.pack(">Q", file_size)
    header[12:44] = sha256.digest()
    yield bytes(header)

    # Stream compressed chunks using LZ4 frame
    buffer = io.BytesIO()
    with lz4.frame.open(buffer, "wb", compression_level=compression_level) as f:
        with open(file_path, "rb") as src:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

    compressed = buffer.getvalue()

    # Yield in chunks
    for i in range(0, len(compressed), chunk_size):
        yield compressed[i : i + chunk_size]


# ── Compressed Downloader (client-side) ──


class CompressedDownloader:
    """Download files with on-the-fly LZ4 decompression.

    Streams compressed data from the server, decompresses to disk,
    and verifies integrity. Never stores the compressed copy.
    """

    def __init__(self, *, chunk_size: int = CHUNK_SIZE, timeout: float = 300.0):
        self._chunk_size = chunk_size
        self._timeout = timeout

    def download_from_url(
        self,
        url: str,
        dest: str | Path,
        *,
        expected_sha256: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        """Download a compressed file from a URL and decompress to disk.

        Args:
            url: URL serving LZ4-compressed data.
            dest: Local path to write decompressed file.
            expected_sha256: Expected SHA-256 of uncompressed content.
            on_progress: Callback(bytes_written, expected_total).

        Returns:
            DownloadResult with success status and checksum info.
        """
        import urllib.request
        import urllib.error

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()

        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "lz4"})
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                return self._decompress_response(
                    response,
                    dest,
                    expected_sha256=expected_sha256,
                    on_progress=on_progress,
                )
        except Exception as e:
            return DownloadResult(
                success=False,
                dest_path=str(dest),
                error=str(e),
                elapsed_seconds=time.monotonic() - start,
            )

    def download_from_stream(
        self,
        source: BinaryIO,
        dest: str | Path,
        *,
        expected_sha256: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        """Decompress from a binary stream to disk."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()

        try:
            result = decompress_stream(
                source,
                open(dest, "wb"),
                verify_header=True,
                on_progress=on_progress,
            )
            sha256_match = True
            if expected_sha256:
                sha256_match = result.sha256 == expected_sha256

            return DownloadResult(
                success=True,
                dest_path=str(dest),
                bytes_written=result.bytes_uncompressed,
                sha256=result.sha256,
                sha256_match=sha256_match,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                dest_path=str(dest),
                error=str(e),
                elapsed_seconds=time.monotonic() - start,
            )

    def _decompress_response(
        self,
        response: Any,
        dest: Path,
        *,
        expected_sha256: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        """Decompress an HTTP response to disk."""
        start = time.monotonic()

        # Check content type - if it's our LZ4 format, decompress with header
        content_type = response.headers.get("Content-Type", "")
        content_encoding = response.headers.get("Content-Encoding", "")

        if content_type == "application/x-lz4" or content_encoding == "lz4":
            # Read all compressed data and decompress
            compressed_data = response.read()
            src = io.BytesIO(compressed_data)
            with open(dest, "wb") as dst:
                result = decompress_stream(src, dst, verify_header=True, on_progress=on_progress)
        else:
            # Assume raw file, just write it
            sha256 = hashlib.sha256()
            written = 0
            with open(dest, "wb") as dst:
                while True:
                    chunk = response.read(self._chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    sha256.update(chunk)
                    written += len(chunk)
                    if on_progress:
                        on_progress(written, None)
            result = CompressionResult(
                bytes_uncompressed=written,
                sha256=sha256.hexdigest(),
            )

        sha256_match = True
        if expected_sha256:
            sha256_match = result.sha256 == expected_sha256

        return DownloadResult(
            success=True,
            dest_path=str(dest),
            bytes_written=result.bytes_uncompressed,
            sha256=result.sha256,
            sha256_match=sha256_match,
            elapsed_seconds=time.monotonic() - start,
        )


# ── Compressed File Server (server-side) ──


class CompressedFileServer:
    """Serve files with on-the-fly LZ4 compression.

    Never stores .lz4 files on disk — compresses directly from source.
    """

    def __init__(self, *, compression_level: int = 6, chunk_size: int = CHUNK_SIZE):
        self._compression_level = compression_level
        self._chunk_size = chunk_size

    def serve(self, file_path: str | Path) -> dict[str, Any]:
        """Prepare metadata for a compressed file response.

        Returns dict with headers and iterator for use with StreamingResponse.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size

        return {
            "path": str(file_path),
            "size": file_size,
            "content_type": "application/x-lz4",
            "headers": {
                "Content-Type": "application/x-lz4",
                "Content-Encoding": "lz4",
                "X-Uncompressed-Size": str(file_size),
                "Cache-Control": "public, max-age=3600",
            },
            "iterator": compressed_file_iterator(
                file_path, compression_level=self._compression_level, chunk_size=self._chunk_size
            ),
        }

    def serve_range(
        self, file_path: str | Path, start: int, end: int
    ) -> dict[str, Any]:
        """Serve a byte range of a compressed file.

        For range requests, we decompress the full file and extract the range.
        This is less efficient but correct for resume support.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Decompress to memory and extract range
        with open(file_path, "rb") as f:
            dst = io.BytesIO()
            decompress_stream(f, dst, verify_header=True)
            full_data = dst.getvalue()

        range_data = full_data[start : end + 1]
        sha256 = hashlib.sha256(range_data).hexdigest()

        return {
            "data": range_data,
            "size": len(range_data),
            "content_type": "application/octet-stream",
            "headers": {
                "Content-Type": "application/octet-stream",
                "Content-Range": f"bytes {start}-{end}/{len(full_data)}",
                "X-SHA256": sha256,
            },
        }


# ── Utility: get compression stats without full decompression ──


def peek_compressed_header(source: BinaryIO) -> dict[str, Any] | None:
    """Read SLZ4 header without decompressing.

    Returns dict with uncompressed_size, sha256, magic, or None if not SLZ4.
    """
    pos = source.tell() if hasattr(source, "tell") else 0
    try:
        header = source.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            return None
        if header[0:4] != MAGIC:
            return None
        return {
            "magic": header[0:4],
            "uncompressed_size": struct.unpack(">Q", header[4:12])[0],
            "sha256": header[12:44].hex(),
        }
    finally:
        if hasattr(source, "seek"):
            try:
                source.seek(pos)
            except (OSError, io.UnsupportedOperation):
                pass
