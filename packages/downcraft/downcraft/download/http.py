"""
HTTP downloader with byte-level resume via Range headers and optional LZ4 compression.

Key design:
- Each file is downloaded to a ``.sgpart`` temp path, then atomically renamed
  to the final name on completion (no corrupt files on crash).
- On resume, the existing ``.sgpart`` file size is sent as the Range header.
- Cleans up stale ``.sgpart`` files for a clean start when no state tracks them.
- Optional LZ4 compression: server sends compressed, client decompresses on-the-fly.
- Auto-detects already-compressed content to skip double compression.
- Checksum-based skip: avoids re-downloading identical files.
"""

import hashlib
import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# File extensions that are already compressed (skip double compression)
COMPRESSED_EXTENSIONS = {
    ".zip", ".gz", ".bz2", ".xz", ".lz4", ".zst", ".lz", ".br", ".tgz",
    ".tar.gz", ".tar.bz2", ".tar.xz", ".7z", ".rar", ".cab",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif",  # image formats
    ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".webm",  # media formats
    ".woff", ".woff2", ".ttf", ".otf",  # font formats
    ".pyc", ".pyo", ".class",  # compiled
    ".exe", ".dll", ".so", ".dylib",  # binaries
}


class DownloadError(Exception):
    """Raised when a download fails permanently."""


def _part_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".sgpart")


def _resolve_range_start(part_path: Path) -> int:
    """Return the byte position to resume from, or 0 for fresh download."""
    if part_path.exists():
        return part_path.stat().st_size
    return 0


def _is_already_compressed(url: str, headers: dict) -> bool:
    """Check if content is likely already compressed based on URL and headers."""
    # Check Content-Encoding header
    content_encoding = headers.get("Content-Encoding", "").lower()
    if content_encoding in ("gzip", "br", "zstd", "lz4", "deflate"):
        return True

    # Check Content-Type header
    content_type = headers.get("Content-Type", "").lower()
    compressed_types = {
        "application/zip", "application/gzip", "application/x-bzip2",
        "application/x-xz", "application/x-lz4", "application/zstd",
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "audio/mpeg", "video/mp4", "video/webm",
        "font/woff", "font/woff2",
    }
    for ct in compressed_types:
        if ct in content_type:
            return True

    # Check file extension in URL
    url_path = url.split("?")[0].lower()
    for ext in COMPRESSED_EXTENSIONS:
        if url_path.endswith(ext):
            return True

    return False


def _validate_content_range(
    headers: dict, expected_start: int, total_size: int,
) -> bool:
    """Validate that the Content-Range header matches the requested range.

    Returns True if the response is consistent with the resume request,
    False if the server returned a mismatched range (caller should restart).
    """
    cr = headers.get("Content-Range", "")
    if not cr:
        return True
    m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", cr)
    if not m:
        return False
    resp_start = int(m.group(1))
    resp_total = m.group(3)
    if resp_start != expected_start:
        return False
    if resp_total != "*" and int(resp_total) != total_size:
        return False
    return True


def _is_compressed_response(headers: dict) -> bool:
    """Check if server response indicates LZ4 compression."""
    content_type = headers.get("Content-Type", "")
    content_encoding = headers.get("Content-Encoding", "")
    return content_type == "application/x-lz4" or content_encoding == "lz4"


def _verify_checksum(file_path: Path, expected: str) -> bool:
    """Verify SHA-256 checksum of a file."""
    if not file_path.exists():
        return False
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(block)
    return hasher.hexdigest() == expected


def get_file_size(url: str) -> Optional[int]:
    """Get file size from server without downloading.

    Returns file size in bytes, or None if unknown.
    """
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            return int(content_length)
    except Exception:
        pass
    return None


def estimate_download_size(url: str) -> dict:
    """Estimate download size and compression savings.

    Returns dict with:
        - total_bytes: Total file size
        - is_compressed: Whether content is already compressed
        - estimated_savings: Estimated bandwidth savings from compression
    """
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        resp.raise_for_status()

        total_bytes = 0
        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            total_bytes = int(content_length)

        is_compressed = _is_already_compressed(url, resp.headers)

        # Estimate savings: if not compressed, LZ4 typically saves 30-50%
        estimated_savings = 0
        if not is_compressed and total_bytes > 0:
            estimated_savings = int(total_bytes * 0.4)  # conservative 40% estimate

        return {
            "total_bytes": total_bytes,
            "is_compressed": is_compressed,
            "estimated_savings": estimated_savings,
            "content_type": resp.headers.get("Content-Type", "unknown"),
            "content_encoding": resp.headers.get("Content-Encoding", "none"),
        }
    except Exception as e:
        return {
            "total_bytes": 0,
            "is_compressed": False,
            "estimated_savings": 0,
            "error": str(e),
        }


def download_file(
    url: str,
    dest: Path,
    expected_size: int = 0,
    checksum: str = "",
    on_chunk: Optional[Callable[[int, int], None]] = None,
    on_complete: Optional[Callable[[Path], None]] = None,
    compressed: bool = False,
    skip_if_exists: bool = True,
) -> Path:
    """Download a single file with resume support.

    Args:
        url: HTTP/HTTPS URL to download from.
        dest: Final destination path on disk.
        expected_size: Expected total bytes (0 = unknown).
        checksum: SHA-256 hex string to verify after download.
        on_chunk: Called after each chunk with ``(bytes_downloaded, total_bytes)``.
                  ``total_bytes`` may be 0 if the server provides no
                  Content-Length and *expected_size* was not given.
        on_complete: Called with final path after successful download.
        compressed: If True, expect LZ4-compressed response and decompress on-the-fly.
        skip_if_exists: If True and checksum provided, skip download if file exists with matching checksum.

    Returns:
        The final destination path on success.

    Raises:
        DownloadError: If the download fails permanently (after retries).
    """
    # Skip download if file exists with matching checksum
    if skip_if_exists and checksum and dest.exists():
        if _verify_checksum(dest, checksum):
            logger.info("Skipping %s (checksum matches)", dest.name)
            if on_chunk:
                size = dest.stat().st_size
                on_chunk(size, size)
            if on_complete:
                on_complete(dest)
            return dest

    part = _part_path(dest)
    resume_at = _resolve_range_start(part)
    headers: dict = {}

    if resume_at > 0:
        headers["Range"] = f"bytes={resume_at}-"
        logger.info("Resuming %s from byte %d", dest.name, resume_at)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            if resume_at > 0 and resp.status_code != 206:
                logger.warning(
                    "Server doesn't support Range (got %d), restarting %s from 0",
                    resp.status_code,
                    dest.name,
                )
                part.unlink(missing_ok=True)
                resume_at = 0
                headers.pop("Range", None)
                continue

            if resume_at > 0 and resp.status_code == 206:
                full_size = expected_size
                if full_size <= 0:
                    cl = resp.headers.get("Content-Length", "0")
                    full_size = int(cl) + resume_at if cl.isdigit() else 0
                if full_size > 0 and not _validate_content_range(
                    resp.headers, resume_at, full_size,
                ):
                    logger.warning(
                        "Content-Range mismatch on %s, restarting from 0",
                        dest.name,
                    )
                    part.unlink(missing_ok=True)
                    resume_at = 0
                    headers.pop("Range", None)
                    continue

            mode = "ab" if resume_at > 0 else "wb"
            total = expected_size or int(resp.headers.get("Content-Length", 0))
            if expected_size <= 0 and resume_at > 0 and total > 0:
                total += resume_at

            dest.parent.mkdir(parents=True, exist_ok=True)

            # Auto-detect compression: skip if content is already compressed
            content_is_compressed = _is_already_compressed(url, resp.headers)
            server_has_lz4 = _is_compressed_response(resp.headers)

            # Decide whether to decompress
            should_decompress = False
            if compressed and server_has_lz4:
                # Explicitly requested LZ4 and server is sending it
                should_decompress = True
            elif content_is_compressed:
                # Content is already compressed (gzip, zstd, etc.) - don't decompress
                should_decompress = False
                logger.info("Content already compressed, skipping LZ4 decompression for %s", dest.name)

            if should_decompress:
                _download_compressed(resp, part, mode, total, on_chunk)
            else:
                _download_raw(resp, part, mode, total, on_chunk)

            if checksum:
                hasher = hashlib.sha256()
                with open(part, "rb") as f:
                    for block in iter(lambda: f.read(CHUNK_SIZE), b""):
                        hasher.update(block)
                actual = hasher.hexdigest()
                if actual != checksum:
                    logger.error(
                        "Checksum mismatch for %s: expected %s, got %s",
                        dest.name,
                        checksum,
                        actual,
                    )
                    part.unlink(missing_ok=True)
                    raise DownloadError(
                        f"Checksum mismatch for {dest.name}"
                    )

            os.replace(str(part), str(dest))
            logger.info("Downloaded %s (%.2f MB)", dest.name, dest.stat().st_size / 1e6)

            if on_complete:
                on_complete(dest)

            return dest

        except (requests.RequestException, OSError) as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                dest.name,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise DownloadError(f"Failed to download {dest.name} after {MAX_RETRIES} attempts: {e}") from e

    raise DownloadError(f"Failed to download {dest.name}")


def _download_raw(
    resp: requests.Response,
    part: Path,
    mode: str,
    total: int,
    on_chunk: Optional[Callable[[int, int], None]],
) -> None:
    """Download raw (uncompressed) data."""
    with open(part, mode) as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            if on_chunk:
                bytes_done = part.stat().st_size
                on_chunk(bytes_done, max(bytes_done, total))


def _download_compressed(
    resp: requests.Response,
    part: Path,
    mode: str,
    total: int,
    on_chunk: Optional[Callable[[int, int], None]],
) -> None:
    """Download LZ4-compressed data and decompress on-the-fly."""
    from downcraft.download.compress import decompress_stream

    # Read all compressed data
    compressed_data = resp.read()

    # Decompress to file
    src = io.BytesIO(compressed_data)
    with open(part, mode) as dst:
        decompress_stream(src, dst, verify_header=True)

    if on_chunk:
        bytes_done = part.stat().st_size
        on_chunk(bytes_done, max(bytes_done, total))


def download_compressed(
    url: str,
    dest: Path,
    expected_size: int = 0,
    checksum: str = "",
    on_chunk: Optional[Callable[[int, int], None]] = None,
    on_complete: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Download a compressed file with resume support.

    Convenience wrapper that sets compressed=True.
    """
    return download_file(
        url,
        dest,
        expected_size=expected_size,
        checksum=checksum,
        on_chunk=on_chunk,
        on_complete=on_complete,
        compressed=True,
    )
