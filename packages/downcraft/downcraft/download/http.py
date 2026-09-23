"""
HTTP downloader with byte-level resume via Range headers and optional LZ4 compression.

Key design:
- Each file is downloaded to a ``.sgpart`` temp path, then atomically renamed
  to the final name on completion (no corrupt files on crash).
- On resume, the existing ``.sgpart`` file size is sent as the Range header.
- Cleans up stale ``.sgpart`` files for a clean start when no state tracks them.
- Optional LZ4 compression: server sends compressed, client decompresses on-the-fly.
- Edge gzip/zstd (gateway): advertises ``Accept-Encoding: zstd, gzip`` and
  transparently decodes to identity before writing, so Range resume offsets
  always refer to the decoded body.
- Auto-detects already-compressed content to skip double compression.
- Checksum-based skip: avoids re-downloading identical files.
"""

import hashlib
import io
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Edge/gateway content codings we can decode to identity.
# gzip is decoded by urllib3/requests transparently; zstd is decoded here.
ACCEPT_ENCODING = "zstd, gzip"

# File extensions that are already compressed (skip double compression)
COMPRESSED_EXTENSIONS = {
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".lz4",
    ".zst",
    ".lz",
    ".br",
    ".tgz",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".7z",
    ".rar",
    ".cab",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".avif",  # image formats
    ".mp3",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",  # media formats
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",  # font formats
    ".pyc",
    ".pyo",
    ".class",  # compiled
    ".exe",
    ".dll",
    ".so",
    ".dylib",  # binaries
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
        "application/zip",
        "application/gzip",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-lz4",
        "application/zstd",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "audio/mpeg",
        "video/mp4",
        "video/webm",
        "font/woff",
        "font/woff2",
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
    headers: dict,
    expected_start: int,
    total_size: int,
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


def _identity_total(headers: dict, expected_size: int = 0) -> int:
    """Decoded-body size for progress, preferring the edge's UCL header.

    Only for **full (200)** responses. On 206, Content-Length is remaining
    bytes — use :func:`_content_range_total` instead.
    """
    if expected_size > 0:
        return expected_size
    ucl = headers.get("X-Uncompressed-Content-Length", "")
    if ucl.isdigit():
        return int(ucl)
    # Already-identity responses: Content-Length is correct.
    if not headers.get("Content-Encoding"):
        cl = headers.get("Content-Length", "")
        if cl.isdigit():
            return int(cl)
    return 0


def _content_range_total(headers: dict) -> int:
    """Full object size from ``Content-Range: bytes a-b/total`` (0 if absent)."""
    cr = headers.get("Content-Range", "")
    m = re.match(r"bytes\s+\d+-\d+/(\d+)", cr)
    if m:
        return int(m.group(1))
    return 0


def _edge_content_encoding(headers: dict) -> str:
    """Return lowercased edge Content-Encoding, or '' if identity/LZ4."""
    ce = headers.get("Content-Encoding", "").lower().strip()
    if ce in ("zstd", "gzip", "deflate", "br"):
        return ce
    return ""


def _verify_checksum(file_path: Path, expected: str) -> bool:
    """Verify SHA-256 checksum of a file."""
    if not file_path.exists():
        return False
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(block)
    return hasher.hexdigest() == expected


def get_file_size(url: str) -> int | None:
    """Get file size from server without downloading.

    Prefers identity size (``X-Uncompressed-Content-Length``) so callers see
    the decoded length even when the edge compresses GET responses.

    Returns file size in bytes, or None if unknown.
    """
    try:
        resp = requests.head(
            url,
            headers={"Accept-Encoding": ACCEPT_ENCODING},
            timeout=10,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return _identity_total(resp.headers, 0) or None
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
        resp = requests.head(
            url,
            headers={"Accept-Encoding": ACCEPT_ENCODING},
            timeout=10,
            allow_redirects=True,
        )
        resp.raise_for_status()

        total_bytes = _identity_total(resp.headers, 0)
        is_compressed = _is_already_compressed(url, resp.headers) or bool(
            _edge_content_encoding(resp.headers)
        )

        # Estimate savings: if not compressed, LZ4 typically saves 30-50%
        estimated_savings = 0
        if not is_compressed and total_bytes > 0:
            estimated_savings = int(total_bytes * 0.4)  # conservative 40% estimate
        elif is_compressed and total_bytes > 0:
            # Edge-compressed: wire is already smaller; estimate remaining
            # gain from identity-vs-wire when UCL is present.
            ucl = resp.headers.get("X-Uncompressed-Content-Length", "")
            cl = resp.headers.get("Content-Length", "")
            if ucl.isdigit() and cl.isdigit() and int(ucl) > 0:
                wire = int(cl)
                identity = int(ucl)
                estimated_savings = max(0, identity - wire)

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
    on_chunk: Callable[[int, int], None] | None = None,
    on_complete: Callable[[Path], None] | None = None,
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
    # Always advertise edge codings; Range responses come back identity from
    # the gateway (skip-compress), so resume offsets stay consistent.
    headers: dict = {"Accept-Encoding": ACCEPT_ENCODING}

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
                    # 206 Content-Length is the *remaining* wire bytes, not
                    # the full object — never treat it as identity total.
                    full_size = _content_range_total(resp.headers)
                    if full_size <= 0:
                        cl = resp.headers.get("Content-Length", "0")
                        if cl.isdigit():
                            full_size = int(cl) + resume_at
                if full_size > 0 and not _validate_content_range(
                    resp.headers,
                    resume_at,
                    full_size,
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
            if resp.status_code == 206:
                # Progress total for a partial body = start + remaining.
                total = expected_size
                if total <= 0:
                    total = _content_range_total(resp.headers)
                if total <= 0:
                    cl = resp.headers.get("Content-Length", "0")
                    total = (int(cl) + resume_at) if cl.isdigit() else resume_at
            else:
                total = _identity_total(resp.headers, expected_size)

            dest.parent.mkdir(parents=True, exist_ok=True)

            # Auto-detect compression: skip if content is already compressed
            server_has_lz4 = _is_compressed_response(resp.headers)
            edge_ce = _edge_content_encoding(resp.headers)

            # Decide how to write the body:
            # 1. LZ4 protocol (explicit compressed=True + server lz4)
            # 2. Edge zstd → decode here (requests does not decode zstd)
            # 3. Edge gzip/deflate/br → urllib3 already decoded via iter_content
            # 4. Identity / already-compressed file → raw write
            if compressed and server_has_lz4:
                _download_compressed(resp, part, mode, total, on_chunk)
            elif edge_ce == "zstd":
                _download_zstd(resp, part, mode, total, on_chunk)
            else:
                # gzip arrives pre-decoded from iter_content; identity unchanged.
                if edge_ce and edge_ce != "gzip":
                    logger.debug(
                        "Content-Encoding %s on %s — writing decoded stream via requests",
                        edge_ce,
                        dest.name,
                    )
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
                    raise DownloadError(f"Checksum mismatch for {dest.name}")

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
                raise DownloadError(
                    f"Failed to download {dest.name} after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise DownloadError(f"Failed to download {dest.name}")


def _download_raw(
    resp: requests.Response,
    part: Path,
    mode: str,
    total: int,
    on_chunk: Callable[[int, int], None] | None,
) -> None:
    """Download raw (uncompressed) data.

    For ``Content-Encoding: gzip`` urllib3/requests has already decoded the
    stream, so chunks written here are identity bytes.
    """
    with open(part, mode) as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            if on_chunk:
                bytes_done = part.stat().st_size
                on_chunk(bytes_done, max(bytes_done, total))


def _download_zstd(
    resp: requests.Response,
    part: Path,
    mode: str,
    total: int,
    on_chunk: Callable[[int, int], None] | None,
) -> None:
    """Stream-decode ``Content-Encoding: zstd`` to identity bytes on disk.

    Writes decoded bytes only — resume Range offsets stay valid against the
    identity body the gateway serves for partial requests.
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    # Use the raw urllib3 body: iter_content would pass zstd frames through
    # undecoded (requests has no zstd codec).
    resp.raw.decode_content = False
    with open(part, mode) as f:
        with dctx.stream_reader(resp.raw) as reader:
            while True:
                chunk = reader.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                if on_chunk:
                    bytes_done = part.stat().st_size
                    on_chunk(bytes_done, max(bytes_done, total))


def _download_compressed(
    resp: requests.Response,
    part: Path,
    mode: str,
    total: int,
    on_chunk: Callable[[int, int], None] | None,
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
    on_chunk: Callable[[int, int], None] | None = None,
    on_complete: Callable[[Path], None] | None = None,
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


def download_range(
    url: str,
    dest: Path | str,
    start: int,
    end: int,
    checksum: str = "",
    on_chunk: Callable[[int, int], None] | None = None,
    on_complete: Callable[[Path], None] | None = None,
) -> Path:
    """Download a single byte range ``[start, end]`` (inclusive) with retries.

    Sends ``Range: bytes={start}-{end}`` and expects ``206 Partial Content``.
    Unlike :func:`download_file` (resume-to-end), this fetches only the
    requested slice so callers pay for part of a file, not the whole file.

    Slice downloads are always raw bytes — never LZ4-decompressed, since a
    partial LZ4 frame cannot be decoded on its own.

    Args:
        url: HTTP/HTTPS URL to download from.
        dest: Final destination path on disk (holds exactly the slice bytes).
        start: First byte offset (inclusive, ``>= 0``).
        end: Last byte offset (inclusive, ``>= start``).
        checksum: Optional SHA-256 hex of the *slice* to verify after download.
        on_chunk: Called with ``(bytes_written, expected_len)`` per chunk.
        on_complete: Called with final path after successful download.

    Returns:
        The final destination path on success.

    Raises:
        ValueError: If ``start``/``end`` are negative or ``end < start``.
        DownloadError: If the server ignores Range (no ``206``), returns a
            mismatched ``Content-Range``, truncates the slice, or fails
            permanently (after retries). Checksum mismatches also raise.
    """
    if start < 0 or end < 0:
        raise ValueError(f"Range offsets must be >= 0 (got start={start}, end={end})")
    if end < start:
        raise ValueError(f"Range end must be >= start (got start={start}, end={end})")

    dest = Path(dest)
    expected_len = end - start + 1

    # Skip download if dest already holds the slice with matching checksum
    if checksum and dest.exists() and dest.stat().st_size == expected_len:
        if _verify_checksum(dest, checksum):
            logger.info("Skipping %s (slice checksum matches)", dest.name)
            if on_chunk:
                on_chunk(expected_len, expected_len)
            if on_complete:
                on_complete(dest)
            return dest

    part = _part_path(dest)
    headers = {"Range": f"bytes={start}-{end}"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            if resp.status_code != 206:
                raise DownloadError(
                    f"Server does not honor Range for {dest.name} "
                    f"(got {resp.status_code}, expected 206); "
                    "refusing full download to save data"
                )

            cr = resp.headers.get("Content-Range", "")
            if cr:
                m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", cr)
                if not m or int(m.group(1)) != start:
                    raise DownloadError(
                        f"Content-Range mismatch for {dest.name}: {cr!r} (requested start={start})"
                    )

            dest.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    # Stop at the slice boundary even if the server sends more.
                    remaining = expected_len - written
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    f.write(chunk)
                    written += len(chunk)
                    if on_chunk:
                        on_chunk(written, expected_len)
                    if written >= expected_len:
                        break
            resp.close()

            if written < expected_len:
                logger.warning(
                    "Truncated range for %s: got %d/%d bytes (attempt %d/%d)",
                    dest.name,
                    written,
                    expected_len,
                    attempt,
                    MAX_RETRIES,
                )
                part.unlink(missing_ok=True)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                raise DownloadError(
                    f"Truncated range for {dest.name}: got {written}/{expected_len} bytes"
                )

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
                    raise DownloadError(f"Checksum mismatch for {dest.name}")

            os.replace(str(part), str(dest))
            logger.info("Downloaded range %s [%d-%d]", dest.name, start, end)

            if on_complete:
                on_complete(dest)

            return dest

        except (requests.RequestException, OSError) as e:
            logger.warning(
                "Attempt %d/%d failed for range %s [%d-%d]: %s",
                attempt,
                MAX_RETRIES,
                dest.name,
                start,
                end,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise DownloadError(
                    f"Failed to download range [{start}-{end}] of {dest.name} "
                    f"after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise DownloadError(f"Failed to download range [{start}-{end}] of {dest.name}")
