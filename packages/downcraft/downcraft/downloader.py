"""
HTTP downloader with byte-level resume via Range headers.

Key design:
- Each file is downloaded to a ``.sgpart`` temp path, then atomically renamed
  to the final name on completion (no corrupt files on crash).
- On resume, the existing ``.sgpart`` file size is sent as the Range header.
- Cleans up stale ``.sgpart`` files for a clean start when no state tracks them.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class DownloadError(Exception):
    """Raised when a download fails permanently."""


def _part_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".sgpart")


def _resolve_range_start(part_path: Path) -> int:
    """Return the byte position to resume from, or 0 for fresh download."""
    if part_path.exists():
        return part_path.stat().st_size
    return 0


def _expected_size_bytes(expected_size_gb: Optional[float]) -> int:
    """Convert an optional GB hint to bytes, or 0 if unknown."""
    if expected_size_gb is not None and expected_size_gb > 0:
        return int(expected_size_gb * (1024 ** 3))
    return 0


def download_file(
    url: str,
    dest: Path,
    expected_size: int = 0,
    checksum: str = "",
    on_chunk: Optional[Callable[[int, int], None]] = None,
    on_complete: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Download a single file with resume support.

    Args:
        url: HTTP/HTTPS URL to download from.
        dest: Final destination path on disk.
        expected_size: Expected total bytes (0 = unknown).
        checksum: SHA-256 hex string to verify after download.
        on_chunk: Called after each chunk with ``(bytes_downloaded, total_bytes)``.
        on_complete: Called with final path after successful download.

    Returns:
        The final destination path on success.

    Raises:
        DownloadError: If the download fails permanently (after retries).
    """
    part = _part_path(dest)
    resume_at = _resolve_range_start(part)
    headers = {}

    if resume_at > 0:
        headers["Range"] = f"bytes={resume_at}-"
        logger.info("Resuming %s from byte %d", dest.name, resume_at)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            # If server didn't return 206 (Partial Content), start over
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

            mode = "ab" if resume_at > 0 else "wb"
            total = expected_size or int(resp.headers.get("Content-Length", 0))
            # For resuming, the Content-Length is the REMAINING bytes
            if resume_at > 0 and total > 0:
                total += resume_at
            total = total or 0

            hasher = hashlib.sha256() if checksum else None

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(part, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    if on_chunk:
                        bytes_done = part.stat().st_size
                        on_chunk(bytes_done, max(bytes_done, total))

            # Verify checksum
            if checksum and hasher:
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

            # Atomically rename .sgpart → final
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

    raise DownloadError(f"Failed to download {dest.name}")  # unreachable
