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
import re
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
                  ``total_bytes`` may be 0 if the server provides no
                  Content-Length and *expected_size* was not given.
        on_complete: Called with final path after successful download.

    Returns:
        The final destination path on success.

    Raises:
        DownloadError: If the download fails permanently (after retries).
    """
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
            with open(part, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if on_chunk:
                        bytes_done = part.stat().st_size
                        on_chunk(bytes_done, max(bytes_done, total))

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
