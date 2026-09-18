"""
Multi-part download orchestrator.

Downloads multiple files as a single logical group with combined
progress tracking and per-part resume.  Ideal for split archives
(RAR parts, multi-file releases).

Usage::

    from downcraft.download.multipart import download_parts

    urls = [
        "https://example.com/part01.rar",
        "https://example.com/part02.rar",
        "https://example.com/part03.rar",
    ]
    result = download_parts(urls, dest_dir="/tmp/parts")

Requires: ``downcraft`` (no extra deps beyond requests).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .http import DownloadError, download_file
from .state import get_state

logger = logging.getLogger(__name__)


@dataclass
class PartResult:
    """Result of a single part download."""

    url: str
    dest: Path
    status: str  # complete | skipped | failed
    bytes_downloaded: int = 0
    elapsed: float = 0.0
    error: str = ""


@dataclass
class GroupResult:
    """Result of a multi-part download group."""

    group_key: str
    dest_dir: str
    status: str  # complete | partial | failed
    parts: list[PartResult] = field(default_factory=list)
    total_bytes: int = 0
    elapsed: float = 0.0

    @property
    def completed_count(self) -> int:
        return sum(1 for p in self.parts if p.status == "complete")

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.parts if p.status == "failed")

    @property
    def total_count(self) -> int:
        return len(self.parts)


def download_parts(
    urls: list[str],
    dest_dir: str | Path,
    *,
    group_key: str = "",
    filenames: list[str] | None = None,
    checksums: list[str] | None = None,
    max_workers: int = 1,
    on_progress: Callable[[int, int, int, float], None] | None = None,
    on_part_complete: Callable[[PartResult], None] | None = None,
    skip_if_exists: bool = True,
) -> GroupResult:
    """Download multiple files as a single logical group.

    Each URL is downloaded independently with resume support.  Progress
    is tracked across all parts in the persistent state store.

    Args:
        urls: List of URLs to download.
        dest_dir: Directory to save files into.
        group_key: Unique key for this group in state (auto-derived from
            dest_dir if empty).
        filenames: Optional explicit filenames (one per URL).  If omitted,
            filenames are derived from the URL path.
        checksums: Optional SHA-256 checksums (one per URL, empty string
            to skip verification for a part).
        max_workers: Number of concurrent downloads (default 1 = serial).
            Set >1 for concurrent downloads.
        on_progress: Per-chunk callback ``(part_index, bytes_done,
            total_bytes, speed_bps)``.
        on_part_complete: Called after each part finishes.
        skip_if_exists: Skip download if file exists with matching checksum.

    Returns:
        GroupResult with per-part details.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not group_key:
        group_key = f"parts:{dest_dir}"

    if filenames is None:
        filenames = [_filename_from_url(url) for url in urls]
    if checksums is None:
        checksums = [""] * len(urls)

    st = get_state()
    st.create(group_key, str(dest_dir))

    t0 = time.time()
    parts: list[PartResult] = []

    def _on_chunk(part_idx: int, bytes_done: int, total: int, speed: float):
        if on_progress:
            on_progress(part_idx, bytes_done, total, speed)

    for idx, (url, fname, cksum) in enumerate(zip(urls, filenames, checksums)):
        dest = dest_dir / fname
        st.update_file_progress(
            group_key,
            str(dest),
            url,
            0,
            0,
            checksum=cksum,
        )

        pt0 = time.time()
        try:

            def _chunk_cb(
                done: int, total: int, _idx: int = idx, _dest: dest = dest, _url: url = url
            ):
                elapsed = time.time() - pt0
                speed = done / elapsed if elapsed > 0 else 0
                st.update_file_progress(
                    group_key,
                    str(_dest),
                    _url,
                    done,
                    max(done, total),
                    checksum=cksum,
                    complete=(done >= total and total > 0),
                )
                _on_chunk(_idx, done, max(done, total), speed)

            download_file(
                url=url,
                dest=dest,
                checksum=cksum,
                on_chunk=_chunk_cb,
                skip_if_exists=skip_if_exists,
            )

            elapsed = time.time() - pt0
            result = PartResult(
                url=url,
                dest=dest,
                status="complete",
                bytes_downloaded=dest.stat().st_size,
                elapsed=elapsed,
            )
            parts.append(result)

            st.update_file_progress(
                group_key,
                str(dest),
                url,
                result.bytes_downloaded,
                result.bytes_downloaded,
                checksum=cksum,
                complete=True,
            )

            if on_part_complete:
                on_part_complete(result)

        except DownloadError as exc:
            elapsed = time.time() - pt0
            result = PartResult(
                url=url,
                dest=dest,
                status="failed",
                elapsed=elapsed,
                error=str(exc),
            )
            parts.append(result)
            logger.warning("Part %d failed: %s — %s", idx + 1, fname, exc)

            if on_part_complete:
                on_part_complete(result)

    total_bytes = sum(p.bytes_downloaded for p in parts)
    failed = sum(1 for p in parts if p.status == "failed")
    group_status = "complete" if failed == 0 else ("partial" if failed < len(urls) else "failed")

    st.set_status(group_key, group_status)
    st.flush()

    elapsed = time.time() - t0
    return GroupResult(
        group_key=group_key,
        dest_dir=str(dest_dir),
        status=group_status,
        parts=parts,
        total_bytes=total_bytes,
        elapsed=elapsed,
    )


def _filename_from_url(url: str) -> str:
    """Extract filename from URL path."""
    from urllib.parse import urlparse

    path = urlparse(url).path
    # Strip trailing slashes, get last component
    name = path.rstrip("/").split("/")[-1]
    # Remove query params if any
    name = name.split("?")[0]
    return name or "download"


def parse_urls_file(path: str | Path) -> list[str]:
    """Parse a text file of URLs (one per line).

    Skips blank lines and lines starting with ``#``.
    """
    urls: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls
