"""
downcraft — Generic HTTP/HTTPS downloader with cross-session resume
via HTTP Range headers and persistent JSON state.

Resumes partial downloads even after power loss, process crash, or
days-long gaps between sessions — as long as the partial file on disk
and server's ``ETag`` still match.

This package is deliberately HuggingFace-agnostic.  It downloads any URL.
The HuggingFace-specific model download/resume/verify workflows live in
the application layer (``domains.infrastructure.hf_hub``), composed from
the generic primitives here (``downloader``, ``state``, ``verify``).

Submodules:
    - ``downloader``: byte-level resume via Range headers
    - ``resolver``: extract real download URLs from ad-heavy pages
    - ``state``: persistent download state across restarts
    - ``verify``: checksum / size verification

Use cases:
    1. Direct download::

        download("https://example.com/bigfile.iso", "/tmp/bigfile.iso")

    2. Resolve + download (scrape page, find real link)::

        from downcraft.resolver import resolve_and_download
        resolve_and_download("https://example.com/download-page", "/tmp/file.zip")

    3. CLI::

        python -m downcraft url https://...
        python -m downcraft resolve https://example.com/download-page
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from . import downloader, resolver, state, verify
from .resolver import resolve_page, resolve_and_download

logger = logging.getLogger(__name__)

__all__ = [
    "download",
    "resolve_page",
    "resolve_and_download",
    "state",
    "downloader",
    "resolver",
    "verify",
]


# ---------------------------------------------------------------------------
# Generic download — any URL
# ---------------------------------------------------------------------------

def download(
    url: str,
    dest: Union[str, Path],
    expected_size: int = 0,
    checksum: str = "",
    label: str = "",
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> Dict:
    """Download a single file from any URL with cross-session resume.

    Uses ``~/.downcraft/state.json`` to track progress so that
    if the process dies mid-download, the next call resumes from
    the last byte received (via HTTP ``Range`` header).

    Args:
        url: HTTP/HTTPS URL to download.
        dest: Local destination path.
        expected_size: Expected total bytes (0 = auto-detect from server).
        checksum: SHA-256 hex string to verify after download.
        label: Human label for logging (defaults to filename).
        on_progress: Called per-chunk with ``(bytes_downloaded, total_bytes, speed_bps)``.

    Returns:
        Dict with keys: ``status``, ``dest``, ``elapsed``, ``total_bytes``.
    """
    dest = Path(dest)
    label = label or dest.name
    st = state.get_state()
    lookup_key = url  # use URL as the state tracking key

    # Check existing state
    existing = st.get(lookup_key)
    if existing and existing.status == "complete":
        logger.info("%s already downloaded", label)
        return {"status": "already_downloaded", "dest": str(dest), "label": label}

    start = time.time()
    total_bytes = [0]

    def _chunk_cb(bytes_done: int, total: int):
        nonlocal total_bytes
        total_bytes[0] = max(total_bytes[0], total)
        st.update_file_progress(
            lookup_key,
            str(dest),
            url,
            bytes_done,
            max(bytes_done, total),
            checksum=checksum,
            complete=(bytes_done >= total and total > 0),
        )
        elapsed = time.time() - start
        speed = bytes_done / elapsed if elapsed > 0 else 0
        if on_progress:
            on_progress(bytes_done, max(bytes_done, total), speed)

    st.create(lookup_key, str(dest.parent))

    try:
        downloader.download_file(
            url=url,
            dest=dest,
            expected_size=expected_size,
            checksum=checksum,
            on_chunk=_chunk_cb,
        )
    except downloader.DownloadError:
        st.set_status(lookup_key, "failed", error=f"Failed to download {label}")
        st.flush()
        raise

    st.set_status(lookup_key, "complete")
    st.flush()
    elapsed = time.time() - start
    return {
        "status": "complete",
        "dest": str(dest),
        "elapsed": round(elapsed, 1),
        "total_bytes": total_bytes[0],
    }
