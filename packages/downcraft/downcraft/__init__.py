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
    - ``download``: direct file download with resume
    - ``resolve``: extract real download URLs from ad-heavy pages
    - ``compress``: LZ4 compression/decompression (optional, requires lz4)

Features:
    - Cross-session resume via HTTP Range headers
    - Auto-detect already-compressed content (skip double compression)
    - Checksum-based skip (avoid re-downloading identical files)
    - Size estimation before download
    - LZ4 compression support (optional)

Use cases:
    1. Direct download::

        download("https://example.com/bigfile.iso", "/tmp/bigfile.iso")

    2. Compressed download (server sends LZ4, client decompresses)::

        download("https://example.com/bigfile.iso.lz4", "/tmp/bigfile.iso", compressed=True)

    3. Resolve + download (scrape page, find real link)::

        from downcraft.resolve import resolve_and_download
        resolve_and_download("https://example.com/download-page", "/tmp/file.zip")

    4. Estimate download size::

        from downcraft import estimate_download_size
        info = estimate_download_size("https://example.com/bigfile.iso")
        print(f"Size: {info['total_bytes'] / 1e6:.1f} MB")

    5. CLI::

        python -m downcraft url https://...
        python -m downcraft resolve https://example.com/download-page
        python -m downcraft estimate https://example.com/bigfile.iso
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from .download import download_file, DownloadError
from .download import state
from .resolve import resolve_page, resolve_and_download
from .resolve import patterns
from .server import start_capture_server, CaptureEntry

# Compression APIs (optional dependency)
try:
    from .download.compress import (
        compress_file,
        decompress_file,
        compress_bytes,
        decompress_bytes,
        compress_stream,
        decompress_stream,
        auto_decompress,
        is_compressed_file,
        peek_compressed_header,
        compressed_file_iterator,
        CompressedFileServer,
        CompressionResult,
    )
    _HAS_COMPRESSION = True
except ImportError:
    _HAS_COMPRESSION = False

logger = logging.getLogger(__name__)

__all__ = [
    "download",
    "resolve_page",
    "resolve_and_download",
    "state",
    "patterns",
    "download_file",
    "DownloadError",
    "start_capture_server",
    "CaptureEntry",
    # Compression (if lz4 installed)
    "compress_file",
    "decompress_file",
    "compress_bytes",
    "decompress_bytes",
    "compress_stream",
    "decompress_stream",
    "auto_decompress",
    "is_compressed_file",
    "peek_compressed_header",
    "compressed_file_iterator",
    "CompressedFileServer",
    "CompressionResult",
    # Download helpers
    "estimate_download_size",
    "get_file_size",
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
    compressed: bool = False,
    skip_if_exists: bool = True,
) -> Dict:
    """Download a single file from any URL with cross-session resume.

    Uses ``~/.downcraft/state.json`` to track progress so that
    if the process dies mid-download, the next call resumes from
    the last byte received (via HTTP ``Range`` header).

    Features:
        - Auto-detects already-compressed content (skips double compression)
        - Checksum-based skip (avoids re-downloading identical files)
        - Cross-session resume via HTTP Range headers

    Args:
        url: HTTP/HTTPS URL to download.
        dest: Local destination path.
        expected_size: Expected total bytes (0 = auto-detect from server).
        checksum: SHA-256 hex string to verify after download.
        label: Human label for logging (defaults to filename).
        on_progress: Called per-chunk with ``(bytes_downloaded, total_bytes, speed_bps)``.
        compressed: If True, expect LZ4-compressed response and decompress on-the-fly.
        skip_if_exists: If True and checksum provided, skip download if file exists with matching checksum.

    Returns:
        Dict with keys: ``status``, ``dest``, ``elapsed``, ``total_bytes``.
    """
    from .download.http import download_file, _verify_checksum

    dest = Path(dest)
    label = label or dest.name
    st = state.get_state()
    lookup_key = url  # use URL as the state tracking key

    # Skip download if file exists with matching checksum
    if skip_if_exists and checksum and dest.exists():
        if _verify_checksum(dest, checksum):
            logger.info("%s already exists with matching checksum", label)
            return {"status": "already_downloaded", "dest": str(dest), "label": label}

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
        download_file(
            url=url,
            dest=dest,
            expected_size=expected_size,
            checksum=checksum,
            on_chunk=_chunk_cb,
            compressed=compressed,
            skip_if_exists=False,  # Already checked above
        )
    except DownloadError:
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
