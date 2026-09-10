"""
Download backend protocol — abstract interface for pluggable download sources.

``DownloadManager`` delegates all source-specific logic (cache resolution,
file listing, download execution, cleanup) to a backend implementing this
protocol.  The manager itself only handles scheduling, progress tracking,
cancellation, and stale cleanup.

Implementations:
    - ``HFDownloadBackend`` in ``hf_hub`` — HuggingFace model downloads
    - (future) generic HTTP, S3, Git, etc.

Usage::

    from domains.infrastructure.download_backend import DownloadBackend

    class MyBackend(DownloadBackend):
        def is_cached(self, resource_id): ...
        def download(self, resource_id, on_progress, on_file_complete): ...
        # ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class FileEstimate:
    """Metadata for a single file to download."""
    path: str
    size: int
    checksum: str = ""
    download_url: str = ""


class DownloadBackend(ABC):
    """Abstract backend for a download source.

    Each method maps to a source-specific operation.  The
    ``DownloadManager`` calls these methods and handles the generic
    orchestration (scheduling, progress, cancel, stale cleanup) on top.
    """

    @abstractmethod
    def is_cached(self, resource_id: str, deep_check: bool = False) -> bool:
        """Return True if the resource is fully cached on disk.

        Args:
            resource_id: Opaque identifier (e.g. HuggingFace model ID, URL).
            deep_check: If True, verify via network (file sizes, checksums).
                        Skip for batch listing.
        """

    @abstractmethod
    def get_cache_dir(self, resource_id: str) -> str:
        """Return the local cache directory path for a resource."""

    @abstractmethod
    def estimate_total(self, resource_id: str) -> int:
        """Estimate total download size in bytes.  Returns 0 on failure."""

    @abstractmethod
    def list_files(self, resource_id: str) -> List[FileEstimate]:
        """List downloadable files for a resource."""

    @abstractmethod
    def download(
        self,
        resource_id: str,
        on_progress: Callable[[str, int, int, float], None],
        on_file_complete: Callable[[str, str], None],
    ) -> Dict:
        """Execute the download.  Called in a thread executor.

        Args:
            resource_id: What to download.
            on_progress: Callback ``(resource_id, bytes_done, total, speed_bps)``.
            on_file_complete: Callback ``(resource_id, file_path)`` per file.

        Returns:
            Dict with at least ``status`` and ``cache_dir`` keys.
        """

    @abstractmethod
    def cleanup(self, resource_id: str) -> bool:
        """Remove incomplete/cached data for a resource.  Returns True if removed."""

    @abstractmethod
    def list_incomplete(self) -> List[str]:
        """Return resource IDs with incomplete downloads."""

    def prepare_download(self, resource_id: str) -> None:
        """Hook called before download starts.  Override to clean markers, etc."""

    def on_cancel(self, resource_id: str) -> None:
        """Hook called when a download is cancelled.  Override to update state."""

    def supports_compression(self, resource_id: str) -> bool:
        """Whether the external server supports LZ4 compression for this resource.

        When True, the download path should use ``CompressedDownloader``
        to fetch compressed data and decompress on-the-fly.
        """
        return False

    def supports_compressed_serve(self) -> bool:
        """Whether this backend can serve files with LZ4 compression.

        When True, ``serve_compressed()`` returns a StreamingResponse
        for use with the API server.
        """
        return False

    def serve_compressed(self, resource_id: str, file_path: str) -> Optional[Dict]:
        """Serve a cached file with on-the-fly LZ4 compression.

        Returns dict with ``iterator``, ``headers``, ``size`` keys
        for use with Starlette/FastAPI StreamingResponse, or None
        if the file is not available or compression is not supported.
        """

    def verify(self, resource_id: str) -> Dict[str, Any]:
        """Verify integrity of a cached resource.

        Checks that all expected files exist, have correct sizes,
        and match checksums where available.

        Returns dict with:
            - valid: bool
            - files_checked: int
            - files_valid: int
            - errors: list of error strings
        """
        cache_dir = self.get_cache_dir(resource_id)
        files = self.list_files(resource_id)
        errors = []
        files_checked = 0
        files_valid = 0

        for f in files:
            import os
            fpath = os.path.join(cache_dir, f.path)
            files_checked += 1
            if not os.path.exists(fpath):
                errors.append(f"Missing: {f.path}")
                continue
            if f.size > 0 and os.path.getsize(fpath) != f.size:
                errors.append(f"Size mismatch: {f.path}")
                continue
            if f.checksum:
                import hashlib
                actual = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
                if actual != f.checksum:
                    errors.append(f"Checksum mismatch: {f.path}")
                    continue
            files_valid += 1

        return {
            "valid": len(errors) == 0 and files_checked > 0,
            "files_checked": files_checked,
            "files_valid": files_valid,
            "errors": errors,
        }
