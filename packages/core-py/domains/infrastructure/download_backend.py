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
