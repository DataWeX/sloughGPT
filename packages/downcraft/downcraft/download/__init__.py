"""
downcraft.download — Direct file download with cross-session resume and optional compression.

Provides HTTP download with Range header resume, persistent state tracking,
SHA-256 verification, optional LZ4 compression, and multi-part group download.
"""

from .http import (
    DownloadError,
    download_compressed,
    download_file,
    estimate_download_size,
    get_file_size,
)
from .multipart import GroupResult, PartResult, download_parts, parse_urls_file
from .state import FileProgress, ModelState, get_state
from .verify import verify_file

__all__ = [
    "DownloadError",
    "download_file",
    "download_compressed",
    "download_parts",
    "get_file_size",
    "estimate_download_size",
    "get_state",
    "ModelState",
    "FileProgress",
    "GroupResult",
    "PartResult",
    "parse_urls_file",
    "verify_file",
]
