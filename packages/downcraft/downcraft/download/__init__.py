"""
downcraft.download — Direct file download with cross-session resume and optional compression.

Provides HTTP download with Range header resume, persistent state tracking,
SHA-256 verification, and optional LZ4 compression support.
"""

from .http import DownloadError, download_file, download_compressed, get_file_size, estimate_download_size
from .state import FileProgress, ModelState, get_state
from .verify import verify_file

__all__ = [
    "DownloadError",
    "download_file",
    "download_compressed",
    "get_file_size",
    "estimate_download_size",
    "get_state",
    "ModelState",
    "FileProgress",
    "verify_file",
]
