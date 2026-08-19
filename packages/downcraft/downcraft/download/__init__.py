"""
downcraft.download — Direct file download with cross-session resume.

Provides HTTP download with Range header resume, persistent state tracking,
and SHA-256 verification.
"""

from .http import DownloadError, download_file
from .state import FileProgress, ModelState, get_state
from .verify import verify_file

__all__ = [
    "DownloadError",
    "download_file",
    "get_state",
    "ModelState",
    "FileProgress",
    "verify_file",
]
