"""
FileManager — unified path resolution across VFS and host filesystem.

Resolves paths by checking VFS mounts first, then falling back to the
host filesystem. Components call fm.read_text(path) and the FileManager
finds the file wherever it lives.

Usage:
    from domains.shell.file_manager import get_file_manager
    fm = get_file_manager()
    content = fm.read_text("/data/shakespeare.txt")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("slo.shell.file_manager")


class FileManager:
    """Unified path resolution: VFS mounts → host filesystem."""

    def __init__(self):
        self._vfs = None  # lazy import to avoid circular deps

    def _get_vfs(self):
        if self._vfs is None:
            try:
                from domains.shell.vfs import get_vfs
                self._vfs = get_vfs()
            except ImportError:
                self._vfs = False  # sentinel: VFS unavailable
        return self._vfs if self._vfs is not False else None

    def read_text(self, path: str) -> Optional[str]:
        """Read text from path. Tries VFS first, then host FS.

        Args:
            path: file path (absolute or relative)

        Returns:
            file content as string, or None if not found
        """
        # 1. Try VFS
        vfs = self._get_vfs()
        if vfs is not None:
            content = vfs.read(path)
            if content is not None:
                return content

        # 2. Fallback to host filesystem
        expanded = os.path.expanduser(path)
        try:
            return Path(expanded).read_text()
        except (OSError, PermissionError):
            logger.debug("file not found: %s", path)
            return None

    def read_bytes(self, path: str) -> Optional[bytes]:
        """Read binary content from path.

        Args:
            path: file path

        Returns:
            file content as bytes, or None if not found
        """
        # VFS only supports text, so skip to host FS for bytes
        expanded = os.path.expanduser(path)
        try:
            return Path(expanded).read_bytes()
        except (OSError, PermissionError):
            return None

    def write_text(self, path: str, data: str) -> Optional[str]:
        """Write text to path. Tries VFS first, then host FS.

        Args:
            path: file path
            data: text content to write

        Returns:
            None on success, error message on failure
        """
        # 1. Try VFS
        vfs = self._get_vfs()
        if vfs is not None:
            result = vfs.write(path, data)
            if result is None:
                return None

        # 2. Fallback to host filesystem
        expanded = os.path.expanduser(path)
        try:
            os.makedirs(os.path.dirname(expanded), exist_ok=True)
            Path(expanded).write_text(data)
            return None
        except (OSError, PermissionError) as e:
            return str(e)

    def exists(self, path: str) -> bool:
        """Check if path exists. VFS first, then host FS."""
        vfs = self._get_vfs()
        if vfs is not None and vfs.exists(path):
            return True
        return os.path.exists(os.path.expanduser(path))

    def isfile(self, path: str) -> bool:
        """Check if path is a file."""
        vfs = self._get_vfs()
        if vfs is not None:
            from domains.shell.vfs import VFS
            if isinstance(vfs, VFS) and vfs.isfile(path):
                return True
        return os.path.isfile(os.path.expanduser(path))

    def isdir(self, path: str) -> bool:
        """Check if path is a directory."""
        vfs = self._get_vfs()
        if vfs is not None:
            from domains.shell.vfs import VFS
            if isinstance(vfs, VFS) and vfs.isdir(path):
                return True
        return os.path.isdir(os.path.expanduser(path))

    def listdir(self, path: str) -> Optional[list[str]]:
        """List directory contents. VFS first, then host FS.

        Returns:
            sorted list of entry names, or None if not found
        """
        vfs = self._get_vfs()
        if vfs is not None:
            entries = vfs.listdir(path)
            if entries is not None:
                return entries
        try:
            return sorted(os.listdir(os.path.expanduser(path)))
        except (OSError, PermissionError):
            return None

    def resolve(self, path: str) -> Optional[str]:
        """Resolve to an absolute path. Returns None if not found anywhere.

        For VFS paths, returns the original path.
        For host FS paths, returns the expanded absolute path.
        """
        vfs = self._get_vfs()
        if vfs is not None and vfs.exists(path):
            return path
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return os.path.abspath(expanded)
        return None


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Get or create the global FileManager singleton."""
    global _instance
    if _instance is None:
        _instance = FileManager()
    return _instance


def reset_file_manager() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
