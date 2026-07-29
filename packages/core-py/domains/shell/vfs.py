"""
Dait Virtual File System — unified I/O abstraction layer.

This module is a thin re-export layer. The canonical implementation lives
in ``addons.filesystem``. This module exists so that legacy imports like
``from domains.shell.vfs import VFS`` continue to work.
"""

from __future__ import annotations

from typing import Optional

# Canonical implementations from the filesystem addon
from .addons.filesystem import (  # noqa: F401
    VFS,
    VFSEntry,
    VFSDirectory,
    VFSGeneratedFile,
    VFSWriteOnlyFile,
    _dir_stat,
    _file_stat,
)


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_vfs_instance: Optional[VFS] = None


def get_vfs() -> VFS:
    global _vfs_instance
    if _vfs_instance is None:
        _vfs_instance = VFS()
    return _vfs_instance


def reset_vfs() -> None:
    global _vfs_instance
    _vfs_instance = None
