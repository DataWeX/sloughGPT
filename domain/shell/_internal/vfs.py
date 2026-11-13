"""
Dait Virtual File System — unified I/O abstraction layer.

This module is a thin re-export layer. The canonical implementation lives
in ``addons.filesystem``. This module exists so that legacy imports like
``from domain.shell._internal.vfs import VFS`` continue to work.
"""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .addons.filesystem import VFS

_REEXPORTS = (
    "VFS",
    "VFSDirectory",
    "VFSEntry",
    "VFSGeneratedFile",
    "VFSWriteOnlyFile",
    "_dir_stat",
    "_file_stat",
)


def __getattr__(name: str):
    """Lazily resolve re-exported names from the optional addons package.

    ``addons`` is a separately-vendored extension (historically shipped as a
    symlink).  Resolving lazily keeps this shim importable when the addons
    package is missing or broken; the ImportError surfaces only when a
    re-exported name is actually used.
    """
    if name not in _REEXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        _filesystem = _importlib.import_module(".addons.filesystem", __package__)
    except Exception as exc:
        raise ImportError(
            "filesystem addon unavailable: addons.filesystem is missing or broken"
        ) from exc
    value = getattr(_filesystem, name, None)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(_REEXPORTS))


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

_vfs_instance: VFS | None = None


def _resolve_vfs_cls():
    try:
        _filesystem = _importlib.import_module(".addons.filesystem", __package__)
    except Exception as exc:
        raise ImportError(
            "filesystem addon unavailable: addons.filesystem is missing or broken"
        ) from exc
    return _filesystem.VFS


def get_vfs() -> VFS:
    global _vfs_instance
    if _vfs_instance is None:
        _vfs_instance = _resolve_vfs_cls()()
    return _vfs_instance


def reset_vfs() -> None:
    global _vfs_instance
    _vfs_instance = None
