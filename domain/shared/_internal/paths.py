"""Repository/interpreter discovery helpers."""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path

__all__ = ["data_root", "find_repo_root", "find_server_python"]


def find_repo_root(start: Path | str = "") -> Path:
    """Walk up from *start* (or this file) to find the repository root.

    The root is identified by having both ``apps/`` and ``packages/``
    subdirectories.  Falls back to ``pyproject.toml`` + ``apps/``.
    """
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "apps").is_dir() and (parent / "packages").is_dir():
            return parent
        if (parent / "pyproject.toml").exists() and (parent / "apps").is_dir():
            return parent
    return here.parents[min(4, len(here.parents) - 1)]


def data_root() -> Path:
    """Canonical data directory — ``<repo>/data`` unless redirected.

    ``SLO_DATA_DIR`` (same shape as the existing ``SLO_CACHE_DIR``) points the
    whole data tree at a throw-away location.  Tests set it so their fixtures
    land in a tmp dir instead of accumulating in the live ``data/`` tree, which
    measured 1033 files under ``data/agents/`` of which 916 matched pytest id
    patterns (``auto-*``/``create-*``/``dup-*``/``upd-*``/``tools-*``).

    Reads the environment on every call and is never cached: the root conftest
    redirects it, so a value captured at import time would miss the override.

    Domain-side twin of ``infrastructure.db_pool._data_root()`` — identical
    semantics, but resolved through :func:`find_repo_root` so ``domain`` never
    imports the api server package.
    """
    override = _os.environ.get("SLO_DATA_DIR", "").strip()
    return Path(override) if override else find_repo_root() / "data"


def find_server_python(repo_root: Path | str = "") -> str:
    """Find the Python executable with the project's dependencies.

    Checks ``<repo_root>/.venv/bin/python3`` first, then ``.venv/bin/python``,
    then falls back to the currently running interpreter.
    """
    root = Path(repo_root) if repo_root else find_repo_root()
    for name in (".venv/bin/python3", ".venv/bin/python"):
        venv_py = root / name
        if venv_py.is_file():
            return str(venv_py)
    return _sys.executable or "python3"
