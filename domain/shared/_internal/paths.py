"""Repository/interpreter discovery helpers."""

from __future__ import annotations

import sys as _sys
from pathlib import Path

__all__ = ["find_repo_root", "find_server_python"]


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
