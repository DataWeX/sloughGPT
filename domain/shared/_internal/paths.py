"""Repository/interpreter discovery helpers."""

from __future__ import annotations

import os as _os
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


def _interpreter_candidates(root: Path) -> list[Path]:
    """Candidate interpreters, most preferred first.

    Mirrors ``scripts/python`` so shell, npm, and Python entry points agree:
    ``SLO_PYTHON`` override → conda project env → ``.venv`` → PATH.
    """
    out: list[Path] = []
    if override := _os.environ.get("SLO_PYTHON"):
        out.append(Path(override))
    if _os.environ.get("CONDA_DEFAULT_ENV") == "sloughgpt":
        if prefix := _os.environ.get("CONDA_PREFIX"):
            out.append(Path(prefix) / "bin" / "python")
    home = Path.home()
    for base in ("miniconda3", "anaconda3", "mambaforge"):
        out.append(home / base / "envs" / "sloughgpt" / "bin" / "python")
    out.append(Path("/opt/conda/envs/sloughgpt/bin/python"))
    out.extend([root / ".venv" / "bin" / "python3", root / ".venv" / "bin" / "python"])
    return out


def find_server_python(repo_root: Path | str = "") -> str:
    """Find the Python executable with the project's dependencies.

    Checks the conda project env (``sloughgpt``) first, then
    ``<repo_root>/.venv/bin/python3``, ``.venv/bin/python``, and finally
    falls back to the currently running interpreter.  Same order as
    ``scripts/python``.
    """
    root = Path(repo_root) if repo_root else find_repo_root()
    for cand in _interpreter_candidates(root):
        try:
            if cand.is_file() and _os.access(cand, _os.X_OK):
                return str(cand)
        except OSError:  # pragma: no cover - unreadable path
            continue
    return _sys.executable or "python3"
