#!/usr/bin/env python3
"""SloughGPT CLI entry point.

Resolves the project interpreter, sets up sys.path, and runs the CLI.
Works standalone (python3 cli.py) or via console_scripts (pip install -e .).

Interpreter resolution order (first hit wins):

  1. ``SLOUGHGPT_PYTHON``   — explicit override
  2. ``.venv/bin/python``    — local venv, when one exists
  3. conda env ``sloughgpt`` — the pinned project env (AGENTS.md)

Missing candidates are skipped silently: this entry point must never
hard-fail just because the preferred interpreter is absent (CI, foreign
machines, fresh worktrees).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# Conda env names we treat as the project env, most preferred first.
_PINNED_CONDA_ENVS = ("sloughgpt",)

# Set immediately before execv so a bad candidate can never re-exec twice.
_REEXEC_SENTINEL = "_SLOUGHGPT_REEXEC"


# ── Interpreter resolution ─────────────────────────────────────────────────────
def _conda_roots() -> list[Path]:
    """Locate conda installation roots to search for pinned envs."""
    roots: list[Path] = []

    # Authoritative when present: <root>/bin/conda or <root>/condabin/conda
    exe = os.environ.get("CONDA_EXE")
    if exe:
        p = Path(exe).resolve()
        for candidate in (p.parent.parent, p.parent):
            if (candidate / "envs").is_dir():
                roots.append(candidate)
                break

    # Derive from the interpreter we are already running.
    me = Path(sys.executable).resolve()
    parts = me.parts
    if "envs" in parts:
        # <root>/envs/<name>/bin/python -> <root>
        roots.append(Path(*parts[: parts.index("envs")]))
    elif me.parent.name == "bin":
        # <root>/bin/python -> <root>
        roots.append(me.parent.parent)

    # Well-known fallbacks, for shells where CONDA_EXE is not exported.
    roots.extend(
        [
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
            Path.home() / "mambaforge",
            Path("/opt/conda"),
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen and (root / "envs").is_dir():
            seen.add(key)
            unique.append(root)
    return unique


def _candidate_interpreters() -> list[Path]:
    """Interpreter candidates in priority order."""
    candidates: list[Path] = []

    explicit = os.environ.get("SLOUGHGPT_PYTHON")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(_REPO / ".venv" / "bin" / "python")

    for root in _conda_roots():
        for env_name in _PINNED_CONDA_ENVS:
            candidates.append(root / "envs" / env_name / "bin" / "python")

    return candidates


def _reexec_target() -> Path | None:
    """Return the interpreter to exec into, or None to keep the current one."""
    if os.environ.get(_REEXEC_SENTINEL) == "1":
        return None  # already switched once; never loop

    current = Path(sys.executable).resolve()
    for candidate in _candidate_interpreters():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        if candidate.resolve() == current:
            return None  # we are already on the best interpreter
        return candidate
    return None


# ── Project interpreter switch ─────────────────────────────────────────────────
if __name__ == "__main__":
    _target = _reexec_target()
    if _target is not None:
        _env = os.environ.copy()
        _env[_REEXEC_SENTINEL] = "1"
        os.execve(str(_target), [str(_target)] + sys.argv, _env)

# ── Path setup ─────────────────────────────────────────────────────────────────
_CORE_PY = _REPO / "packages" / "core-py"
_CLI_SRC = _REPO / "apps" / "cli" / "src"
# mogdb is source-only (never pip-installed — see startup_history._import_mogdb),
# so the composition root must declare it, exactly as pytest.ini pythonpath does.
_MOGDB_SRC = _REPO / "packages" / "mogdb" / "src"
for _p in [_CLI_SRC, _CORE_PY, _MOGDB_SRC]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apps.cli.src.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
