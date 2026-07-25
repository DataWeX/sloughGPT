#!/usr/bin/env python3
"""SloughGPT CLI entry point.

Auto-detects .venv, sets up sys.path, and runs the CLI.
Works standalone (python3 cli.py) or via console_scripts (pip install -e .).
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# ── Venv auto-detection ─────────────────────────────────────────────────
_venv_py = _REPO / ".venv" / "bin" / "python"
if _venv_py.exists() and sys.executable != str(_venv_py):
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

# ── Path setup ──────────────────────────────────────────────────────────
_CORE_PY = _REPO / "packages" / "core-py"
_CLI_SRC = _REPO / "apps" / "cli" / "src"
for _p in [_CLI_SRC, _CORE_PY]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apps.cli.src.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
