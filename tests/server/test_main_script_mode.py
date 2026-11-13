"""Script-mode entry must import without ``PYTHONPATH`` — the slough-api contract.

``apps/api/server/main.py`` bootstraps ``sys.path`` from the repo root, but it
used to import ``domain.shared`` *before* that bootstrap. Under script mode
(``python apps/api/server/main.py`` — exactly what ``slough-api.service``'s
ExecStart runs) only ``apps/api/server`` lands on ``sys.path``, so the import
resolved only while the editable install still mapped the plural ``domains``
tree. The plural→singular migration (8e1a0c75c) removed that mapping and the
shared :8000 server crash-looped ~1800× with
``ModuleNotFoundError: No module named 'domain'``.

``--help`` executes the entire import chain and exits through argparse before
uvicorn binds, so it is a cheap, side-effect-free regression probe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script_mode(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the entry point with ``PYTHONPATH`` stripped (the unit sets none)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "apps" / "api" / "server" / "main.py"), "--help"],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_script_mode_entry_imports_domain_without_pythonpath() -> None:
    """Repo-root cwd (the unit's WorkingDirectory) must boot the import chain."""
    proc = _run_script_mode(REPO_ROOT)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "ModuleNotFoundError" not in proc.stderr
    assert "usage:" in proc.stdout


def test_script_mode_entry_is_cwd_independent() -> None:
    """No path may depend on the process cwd — script mode never adds it."""
    proc = _run_script_mode(Path(os.sep))
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "ModuleNotFoundError" not in proc.stderr
    assert "usage:" in proc.stdout
