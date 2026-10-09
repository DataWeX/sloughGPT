"""Packaging regression: ``setup.py egg_info`` must succeed from the repo root.

Card 56cf354c — the domain consolidation moved ``domain/`` to the repo root
while ``package_dir`` still resolved it under ``packages/core-py``:

    $ python setup.py egg_info --egg-base /tmp/...
    error: package directory 'packages/core-py/domain' does not exist

That hard-fails the documented install command (``pip install -e ".[dev]"``
per packages/core-py/README.md). Discovery (``packages.find``) and resolution
(``package_dir``) must agree on where every published root lives.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_egg_info_succeeds(tmp_path: Path) -> None:
    """egg_info exits 0 and publishes the real top-level packages."""
    proc = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"setup.py egg_info failed (exit {proc.returncode}):\n"
        f"{proc.stderr[-2000:] or proc.stdout[-2000:]}"
    )

    egg_info_dirs = list(tmp_path.glob("*.egg-info"))
    assert egg_info_dirs, f"no .egg-info produced under {tmp_path}"
    top_level = (egg_info_dirs[0] / "top_level.txt").read_text().split()

    # The published roots after the consolidation: domain lives at the repo
    # root, the CLI under apps/. tests/ must never be published.
    assert "domain" in top_level, f"domain not published: {top_level}"
    assert "apps" in top_level, f"apps not published: {top_level}"
    assert "tests" not in top_level, f"tests leaked into the distribution: {top_level}"
