"""CI gate for ``scripts/check_dockerignore.py`` — every COPY source must survive.

The checker itself predates this test and lives in ``scripts/``; it was
orphaned (referenced only by its own docstring and a docs entry), so nothing
ran it in CI. The failure mode it exists to catch is expensive: card 0464c3d4,
where root ``.dockerignore`` gained blanket excludes for ``packages/strui/``,
``apps/web/`` and ``apps/cli/`` — the very directories the Dockerfiles COPY out
of the build context — and Docker stripped them, so ``build`` went red for 60
consecutive runs before anyone audited the COPY sources.

This test only invokes the script (no reimplementation of Docker's
patternmatcher — that would be a second, drifting copy of the same semantics)
and asserts the exit contract: 0 = all COPY sources reachable, 1 = at least one
excluded source.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_dockerignore.py"


def _run_checker() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(ROOT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def test_checker_exists() -> None:
    """The gate must ship — a missing checker is a silent loss of coverage."""
    assert CHECKER.is_file(), f"missing {CHECKER.relative_to(ROOT)}"


def test_every_copy_source_survives_dockerignore() -> None:
    """Exit 0: no Dockerfile COPY source is excluded from the build context."""
    proc = _run_checker()
    assert proc.returncode == 0, (
        "check_dockerignore.py reported excluded COPY source(s):\n"
        f"{proc.stdout}{proc.stderr}"
    )


def test_checker_actually_inspected_sources() -> None:
    """Guard against a vacuous pass (no Dockerfiles found == nothing checked).

    Without this, deleting or renaming every Dockerfile would make the gate
    pass forever while protecting nothing.
    """
    proc = _run_checker()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # "checked 79 COPY sources across 12 Dockerfiles"
    assert "checked" in proc.stdout, proc.stdout
    count = int(proc.stdout.split("checked", 1)[1].split()[0])
    assert count > 0, f"checker inspected {count} COPY sources — vacuous pass"


def test_rejects_a_dockerignore_that_excludes_a_copy_source(tmp_path: Path) -> None:
    """RED-proven non-vacuity: the checker must fail on the original defect.

    Rebuilds the exact 0464c3d4 regression (blanket dir excludes nuking COPY
    sources) in a throwaway tree and asserts exit 1.
    """
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nCOPY packages/strui/ /app/strui\n", encoding="utf-8"
    )
    (tmp_path / "packages" / "strui").mkdir(parents=True)
    (tmp_path / ".dockerignore").write_text(
        "packages/strui/\napps/web/\n", encoding="utf-8"
    )

    proc = subprocess.run(
        [sys.executable, str(CHECKER), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1, (
        "checker passed on a .dockerignore that excludes a COPY source:\n"
        f"{proc.stdout}{proc.stderr}"
    )
    assert "packages/strui" in proc.stdout, proc.stdout
