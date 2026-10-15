"""scripts/python must resolve imports to the working tree, not site-packages.

Card dd670ece: a Sep-18 pip snapshot of downcraft 0.2.0 in the conda env's
site-packages shadowed ``packages/downcraft`` — ``scripts/python -m downcraft``
ran stale code and the merged reget fix (6c626f043) was dead on every
scripts/python consumer (e2e, benchmarks, ad-hoc shells).

The fix lives in ``scripts/python``: it prepends the repo root plus every
python project root under ``packages/`` to PYTHONPATH. These tests run the
wrapper itself with NO inherited PYTHONPATH, so they fail on the old wrapper
and pass only when the wrapper adds the entries.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "python"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run scripts/python with a hermetic interpreter and no inherited PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["SLO_PYTHON"] = sys.executable
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )


def test_repo_downcraft_wins_over_site_packages() -> None:
    proc = _run("-c", "import downcraft; print(downcraft.__file__)")
    assert proc.returncode == 0, proc.stderr
    resolved = Path(proc.stdout.strip()).resolve()
    expected = (REPO / "packages" / "downcraft" / "downcraft" / "__init__.py").resolve()
    assert resolved == expected, f"downcraft resolved to {resolved}, expected {expected}"


def test_sys_path_carries_repo_entries_ahead_of_site_packages() -> None:
    proc = _run("-c", "import sys; print(chr(10).join(sys.path))")
    assert proc.returncode == 0, proc.stderr
    entries = [e for e in proc.stdout.splitlines() if e]
    assert str(REPO) in entries, "repo root missing from sys.path"
    assert str(REPO / "packages" / "downcraft") in entries, "downcraft project root missing"
    site_idx = [i for i, e in enumerate(entries) if "site-packages" in e]
    repo_idx = [i for i, e in enumerate(entries) if e.startswith(str(REPO))]
    assert repo_idx, "no repo entries on sys.path"
    if site_idx:
        assert max(repo_idx) < min(site_idx), "site-packages precedes repo entries"


def test_resolver_honors_slo_python() -> None:
    proc = _run("--resolve")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == sys.executable
