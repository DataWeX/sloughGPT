"""CI gate for ``scripts/check_surface_scope.py`` — the Surface Rule's advisory.

The rule (AGENTS.md "Surface Rule"): one change unit = one shipping surface,
because surfaces ship on different cadences and a branch spanning N of them
accumulates drift from all N — conflict surface = product of the cadences,
not the sum. Evidence: PR 258 spanned 341 files / 12 surfaces and 187 files
conflicted into unlandability.

Exit contract: ALWAYS 0 (rule clause 5 — advisory first, ratchet later).
The value is in the surface list it prints, which the test asserts rather
than re-implementing ``classify`` twice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_surface_scope.py"

# PR 258 (fix/audit-a11y-design-tokens) — the change unit that motivated
# the rule. Range: merge-base with main -> branch tip.
PR258_BASE = "59d5c869b"
PR258_TIP = "66dd32967"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def test_checker_exists() -> None:
    assert CHECKER.is_file(), f"missing {CHECKER.relative_to(ROOT)}"


def test_advisory_never_fails_on_pr258_range() -> None:
    """The worst real-world case must still exit 0 — advisory, not a wall."""
    proc = _run(PR258_BASE, PR258_TIP)
    if "not a valid object" in proc.stderr or proc.returncode == 128:
        return  # PR refs were pruned; nothing to check
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pr258_range_reports_multiple_surfaces() -> None:
    """The motivating case must be flagged as multi-surface (it was 12)."""
    proc = _run(PR258_BASE, PR258_TIP)
    if proc.returncode == 128:
        return
    out = proc.stdout
    assert "ADVISORY" in out, f"PR 258's 341-file diff must flag:\n{out}"
    # apps/web, apps/api, apps/mobile must be distinct cadences, not one "apps/"
    assert "apps/web/" in out
    assert "apps/api/" in out
    assert "apps/mobile/" in out


def test_docs_only_change_is_clean() -> None:
    """A docs/config-only diff must report no surface (exception clause)."""
    # classify() is our own logic — direct unit checks, no second copy.
    sys.path.insert(0, str(CHECKER.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("_surface_scope", CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for path in ("docs/PRODUCT_ENGINEERING.md", "AGENTS.md", "README.md", ".husky/pre-commit"):
        assert mod.classify(path) is None, f"{path} must ride along"
    for path in (
        ".dev-notes/x.md",
        ".kanban/board.jsonl",
        ".planner/state.json",
    ):
        assert mod.classify(path) is None, f"{path} is canonical store, never counted"

    assert mod.classify("apps/web/lib/foo.ts") == "apps/web/"
    assert mod.classify("apps/api/server/main.py") == "apps/api/"
    assert mod.classify("apps/mobile/src/App.tsx") == "apps/mobile/"
    assert mod.classify("domain/core/_internal/x.py") == "domain/"
    assert mod.classify("packages/grib/src/grib/cli.py") == "packages/grib"
    assert mod.classify("packages/sdk-ts/src/index.ts") == "packages/sdk-ts"


def test_no_range_prints_usage_contract() -> None:
    """Bad argv → doc + exit 2; never a traceback."""
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--only-one-arg"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
