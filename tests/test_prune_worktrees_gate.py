"""Regression gate for scripts/prune-worktrees.sh — the fail-closed liveness contract.

Why this exists: the liveness gate historically returned "no owner" both when
nobody lived in a worktree AND when `/proc` could not be read at all. Those are
different answers. Collapsing them means an unattended run in a restricted
environment reports `live 0` for every tree and prunes live worktrees — the
2026-10-21 failure class (name-pattern sweep wiped ~60 of them) reached through
a different door.

The contract asserted here:
  * degraded /proc  -> exit 3, before any verdict is printed (fail CLOSED)
  * healthy /proc   -> runs, exit 0
  * --report pins dry-run and is mutually exclusive with --apply
  * --help prints only the doc block (a hard-coded line range leaked code once)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prune-worktrees.sh"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )


def _run_variant(args: list[str], proc_root: Path) -> subprocess.CompletedProcess[str]:
    """Run the script with every `/proc` reference pointed at *proc_root*."""
    src = SCRIPT.read_text(encoding="utf-8")
    variant = proc_root.parent / "prune_variant.sh"
    variant.write_text(src.replace("/proc", str(proc_root)), encoding="utf-8")
    return subprocess.run(
        ["bash", str(variant), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_dry_run_exits_zero(tmp_path: Path) -> None:
    """A missing target is a verdict, not a failure — dry runs stay exit 0."""
    res = _run(["--report", "--path", str(tmp_path / "nope")])
    assert res.returncode == 0, res.stderr
    assert "MISSING" in res.stdout


@pytest.mark.parametrize(
    "case",
    ["absent", "empty", "denied"],
)
def test_fail_closed_when_proc_unusable(tmp_path: Path, case: str) -> None:
    """Exit 3, never a green `live 0`, whenever liveness cannot be judged."""
    proc_root = tmp_path / "fakeproc"
    if case == "absent":
        pass  # never created -> `[ -d /proc ]` fails
    elif case == "empty":
        proc_root.mkdir()
    else:
        # pid dirs exist but no cwd can be read (EACCES analogue)
        for pid in ("1", "2"):
            d = proc_root / pid
            d.mkdir(parents=True)
            cwd = d / "cwd"
            cwd.write_text("", encoding="utf-8")
            cwd.chmod(0o000)

    res = _run_variant(["--report"], proc_root)
    assert res.returncode == 3, (case, res.returncode, res.stdout, res.stderr)
    # It must abort BEFORE emitting verdicts, not print PRUNE then die.
    assert "PRUNE" not in res.stdout
    assert "liveness" in res.stderr.lower()


def test_positive_control_real_proc_not_trip() -> None:
    """The preflight must not fire on a healthy host — otherwise it blocks all use."""
    res = _run(["--report", "--path", "/nonexistent-for-this-test"])
    assert res.returncode == 0, res.stderr
    assert "fail closed" not in res.stderr


@pytest.mark.parametrize(
    "args",
    [["--report", "--apply"], ["--apply", "--report"]],
)
def test_report_and_apply_are_mutually_exclusive(args: list[str]) -> None:
    """Automation pins --report; it can never be combined into an apply run."""
    res = _run(args)
    assert res.returncode == 1, res.stderr
    assert "mutually exclusive" in res.stderr


def test_help_prints_docs_only() -> None:
    """--help must never leak code (the old hard-coded `sed 2,45p` range did)."""
    res = _run(["--help"])
    assert res.returncode == 0
    assert "report" in res.stdout
    # Shell statements must not appear in help output.
    leaked = [
        line
        for line in res.stdout.splitlines()
        if line.strip().startswith(("for ", "if ", "local ", "while ", "DST="))
    ]
    assert leaked == [], leaked


def test_help_documents_fail_closed_exit() -> None:
    res = _run(["--help"])
    assert "3 = FAIL CLOSED" in res.stdout
