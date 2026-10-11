"""Guard tests for scripts/check_feature_tags.py.

This script protects FEATURE-tagged files from deletion, so the number it
reports has to be the number of files it actually protects. It previously
enumerated candidates with ``root.rglob("*.py")``, which walked every ``.wt-*``
git worktree living inside the repo root — each a full checkout — and counted
every tagged file once per worktree. Measured 2026-10-11 with 68 worktrees
present: it reported **701** tagged files where **11** are protected (card
3dfbbb90).

That is the failure mode worth guarding against here. A deletion guard that
counts untracked worktree copies as if they were distinct protected files
reports green while the one tracked file it exists to protect is deleted — the
copies are untracked and vanish with their worktree. Reporting a plausible but
wrong number is worse than reporting nothing, because it invites trust.

The scan must therefore enumerate *tracked* files only (``git ls-files``),
which makes worktrees invisible by construction.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The script skips anything under scripts/ — by design, it does not protect
# itself. One of the 12 tracked tagged files is scripts/check_feature_tags.py,
# so the number the script should report is 11, not 12.
SCRIPTS_PREFIX = "scripts/"

# Assembled rather than written literally, so this test module does not itself
# match the FEATURE regex and inflate the very count the tests assert on.
_DASH = "\u2014"
TAG_DOCSTRING = f'"""Demo module.\n\nFEATURE: widget-demo {_DASH} A demo widget.\n"""\n'


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "check_feature_tags", REPO / "scripts" / "check_feature_tags.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _git_tracked_tagged() -> list[str]:
    """The ground truth the script is expected to agree with."""
    out = subprocess.run(
        ["git", "grep", "-lE", r"FEATURE:\s*[a-z][a-z-]+\s+—", "--", "*.py"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    ).stdout
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip() and not line.startswith(SCRIPTS_PREFIX)
    ]


def test_scan_reports_no_worktree_copies(script):
    """No .wt-* path may appear, regardless of how many worktrees exist."""
    found = script.find_feature_tagged_files(REPO)
    stray = [str(p) for p, _, _ in found if ".wt-" in p.parts]
    assert stray == [], f"worktree copies leaked into the scan: {stray[:3]}"


def test_count_matches_the_protected_tracked_files(script):
    """The reported set must equal the tracked files the guard protects.

    Set equality, not just count: a count can stay level while the wrong file
    is counted (e.g. the scripts/ exclusion diverging between oracle and
    scan), and that is exactly the kind of quiet drift this file exists to
    catch.
    """
    found = {str(p.relative_to(REPO)) for p, _, _ in script.find_feature_tagged_files(REPO)}
    expected = set(_git_tracked_tagged())
    assert expected, "ground truth unexpectedly empty — is the repo checked out?"
    assert found == expected, (
        f"scan/git drift — only in scan: {sorted(found - expected)[:3]}; "
        f"only in git: {sorted(expected - found)[:3]}"
    )


def test_each_reported_file_is_tracked_by_git(script):
    """Every reported path must exist in the index, never only on disk."""
    found = script.find_feature_tagged_files(REPO)
    tracked = {
        line.strip()
        for line in subprocess.run(
            ["git", "ls-files", "-z", "*.py"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            check=True,
        ).stdout.split("\0")
        if line.strip()
    }
    untracked = [
        str(p.relative_to(REPO)) for p, _, _ in found if str(p.relative_to(REPO)) not in tracked
    ]
    assert untracked == [], f"reported untracked files as protected: {untracked[:3]}"


def test_plain_directory_falls_back_and_still_skips_worktrees(script, tmp_path):
    """Outside a git repo the scan degrades gracefully, but worktrees still lose.

    The fallback matters because find_feature_tagged_files() takes an arbitrary
    root; tests and ad-hoc runs point it at directories that are not checkouts.
    """
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "widget.py").write_text(TAG_DOCSTRING)
    (tmp_path / ".wt-copy").mkdir()
    (tmp_path / ".wt-copy" / "widget.py").write_text(TAG_DOCSTRING)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text(TAG_DOCSTRING)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text(TAG_DOCSTRING)

    found = script.find_feature_tagged_files(tmp_path)

    names = sorted(str(p.relative_to(tmp_path)) for p, _, _ in found)
    assert names == ["real/widget.py"], (
        "fallback must skip .wt-*, __pycache__ and scripts/; got {names}"
    )


def test_tracked_checkout_ignores_untracked_worktree_copies(script, tmp_path):
    """On the primary (git ls-files) path, .wt-* copies must be invisible.

    The original defect in miniature: a real checkout holding one tracked
    tagged file plus an untracked duplicate under a .wt-* directory (card
    3dfbbb90). The old rglob counted 2; the guard must report 1. The fallback
    test above covers the non-git path only, and the repo-level no-.wt-
    assertion is vacuous in a fresh CI clone that ships no worktrees — this
    constructs the case everywhere.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "widget.py").write_text(TAG_DOCSTRING)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "widget.py"],
        check=True,
        capture_output=True,
    )
    copy_dir = tmp_path / ".wt-copy"
    copy_dir.mkdir()
    (copy_dir / "widget.py").write_text(TAG_DOCSTRING)

    found = script.find_feature_tagged_files(tmp_path)

    names = sorted(str(p.relative_to(tmp_path)) for p, _, _ in found)
    assert names == ["widget.py"], f"untracked worktree copy leaked into a git-backed scan: {names}"


def test_tagged_file_is_recognised_at_all(script, tmp_path):
    """Sanity: the regex still matches the documented tag format."""
    (tmp_path / "only.py").write_text(TAG_DOCSTRING)
    found = script.find_feature_tagged_files(tmp_path)
    assert len(found) == 1
    path, feature, description = found[0]
    assert path.name == "only.py"
    assert feature == "widget-demo"
    assert description == "A demo widget"
