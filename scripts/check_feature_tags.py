#!/usr/bin/env python3
"""
Check for FEATURE tags in Python files and protect them from deletion.

Scans the codebase for files tagged with "FEATURE:" in their docstring.
Prevents deletion of tagged files unless explicitly overridden.

Usage:
    python scripts/check_feature_tags.py [--check-deletions] [--list] [--validate]

Features are tagged with docstring comments like:
    FEATURE: my-feature-name — Description of what it does.

Exit codes:
    0 = all checks passed
    1 = feature-tagged file deleted or validation failed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Directory names the scan must never descend into. `scripts/` is excluded by
# design: this guard does not protect itself.
_SKIP_DIR_NAMES = frozenset({"__pycache__", "node_modules", "scripts"})


def _iter_candidate_files(root: Path):
    """Yield the Python files the FEATURE scan should consider.

    Tracked files only, enumerated with `git ls-files`. The previous
    implementation used `root.rglob("*.py")`, which walked every `.wt-*` git
    worktree living inside the repo root — each a full checkout — and counted
    every tagged file once per worktree. With 68 worktrees present that
    reported 701 tagged files where 11 are actually protected (card 3dfbbb90).

    Tracking-only is not merely cheaper; it is the correct scope for this
    guard. It protects files under version control, and the worktree copies
    are untracked, so they vanish with their worktree and can never be
    protected. Counting them made the guard report green while the one file it
    existed to protect could be deleted. Enumerating the index also makes
    worktrees invisible by construction, so no path-prefix exclusion list can
    fall behind the set of worktrees that happen to exist.

    A plain (non-git) directory falls back to a filesystem walk that still
    skips worktrees and caches, because callers may point this at any root.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            capture_output=True,
            text=True,
            cwd=str(root),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        proc = None

    if proc is not None:
        for rel in proc.stdout.split("\0"):
            if not rel:
                continue
            # Same exclusions as the fallback — scripts/ in particular, since
            # this guard does not protect itself.
            if any(p in _SKIP_DIR_NAMES for p in Path(rel).parts):
                continue
            yield root / rel
        return

    for py_file in root.rglob("*.py"):
        parts = py_file.relative_to(root).parts
        if any(p in _SKIP_DIR_NAMES or p.startswith(".wt-") for p in parts):
            continue
        yield py_file


def find_feature_tagged_files(root: Path) -> list[tuple[Path, str, str]]:
    """Find all files with FEATURE tags in their docstrings.

    Returns list of (filepath, feature_name, description) tuples.
    """
    results = []
    pattern = re.compile(r"FEATURE:\s*([a-z][a-z\-]+)\s+—\s+(.+?)(?:\n|$)")

    for py_file in _iter_candidate_files(root):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            # Look for FEATURE: name — description pattern
            match = pattern.search(content)
            if match:
                feature_name = match.group(1)
                description = match.group(2).strip().rstrip(".")
                results.append((py_file, feature_name, description))
        except Exception:
            continue

    return results


def check_deletions(root: Path) -> list[str]:
    """Check if any FEATURE-tagged files have been deleted (git diff)."""
    errors = []

    # Get deleted files from git
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "--diff-filter=D", "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        deleted = [
            line.split("\t")[1]
            for line in result.stdout.strip().split("\n")
            if line.startswith("D\t")
        ]
    except Exception:
        return errors

    # Check if any deleted files had FEATURE tags
    for deleted_file in deleted:
        file_path = root / deleted_file
        if not file_path.exists():
            # File was deleted - check if it had a FEATURE tag in the previous commit
            try:
                prev_content = subprocess.run(
                    ["git", "show", f"HEAD~1:{deleted_file}"],
                    capture_output=True,
                    text=True,
                    cwd=str(root),
                )
                if "FEATURE:" in prev_content.stdout:
                    match = re.search(r"FEATURE:\s*([\w_]+)", prev_content.stdout)
                    if match:
                        errors.append(
                            f"DELETED FEATURE: {deleted_file} (feature: {match.group(1)})"
                        )
            except Exception:
                pass

    return errors


def list_features(root: Path) -> None:
    """List all feature-tagged files."""
    files = find_feature_tagged_files(root)

    if not files:
        print("No FEATURE-tagged files found.")
        return

    print(f"\n{'Feature':<25} {'File':<60} {'Description'}")
    print("-" * 120)

    for filepath, feature_name, description in sorted(files, key=lambda x: x[1]):
        rel_path = filepath.relative_to(root)
        print(f"{feature_name:<25} {str(rel_path):<60} {description}")

    print(f"\nTotal: {len(files)} feature-tagged files")


def validate_feature_flags(root: Path) -> list[str]:
    """Validate that all FEATURE tags have corresponding feature flag entries."""
    # The _internal import is deliberate, not a shortcut: FeatureFlags is not
    # part of domain.shared's declared facade API (see domain/shared/__init__
    # __all__), and scripts already import domain.shell._internal.* directly.
    # The old pre-refactor path `domain.shared.feature_flags` raised
    # ModuleNotFoundError here, which took down the whole --validate run.
    from domain.shared._internal.feature_flags import FeatureFlags

    errors = []
    files = find_feature_tagged_files(root)
    all_flags = FeatureFlags.list_all()

    for filepath, feature_name, _description in files:
        # Normalize: tags use hyphens, flags use underscores
        normalized = feature_name.replace("-", "_")
        if normalized not in all_flags:
            errors.append(
                f"MISSING FLAG: {feature_name} → {normalized} (tagged in {filepath.relative_to(root)})"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check FEATURE tags")
    parser.add_argument(
        "--check-deletions", action="store_true", help="Check for deleted feature files"
    )
    parser.add_argument("--list", action="store_true", help="List all feature-tagged files")
    parser.add_argument("--validate", action="store_true", help="Validate feature flags match tags")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).parent.parent, help="Root directory"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    errors = []

    if args.list:
        list_features(root)
        return 0

    if args.check_deletions:
        deletion_errors = check_deletions(root)
        errors.extend(deletion_errors)

    if args.validate:
        validation_errors = validate_feature_flags(root)
        errors.extend(validation_errors)

    if not any([args.check_deletions, args.validate]):
        # Default: list features
        list_features(root)
        return 0

    if errors:
        print("\nERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  ✗ {err}", file=sys.stderr)
        return 1

    print("All feature tag checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
