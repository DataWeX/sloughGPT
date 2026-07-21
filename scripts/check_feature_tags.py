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

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_feature_tagged_files(root: Path) -> List[Tuple[Path, str, str]]:
    """Find all files with FEATURE tags in their docstrings.

    Returns list of (filepath, feature_name, description) tuples.
    """
    results = []
    pattern = re.compile(r"FEATURE:\s*([a-z][a-z\-]+)\s+—\s+(.+?)(?:\n|$)")

    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file) or "node_modules" in str(py_file) or "scripts/" in str(py_file):
            continue
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


def check_deletions(root: Path) -> List[str]:
    """Check if any FEATURE-tagged files have been deleted (git diff)."""
    import subprocess

    errors = []

    # Get deleted files from git
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "--diff-filter=D", "HEAD~1", "HEAD"],
            capture_output=True, text=True, cwd=str(root)
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
                    capture_output=True, text=True, cwd=str(root)
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


def validate_feature_flags(root: Path) -> List[str]:
    """Validate that all FEATURE tags have corresponding feature flag entries."""
    from domains.shared.feature_flags import FeatureFlags

    errors = []
    files = find_feature_tagged_files(root)
    all_flags = FeatureFlags.list_all()

    for filepath, feature_name, description in files:
        # Normalize: tags use hyphens, flags use underscores
        normalized = feature_name.replace("-", "_")
        if normalized not in all_flags:
            errors.append(
                f"MISSING FLAG: {feature_name} → {normalized} (tagged in {filepath.relative_to(root)})"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Check FEATURE tags")
    parser.add_argument("--check-deletions", action="store_true", help="Check for deleted feature files")
    parser.add_argument("--list", action="store_true", help="List all feature-tagged files")
    parser.add_argument("--validate", action="store_true", help="Validate feature flags match tags")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent.parent, help="Root directory")
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
