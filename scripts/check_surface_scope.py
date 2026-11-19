#!/usr/bin/env python3
"""Surface-scope advisory — count the shipping surfaces a diff spans.

Rule (AGENTS.md "Surface Rule"): one change unit = one shipping surface.
Surfaces ship on different cadences (web deploys per merge, mobile gets only
Dependabot, domain has its own test closure), so a branch spanning N surfaces
accumulates drift from ALL N — its conflict surface is the product of the
cadences, not the sum. Evidence: PR 258 spanned 8+ surfaces / 341 files and
187 of them conflicted into unlandability.

ADVISORY by design (rule clause 5): this prints the surface list and exits 0.
Promoting it to a failing gate is a deliberate, reviewed change, not the
default. Docs (docs/, *.md), .github/ and root config ride along with any
surface — they ship nowhere and drift with nothing.

Usage:
    scripts/check_surface_scope.py                 # branch vs merge-base
    scripts/check_surface_scope.py <base> <head>   # explicit range
    scripts/check_surface_scope.py --staged        # staged changes only
"""

from __future__ import annotations

import subprocess
import sys

# Shipping surfaces — one entry per cadence. Each packages/* dir is its own
# surface (they publish independently); everything else unmatched is "repo"
# (docs/CI/root config) and never counts against the limit.
SURFACE_PREFIXES = (
    "apps/web/",
    "apps/mobile/",
    "apps/api/",
    "domain/",
    "packages/",
)

# Never surface-counted: the canonical store lives outside the diff
# (moved 2026-10-28); PR 258's edits there were harmful as-is.
STORE_PREFIXES = (
    ".dev-notes/",
    ".kanban/",
    ".planner/",
)

# Docs/CI/root config ship nowhere — they may ride along with any surface.
RIDES_ALONG_PREFIXES = (
    "docs/",
    ".github/",
    ".husky/",
    ".opencode/",
    "scripts/",
    "tests/",  # test closures follow the surface they exercise
)


def classify(path: str) -> str | None:
    """Return the surface a path belongs to, or None if it doesn't count.

    packages/<name>/... is its own surface; other prefixes map verbatim.
    """
    if path.startswith(STORE_PREFIXES) or path.startswith(RIDES_ALONG_PREFIXES):
        return None
    if path.startswith("packages/"):
        parts = path.split("/", 2)
        return f"packages/{parts[1]}" if len(parts) > 1 else None
    if path.startswith(SURFACE_PREFIXES):
        parts = path.split("/", 2)
        # apps/<name>/... is its own surface (web/api/mobile ship separately);
        # domain/... maps to the domain surface as a whole.
        if parts[0] == "apps" and len(parts) > 1:
            return f"apps/{parts[1]}/"
        return parts[0] + "/"
    if path.endswith((".md",)) or "/" not in path:
        return None  # root-level docs/config ride along
    return None


def changed_files(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [l for l in out.stdout.splitlines() if l]


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [l for l in out.stdout.splitlines() if l]


def default_range() -> tuple[str, str]:
    """Branch tip vs its merge-base with main (or origin/main, or HEAD~1)."""
    for base in ("main", "origin/main"):
        r = subprocess.run(
            ["git", "merge-base", base, "HEAD"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip(), "HEAD"
    return "HEAD~1", "HEAD"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "--staged":
        files = staged_files()
        label = "staged changes"
    elif len(argv) >= 2:
        files = changed_files(argv[0], argv[1])
        label = f"{argv[0]}...{argv[1]}"
    elif not argv:
        base, head = default_range()
        files = changed_files(base, head)
        label = f"{base[:12]}...{head}"
    else:
        print(__doc__)
        return 2

    surfaces: dict[str, list[str]] = {}
    for path in files:
        surface = classify(path)
        if surface:
            surfaces.setdefault(surface, []).append(path)

    print(f"surface-scope advisory ({label}): {len(files)} file(s)")
    if not surfaces:
        print("  ok — no shipping surface touched (docs/config/tests only)")
        return 0

    for surface in sorted(surfaces):
        paths = surfaces[surface]
        print(f"  {surface:24s} {len(paths):3d} file(s)")
        for p in sorted(paths)[:5]:
            print(f"      {p}")
        if len(paths) > 5:
            print(f"      … +{len(paths) - 5} more")

    if len(surfaces) > 1:
        # Advisory, never a failure (rule clause 5): print the list so a
        # reviewer can confirm a claimed cross-cutting exception is real.
        print(
            f"\n  ADVISORY: {len(surfaces)} shipping surfaces in one change unit "
            f"({', '.join(sorted(surfaces))})."
        )
        print(
            "  The Surface Rule asks for one surface per PR — split the rest "
            "out, or note the cross-cutting exception in the PR description."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
