#!/usr/bin/env python3
"""Surface Rule advisory gate — print the shipping-surface span of a diff.

Policy: AGENTS.md "Surface Rule (one change unit spans one shipping surface)".

Failure this prevents: PR 258 (fix/audit-a11y-design-tokens) — 28 commits,
341 files across apps/web + apps/mobile + domain/ + apps/api + docs/; after
main moved, 187/341 files conflicted and the PR was closed unmerged. The
follow-up audit cost hours to conclude 0 of 165 candidates were worth
porting. Scope, not conflict difficulty, made it unlandable.

Advisory by default: always exit 0 (prints WARN on a multi-surface span).
--strict exits 1 — that is the ratchet step, for when this is wired as a
required check (deferred; see the Surface Rule clause 4).

Usage:
  scripts/check_surface_scope.py                  # merge-base(origin/main)..HEAD
  scripts/check_surface_scope.py --base <ref>     # <ref>..HEAD
  scripts/check_surface_scope.py --range a..b     # explicit range
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# The five shipping surfaces (AGENTS.md Surface Rule clause 1). Meta files
# (docs/, scripts/, .github/, tests/, planner stores) map to None: they are
# travel companions, not a second surface (clause 2).
SURFACES: tuple[tuple[str, str], ...] = (
    ("apps/web", "web"),
    ("apps/mobile", "mobile"),
    ("apps/api", "api"),
    ("domain", "domain"),
    ("packages", "packages"),
)


def surface_of(path: str) -> str | None:
    for prefix, name in SURFACES:
        if path == prefix or path.startswith(prefix + "/"):
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=None, help="base ref (default: merge-base with origin/main)")
    ap.add_argument("--range", default=None, help="explicit diff range <a>..<b>")
    ap.add_argument("--strict", action="store_true", help="exit 1 on a multi-surface span")
    args = ap.parse_args()

    if args.range:
        rng = args.range
    else:
        try:
            base = (
                args.base
                or subprocess.check_output(
                    ["git", "merge-base", "origin/main", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except subprocess.CalledProcessError:
            print("surface-scope: no origin/main to merge-base against — pass --base or --range")
            return 0  # advisory: never block on missing history
        rng = f"{base}..HEAD"

    files = subprocess.check_output(["git", "diff", "--name-only", rng], text=True).splitlines()

    span: dict[str, list[str]] = {}
    for f in files:
        s = surface_of(f)
        if s is not None:
            span.setdefault(s, []).append(f)

    if not span:
        print(f"surface-scope: no shipping-surface files in {rng}")
        return 0
    for s in sorted(span):
        print(f"  {s}: {len(span[s])} file(s)")

    if len(span) > 1:
        names = ", ".join(sorted(span))
        print(
            f"WARN surface-scope: this change spans {len(span)} shipping surfaces ({names}).\n"
            "  One change unit spans one shipping surface (AGENTS.md Surface Rule;\n"
            "  PR 258: 341 files across 6 areas -> 187 conflicts -> closed unmerged).\n"
            "  Split into one PR per surface, or — if it is one mechanical identity\n"
            "  across surfaces (rename/token/dep bump) — say so in the PR title."
        )
        return 1 if args.strict else 0

    print(f"surface-scope: OK — single surface ({next(iter(span))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
