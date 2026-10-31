#!/usr/bin/env python3
"""
Recompute the per-dir git tree hashes in domain/domain-baseline.json.

When `tests/core-py/test_domain_baseline.py::TestDirectoryHashDrift` reds:
a domain dir's committed contents changed since the catalogue was recorded.
Placement deserves a second look before the hash moves (that is the point of
the drift gate) — check each printed dir's class/facets/note, adjust them if
the code moved, then:

    python3 scripts/domain_baseline_rehash.py            # rewrite hashes
    python3 scripts/domain_baseline_rehash.py --check    # CI-style: exit 1 on drift

Stdlib only. Rewrites metadata.recorded_commit to HEAD. Never touches
class/facets/consumers/note/debt — this script moves hashes, humans move
classification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "domain" / "domain-baseline.json"
PRUNE = {"__pycache__"}


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(f"git {' '.join(args)} failed: {res.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return res.stdout.strip()


def dir_hashes() -> dict[str, str]:
    names = sorted(
        p.name for p in (REPO / "domain").iterdir() if p.is_dir() and p.name not in PRUNE
    )
    return {n: git("rev-parse", f"HEAD:domain/{n}") for n in names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on drift instead of rewriting (no file changes)",
    )
    args = parser.parse_args(argv)

    doc = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = dir_hashes()
    drifted = {
        name: (doc["dirs"][name]["tree_hash"], h)
        for name, h in current.items()
        if name in doc["dirs"] and doc["dirs"][name]["tree_hash"] != h
    }
    uncatalogued = sorted(set(current) - set(doc["dirs"]))
    ghosts = sorted(set(doc["dirs"]) - set(current))

    if uncatalogued or ghosts:
        print(
            f"catalogue mismatch: uncatalogued={uncatalogued} ghosts={ghosts} "
            "— edit domain/domain-baseline.json entries by hand first",
            file=sys.stderr,
        )
        return 2

    if not drifted:
        print("no drift: all dir tree hashes match the baseline")
        return 0

    print(f"drifted dirs ({len(drifted)}) — re-review placement:")
    for name, (old, new) in sorted(drifted.items()):
        print(f"  {name}: {old[:8]} -> {new[:8]}")

    if args.check:
        return 1

    for name, (_, new) in drifted.items():
        doc["dirs"][name]["tree_hash"] = new
    doc["metadata"]["recorded_commit"] = git("rev-parse", "HEAD")
    BASELINE.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rewrote {len(drifted)} hash(es) in {BASELINE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
