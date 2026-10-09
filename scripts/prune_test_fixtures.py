#!/usr/bin/env python3
"""Prune leaked test fixtures out of the pooled MogDB stores.

Why this exists
---------------
The companion/experiment routers create throwaway records during tests —
``test-1791539561968``, ``dup-...``, ``del-...``, named "Test Preset"/"First"/
"To Delete".  ``db_pool.get_db()`` resolves the *live* repo data dir, so any
suite that escaped test isolation (the ``SLO_DATA_DIR`` fixture only reached
``tests/`` before it moved to the root conftest) wrote those fixtures into
production stores.  The presets store reached 210 fixture records against 4
real ones.

This removes them again.  It is deliberately conservative:

* **Dry-run by default.**  ``--apply`` is required to write anything.
* **Timestamp-shaped match only.**  A record must match
  ``^(test|dup|del)-\\d{13}$`` — the exact shape the fixtures are minted in.
  A bare prefix match would also catch a real preset somebody named "deltoid".
* **Backup before any write.**  ``data/_pre_prune_backup_<UTC>/``.

Usage::

    python scripts/prune_test_fixtures.py                    # dry-run, all stores
    python scripts/prune_test_fixtures.py --store presets    # one collection
    python scripts/prune_test_fixtures.py --apply            # backup + write
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in ("packages/mogdb/src", "packages/core-py"):
    _full = _REPO_ROOT / _p
    if _full.is_dir() and str(_full) not in sys.path:
        sys.path.insert(0, str(_full))

# The exact shape tests mint: a fixed prefix plus a millisecond timestamp.
FIXTURE_RE = re.compile(r"^(test|dup|del)-\d{13}$")


def _is_fixture(doc: dict) -> bool:
    return bool(FIXTURE_RE.match(str(doc.get("id", ""))))


def _collections(store_dir: Path) -> list[str]:
    """Collection names present in a store dir, from ``<name>.mogdb``."""
    return sorted(p.stem for p in store_dir.glob("*.mogdb"))


def prune_store(store_dir: Path, apply: bool) -> int:
    """Report (and, with ``apply``, delete) fixture docs. Returns count removed."""
    from mogdb import MogDB

    # Same convention db_pool uses: companion_mogdb -> companion_json.
    sync_dir = store_dir.parent / store_dir.name.replace("_mogdb", "_json")

    removed = 0
    for name in _collections(store_dir):
        # compact_on_close=False for the probe pass so a read never rewrites.
        db = MogDB(str(store_dir), compact_on_close=False)
        try:
            docs = db.collection(name).find({})
        finally:
            db.close()

        fixtures = [d for d in docs if _is_fixture(d)]
        real = len(docs) - len(fixtures)
        print(f"  {name:<16} total={len(docs):<5} keep={real:<5} fixtures={len(fixtures)}")

        if not apply:
            continue

        # Write pass. sync_dir keeps the derived *_json mirror in step with the
        # authoritative store: open without it and a prune leaves the mirror
        # holding every deleted doc, and _bootstrap_from_json would resurrect
        # them all if the snapshot were ever rebuilt from that mirror.
        db = MogDB(str(store_dir), sync_dir=str(sync_dir))
        try:
            col = db.collection(name)
            for doc in fixtures:
                col.delete_one({"_id": doc["_id"]})
            # Force the mirror to the authoritative state even when nothing was
            # deleted — an earlier prune may already have left it stale.
            col._sync_to_json()
        finally:
            db.close()
        removed += len(fixtures)

    return removed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--apply", action="store_true", help="write the pruned state (default: dry-run)"
    )
    ap.add_argument("--root", type=Path, default=None, help="repo root (auto-detected)")
    ap.add_argument("--data-dir", type=Path, default=None, help="data dir (default: <root>/data)")
    ap.add_argument(
        "--store", action="append", default=None, help="limit to a store dir name (repeatable)"
    )
    args = ap.parse_args()

    root = args.root or _REPO_ROOT
    data_dir = args.data_dir or (root / "data")
    if not data_dir.is_dir():
        print(f"no data dir at {data_dir}", file=sys.stderr)
        return 1

    store_dirs = (
        [data_dir / s for s in args.store]
        if args.store
        else sorted(p for p in data_dir.glob("*_mogdb") if p.is_dir())
    )

    total = 0
    backup_root: Path | None = None
    if args.apply:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_root = data_dir / f"_pre_prune_backup_{stamp}"
        backup_root.mkdir(parents=True, exist_ok=True)
        for store_dir in store_dirs:
            if store_dir.is_dir():
                shutil.copytree(store_dir, backup_root / store_dir.name, dirs_exist_ok=True)
        print(f"backup -> {backup_root}\n")

    for store_dir in store_dirs:
        if not store_dir.is_dir():
            print(f"skip {store_dir.name}: not a directory", file=sys.stderr)
            continue
        print(f"{store_dir.name}")
        total += prune_store(store_dir, args.apply)

    if not args.apply:
        print("\nDRY-RUN: nothing written. Re-run with --apply to backup + prune.")
        return 0

    print(f"\npruned {total} fixture document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
