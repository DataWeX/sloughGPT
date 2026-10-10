#!/usr/bin/env python
"""Merge the orphaned out-of-repo data tree back into the repo's data/.

Why this exists
---------------
``db_pool._get_repo_root`` used ``parents[5]``, one level above the repo, so
every pooled MogDB wrote to ``<repo-parent>/data/`` while ``FilesRouter`` and
friends wrote to ``<repo>/data/`` — file bytes and file metadata lived in
different trees (card 812cf351). The root was fixed (``domain.shared
.find_repo_root``), but the data was never moved, so the pre-fix history
stayed stranded: e.g. ``uploads_mogdb`` held 3851 records out-of-repo against
1363 in-repo.

What is actually stranded (measured 2026-10-10, not guessed)
-----------------------------------------------------------
The journal *line* counts overstate the loss badly — replaying each store to
its final live state shows the out-of-repo tree is almost entirely test
pollution and already-deleted history:

* ``uploads_mogdb/files`` — 3851 journal lines, but replays to **0 live
  documents** (2780 inserts against ~700 later deletes). Nothing to rescue.
* ``companion_mogdb/presets`` — 184 live docs, of which 180 are pytest
  fixtures (90 "Test Preset", 90 "First") and the other 4 are the same four
  friend presets already in-repo.
* ``experiments_mogdb/*`` — 0 live in either tree.
* ``agents/`` — flat JSON, overwhelmingly ``test_agents_router.py`` output.

Genuinely stranded user data totals three records: chat session
``chat_p7ooipz5`` plus its unsent draft ("I want a short sleep story about
the ocean"), and the ``sloughgpt-model-favorites`` kv key. Select those with
``--collection docstore/sessions --collection docstore/drafts --collection
docstore/kv`` rather than migrating wholesale.

Why it merges instead of moving
-------------------------------
The two trees are not disjoint snapshots — each has its own delete history
(``uploads`` replays 2780 inserts against 700 deletes), so copying the old
journal over would resurrect records the user deleted. Each tree is replayed
to its *final* live set first, the sets are unioned with the in-repo copy
winning, and the result is written back as a MogDB compacted snapshot.

Why it refuses to run against a live server
-------------------------------------------
``Collection.compact`` rebuilds its snapshot from ``self._docs`` — the
in-memory state — and then unlinks the journal. A server holding the store has
no knowledge of records merged underneath it, so a clean shutdown would
silently drop everything this script added. Hence the /proc holder check: the
merge needs a window where the API server is stopped.

Usage
-----
    # dry run — prints what would be added, writes nothing
    python scripts/migrate_out_of_repo_data.py

    # real run, with the API server stopped
    python scripts/migrate_out_of_repo_data.py --apply

    # only one store
    python scripts/migrate_out_of_repo_data.py --collection uploads_mogdb/files
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "mogdb" / "src"))

from mogdb import MogDB  # noqa: E402

# Plain directories (not MogDB): merged by copying files absent from the
# target. ``agents/`` is excluded by default because it is overwhelmingly
# pytest pollution — ``auto-*``/``create-*``/``dup-*``/``get-*``/``list-test-*``
# /``tools-*``/``upd-*`` files matching tests/api-server/test_agents_router.py
# — and importing it would plant that junk in the live tree.
PLAIN_DIRS = ("experiments", "user_adapters")
PLAIN_DIRS_SKIPPED = ("agents",)


def _holders(target: Path) -> list[int]:
    """PIDs with *target* (or a file beside it) open, via /proc.

    Used to refuse a merge underneath a running server: ``compact`` snapshots
    in-memory state and unlinks the journal, so any record merged into a store
    a live process holds would be discarded on its next clean shutdown.
    """
    resolved = target.resolve()
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fd_dir = proc / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    if fd.resolve().parent == resolved:
                        pids.append(int(proc.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def _discover(root: Path) -> list[tuple[str, list[str]]]:
    """``(<db dir>, [<collection names>])`` pairs found under *root*."""
    out: list[tuple[str, list[str]]] = []
    if not root.is_dir():
        return out
    for db_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        names = set()
        for f in db_dir.iterdir():
            if f.suffix == ".jsonl" and f.stem.endswith(".journal"):
                names.add(f.stem[: -len(".journal")])
            elif f.suffix == ".mogdb":
                names.add(f.stem)
        if names:
            out.append((db_dir.name, sorted(names)))
    return out


def _live_docs(db_dir: Path, collection: str) -> dict[str, dict]:
    """Replay *collection* to its final on-disk state using MogDB itself."""
    db = MogDB(str(db_dir), compact_on_close=False, sync_dir=None)
    col = db.collection(collection)
    docs = getattr(col, "_docs", {})
    return {k: dict(v) for k, v in docs.items()}


def migrate_collection(
    src_dir: Path, dst_dir: Path, db_name: str, collection: str, apply: bool
) -> tuple[int, int]:
    """Merge one collection. Returns (records added, records in merged set)."""
    src = _live_docs(src_dir, collection)
    dst = _live_docs(dst_dir, collection)

    # In-repo wins on conflict; only ids absent in-repo are carried over.
    merged = dict(src)
    merged.update(dst)
    added_ids = set(merged) - set(dst)

    if not added_ids or not apply:
        return len(added_ids), len(merged)

    holders = _holders(dst_dir)
    if holders:
        raise SystemExit(
            f"REFUSING {db_name}/{collection}: pids {holders} hold {dst_dir} open. "
            f"Collection.compact() snapshots in-memory state and unlinks the "
            f"journal, so merging under a live process would silently discard "
            f"these records on its next clean shutdown. Stop the API server "
            f"and re-run."
        )

    # Build the merged set in a scratch DB, then let MogDB's own compact()
    # write the snapshot — so the on-disk format is produced by MogDB, not by
    # this script.
    with tempfile.TemporaryDirectory(prefix="mogdb-merge-") as tmp:
        scratch = MogDB(tmp, compact_on_close=False, sync_dir=None)
        col = scratch.collection(collection)
        for doc in merged.values():
            col.insert_one(dict(doc))
        col.compact()
        snapshot = Path(tmp) / f"{collection}.mogdb"
        if not snapshot.is_file():
            raise SystemExit(f"compact produced no snapshot for {db_name}/{collection}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Publish the snapshot, then drop the journal it supersedes — the same
        # order compact() itself uses, so a crash in between leaves the journal
        # authoritative rather than losing records.
        os.replace(snapshot, dst_dir / f"{collection}.mogdb")
        journal = dst_dir / f"{collection}.journal.jsonl"
        if journal.exists():
            journal.unlink()
        shutil.rmtree(tmp, ignore_errors=True)

    return len(added_ids), len(merged)


def migrate_plain(src: Path, dst: Path, apply: bool) -> tuple[int, int]:
    """Copy files present only in *src*. Never overwrites an existing target."""
    missing = [p for p in src.rglob("*") if p.is_file() and not (dst / p.relative_to(src)).exists()]
    if apply:
        for p in missing:
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    return len(missing), sum(1 for _ in src.rglob("*") if _.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=str(REPO.parent / "data"), help="stranded out-of-repo tree")
    ap.add_argument("--to", dest="dst", default=str(REPO / "data"), help="canonical in-repo tree")
    ap.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="limit to '<db>/<collection>'; repeat for several",
    )
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        print(f"source tree not found: {src_root} — nothing to migrate")
        return 0

    print(f"source (stranded): {src_root}")
    print(f"target (canonical): {dst_root}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN (no writes)'}\n")

    total_added = 0
    for db_name, collections in _discover(src_root):
        for collection in collections:
            key = f"{db_name}/{collection}"
            if args.collections and key not in args.collections:
                continue
            added, merged_n = migrate_collection(
                src_root / db_name, dst_root / db_name, db_name, collection, args.apply
            )
            total_added += added
            verb = "would add" if not args.apply else "added"
            print(f"  {key:42} {verb:10} {added:>6}   (merged set: {merged_n})")

    for name in PLAIN_DIRS:
        s, d = src_root / name, dst_root / name
        if not s.is_dir():
            continue
        missing, total = migrate_plain(s, d, args.apply)
        verb = "would add" if not args.apply else "added"
        print(f"  {name + ' (plain files)':42} {verb:10} {missing:>6}   (of {total})")
        total_added += missing

    if PLAIN_DIRS_SKIPPED:
        print(f"\nskipped (test pollution, not user data): {', '.join(PLAIN_DIRS_SKIPPED)}")

    print(f"\n{'TOTAL records/files carried over: ' if args.apply else 'TOTAL that would be carried over: '}{total_added}")
    if not args.apply:
        print("Dry run only. Re-run with --apply (API server stopped) to write.")
    else:
        print("Done. Restart the API server to load the merged stores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
