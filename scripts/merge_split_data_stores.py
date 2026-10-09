#!/usr/bin/env python
"""Reconcile the split-brain MogDB stores left by the db_pool off-by-one.

``db_pool._get_repo_root()`` resolved one directory *above* the repo, so every
pooled store (``uploads_mogdb``, ``companion_mogdb``, ``experiments_*``) was
written to ``<parent-of-repo>/data`` while ``auth.py``/``docstore.py`` kept
writing to the real ``<repo>/data``.  Two live halves grew, and neither is a
subset of the other, so this performs a **union merge** keyed by ``_id``:

* the document with the later ``_updated`` wins;
* ties go to the outside half — that is the copy the running server (and so the
  user's UI) has actually been reading.

Safety properties:

* **Dry-run by default.** ``--apply`` is required to write anything.
* **Both originals are backed up** to ``data/_split_brain_backup_<ts>/`` before
  a single byte is rewritten, and are never deleted.
* **Read-only replay.**  Journals are parsed in-process; MogDB is never opened
  on the live files, because its loader quarantines corrupt lines by
  *rewriting* them — a mutation we do not want during triage.
* Output is written as a compacted ``.mogdb`` snapshot (the format ``compact()``
  produces), with a seq watermark far above anything observed so a leftover
  journal can never shadow it.  ``*_json`` sync mirrors are written as the bare
  JSON array their loader bootstraps from — never as a ``.mogdb``.
* Each side is replayed as a **stack**: snapshot first (baseline), then the
  journal's ops on top, exactly as MogDB's loader does.  The derived
  ``*_json`` mirror is used only when neither store file exists.

Usage::

    python scripts/merge_split_data_stores.py            # dry-run report
    python scripts/merge_split_data_stores.py --apply    # backup + write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Stores db_pool owned (the leaking half).  auth_mogdb/docstore were never
# split — they always used the real repo root — so they are not merged here.
SPLIT_COLLECTIONS = (
    ("uploads_mogdb", "files"),
    ("uploads_json", "files"),
    ("companion_mogdb", "presets"),
    ("companion_json", "presets"),
    ("experiments_mogdb", "experiments"),
    ("experiments_mogdb", "metrics"),
    ("experiments_mogdb", "params"),
    ("experiments_mogdb", "status"),
    ("experiments_json", "experiments"),
)

# seq watermark for the rewritten snapshot: above any real seq so a surviving
# journal can never be replayed over the merged state.
MERGED_SEQ = 10**12


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "apps").is_dir() and (parent / "packages").is_dir():
            return parent
    raise SystemExit("repo root not found")


def replay(
    path: Path,
    docs: dict[str, dict] | None = None,
    max_seq: int = 0,
    lines: int = 0,
) -> tuple[dict[str, dict], int, int]:
    """Replay a journal, snapshot, or JSON-sync mirror into ``{_id: doc}``.

    Handles all three on-disk shapes: append-only JSONL journals, compacted
    ``.mogdb`` snapshots (meta line + bare docs), and the ``*_json`` sync
    mirrors, which are a single pretty-printed JSON *array* — line-wise
    ``json.loads`` yields nothing for those, so the mirror half must be parsed
    whole-document or it is silently dropped.

    ``docs``/``max_seq``/``lines`` may be supplied to **layer** sources: replay
    the snapshot first to get the baseline, then replay the journal into the
    same dict so its ops land on top.  Reading only the journal — as this
    function's original single-source caller did — throws the snapshot away and
    reports the journal's handful of lines as the whole store.

    Returns ``(docs, max_seq, source_lines)``.  Never mutates *path*.
    """
    if docs is None:
        docs = {}
    if not path.exists():
        return docs, max_seq, lines

    text = path.read_text(encoding="utf-8", errors="replace")

    # JSON-sync mirror: one pretty-printed array of bare docs.
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict) or "_mogdb_meta" in entry:
                    continue
                doc_id = str(entry.get("_id", ""))
                if doc_id:
                    docs[doc_id] = entry
            return docs, max_seq, lines + len(data)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail / NUL poisoning — same as MogDB's quarantine
        if not isinstance(entry, dict):
            continue

        # Compacted snapshot: optional meta line, then bare docs.
        if "_mogdb_meta" in entry:
            max_seq = max(max_seq, int(entry["_mogdb_meta"].get("seq", 0)))
            continue
        if "op" not in entry:
            doc_id = str(entry.get("_id", ""))
            if doc_id:
                docs[doc_id] = entry
            continue

        max_seq = max(max_seq, int(entry.get("seq", 0) or 0))
        op = entry.get("op")
        data = entry.get("data")
        if not isinstance(data, dict):
            continue

        if op == "insert":
            doc_id = str(data.get("_id", ""))
            if doc_id:
                docs[doc_id] = data
        elif op == "update":
            doc = docs.get(str(data.get("_id", "")))
            if doc is not None:
                doc.update(data.get("update") or {})
                doc["_updated"] = time.time()
        elif op == "update_many":
            update = data.get("update") or {}
            query = data.get("query") or {}
            for doc in list(docs.values()):
                if all(doc.get(k) == v for k, v in query.items()):
                    doc.update(update)
                    doc["_updated"] = time.time()
        elif op == "delete":
            docs.pop(str(data.get("_id", "")), None)
        elif op == "delete_many":
            query = data.get("query") or {}
            if not query:
                docs.clear()
            else:
                for doc_id in [
                    d
                    for d, v in list(docs.items())
                    if all(v.get(k) == val for k, val in query.items())
                ]:
                    docs.pop(doc_id, None)

    return docs, max_seq, lines


def load_side(store_dir: Path, coll: str) -> tuple[dict[str, dict], int, int]:
    """Replay one side's sources in MogDB's own order: snapshot, then journal.

    The ``*_json`` mirror is consulted only when neither store file exists —
    it is a *derived* bootstrap source (``_bootstrap_from_json`` runs solely
    when the collection is empty), so preferring it would re-inject content the
    snapshot has already superseded.

    Picking a single "first file that exists" was the bug: once a journal
    appeared beside the snapshot, only the journal's few ops were replayed and
    the snapshot baseline vanished from the report (208 docs reported as 16).
    """
    docs: dict[str, dict] = {}
    max_seq = 0
    lines = 0

    snapshot = store_dir / f"{coll}.mogdb"
    journal = store_dir / f"{coll}.journal.jsonl"
    mirror = store_dir / f"{coll}.json"

    if snapshot.exists() or journal.exists():
        if snapshot.exists():
            docs, max_seq, lines = replay(snapshot, docs, max_seq, lines)
        if journal.exists():
            docs, max_seq, lines = replay(journal, docs, max_seq, lines)
    elif mirror.exists():
        docs, max_seq, lines = replay(mirror, docs, max_seq, lines)

    return docs, max_seq, lines


def merge_docs(outside: dict[str, dict], inside: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Union by ``_id``; later ``_updated`` wins, ties prefer the outside half."""
    merged: dict[str, dict] = {}
    stats = {"outside_only": 0, "inside_only": 0, "outside_won": 0, "inside_won": 0, "identical": 0}

    for doc_id, out_doc in outside.items():
        in_doc = inside.get(doc_id)
        if in_doc is None:
            merged[doc_id] = out_doc
            stats["outside_only"] += 1
            continue
        out_ts = out_doc.get("_updated", 0) or 0
        in_ts = in_doc.get("_updated", 0) or 0
        if in_ts > out_ts:
            merged[doc_id] = in_doc
            stats["inside_won"] += 1
        elif out_ts > in_ts:
            merged[doc_id] = out_doc
            stats["outside_won"] += 1
        else:
            merged[doc_id] = out_doc
            stats["identical"] += 1

    for doc_id, in_doc in inside.items():
        if doc_id not in outside:
            merged[doc_id] = in_doc
            stats["inside_only"] += 1

    return merged, stats


def snapshot_payload(docs: dict[str, dict]) -> bytes:
    """Serialise as a compacted ``.mogdb`` snapshot (meta line + bare docs)."""
    lines = [json.dumps({"_mogdb_meta": {"seq": MERGED_SEQ, "v": 1}}, default=str)]
    lines.extend(json.dumps(doc, default=str) for doc in docs.values())
    return ("\n".join(lines) + "\n").encode("utf-8")


def mirror_payload(docs: dict[str, dict]) -> bytes:
    """Serialise in the shape ``MogDB._sync_to_json`` writes: a bare array.

    The ``*_json`` directories are sync mirrors, not stores — dropping a
    ``.mogdb`` snapshot in there would be an orphan the loader never reads.
    """
    return (json.dumps(list(docs.values()), indent=2, default=str) + "\n").encode("utf-8")


def is_mirror(store: str) -> bool:
    """True for a ``*_json`` sync-mirror directory (vs. a ``*_mogdb`` store)."""
    return store.endswith("_json")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--apply", action="store_true", help="write the merged state (default: dry-run)"
    )
    ap.add_argument("--root", type=Path, default=None, help="repo root (auto-detected)")
    args = ap.parse_args()

    root = (args.root or find_root()).resolve()
    outside_root = root.parent / "data"
    inside_root = root / "data"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    print(f"repo root : {root}")
    print(f"outside   : {outside_root}")
    print(f"inside    : {inside_root}")
    print(f"mode      : {'APPLY' if args.apply else 'DRY-RUN (no writes)'}")
    print()

    plan: list[tuple[str, str, Path, Path, dict[str, dict], dict, int]] = []

    for store, coll in SPLIT_COLLECTIONS:
        out_dir = outside_root / store
        in_dir = inside_root / store

        if not out_dir.is_dir() and not in_dir.is_dir():
            continue

        outside_docs, out_seq, out_lines = load_side(out_dir, coll)
        inside_docs, in_seq, in_lines = load_side(in_dir, coll)

        if not outside_docs and not inside_docs:
            continue

        merged, stats = merge_docs(outside_docs, inside_docs)
        plan.append((store, coll, out_dir, in_dir, merged, stats, max(out_seq, in_seq)))

        total = len(merged)
        changed = (
            stats["outside_only"]
            + stats["inside_only"]
            + stats["outside_won"]
            + stats["inside_won"]
        )
        print(f"{store}/{coll}")
        print(
            f"    outside={len(outside_docs):5d}  inside={len(inside_docs):5d}  ->  merged={total:5d}"
        )
        print(
            f"    only-outside={stats['outside_only']:4d}  only-inside={stats['inside_only']:4d}"
            f"  outside-newer={stats['outside_won']:3d}  inside-newer={stats['inside_won']:3d}"
            f"  same-ts={stats['identical']:4d}"
        )
        print(f"    changed={changed}  max_seq_seen={max(out_seq, in_seq)}")
        print()

    if not plan:
        print("nothing to merge")
        return 0

    if not args.apply:
        print("DRY-RUN: nothing written. Re-run with --apply to backup + merge.")
        return 0

    # ---- backups: copy BOTH halves verbatim before touching anything ----
    backup = inside_root / f"_split_brain_backup_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    for store, coll, out_dir, in_dir, *_ in plan:
        for side, src_dir in (("outside", out_dir), ("inside", in_dir)):
            if src_dir.is_dir():
                dest = backup / side / store
                if not dest.exists():
                    shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    print(f"backups -> {backup}")
    print()

    # ---- write merged state into the in-repo half ----
    written = 0
    for store, coll, out_dir, in_dir, merged, stats, _ in plan:
        in_dir.mkdir(parents=True, exist_ok=True)
        mirror = is_mirror(store)
        # Each store kind keeps its own format: a *.mogdb store takes a
        # compacted snapshot, a *_json sync mirror takes the bare array its
        # loader bootstraps from.
        out_file = in_dir / (f"{coll}.json" if mirror else f"{coll}.mogdb")
        payload = mirror_payload(merged) if mirror else snapshot_payload(merged)
        tmp = in_dir / (f"{coll}.json.tmp" if mirror else f"{coll}.mogdb.tmp")
        tmp.write_bytes(payload)
        tmp.replace(out_file)  # atomic — never a half-written file

        # The journal is now fully shadowed by the snapshot's watermark; removing
        # it is exactly what compact() does.  Keep it in the backup only.
        if not mirror:
            journal = in_dir / f"{coll}.journal.jsonl"
            if journal.exists():
                journal.unlink()
        written += 1
        print(f"wrote {out_file.relative_to(root)} ({len(merged)} docs)")

    print()
    print(f"merged {written} collection(s).")
    print(f"originals preserved in {backup}")
    print("restart the API server so it loads the fixed db_pool root.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
