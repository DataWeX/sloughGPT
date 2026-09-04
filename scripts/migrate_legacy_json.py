"""Import a legacy JSON/JSONL/CSV file into a MogDB collection.

One-time migration helper. Loads documents from an external file and syncs
them into a MogDB collection using the diff engine in ``mogdb.sync``. Auto-
detects format from the file extension unless overridden. Idempotent: run
it again to pick up changes rather than duplicate documents.

Example::

    python3 scripts/migrate_legacy_json.py \\
        data/knowledge/entries.json --collection knowledge --key id
"""

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MOGDB_SRC = _REPO / "packages" / "mogdb" / "src"
if str(_MOGDB_SRC) not in sys.path:
    sys.path.insert(0, str(_MOGDB_SRC))

from mogdb import MogDB  # noqa: E402
from mogdb.sync import SyncResult, preview_from_files, sync_from_files  # noqa: E402


def default_collection(file_name: str) -> str:
    """Derive a collection name from a file name (stem, dashes to underscores)."""
    stem = Path(file_name).stem
    return stem.replace("-", "_").replace(".", "_").lower()


def migrate_file(args) -> "dict":
    """Run the migration against a real MogDB collection and return SyncResult dict."""
    # In dry-run, hold the DB read-only: no compaction artifact is written.
    db = MogDB(args.db, compact_on_close=not args.dry_run, sync_dir=args.sync_dir)
    try:
        coll = db.collection(args.collection)
        if args.dry_run:
            result = preview_from_files(
                coll,
                str(args.file),
                key_field=args.key,
                delete_missing=args.delete_missing,
                file_format=args.format,
            )
        else:
            result = sync_from_files(
                coll,
                str(args.file),
                key_field=args.key,
                delete_missing=args.delete_missing,
                file_format=args.format,
            )
    finally:
        db.close()
    return result.to_dict() if isinstance(result, SyncResult) else result


def _print_summary(result: dict, dry_run: bool) -> None:
    mode = " (dry run — no changes written)" if dry_run else ""
    print(f"migration complete{mode}")
    for label in ("inserted", "updated", "deleted", "unchanged"):
        print(f"  {label}: {result[label]}")
    if result["errors"]:
        print(f"  errors: {len(result['errors'])}")
        for e in result["errors"]:
            print(f"    - {e}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="migrate_legacy_json",
        description="Import legacy JSON/JSONL/CSV data into a MogDB collection.",
    )
    p.add_argument("file", help="Path to the source file (.json, .jsonl, or .csv)")
    p.add_argument("--collection", help="Target MogDB collection (default: from filename)")
    p.add_argument("--db", default=str(_REPO / "data" / "mogdb"),
                   help="MogDB data directory (default: %(default)s)")
    p.add_argument("--key", required=True,
                   help="Field name used as the document identity / key")
    p.add_argument("--format", choices=["json", "jsonl", "csv"],
                   help="Source format (default: auto-detect from extension)")
    p.add_argument("--delete-missing", action="store_true",
                   help="Delete collection docs whose key is absent from the file")
    p.add_argument("--sync-dir", default=None,
                   help="Directory to write human-readable JSON sync files (enables JSON sync)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts without writing any changes")
    args = p.parse_args(argv)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"error: source file not found: {args.file}", file=sys.stderr)
        return 2
    if args.collection is None:
        args.collection = default_collection(file_path.name)

    try:
        result = migrate_file(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
