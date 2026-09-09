"""
DB command group — MogDB embedded database: migrate, sync, status.
"""

import json
import sys
from pathlib import Path

from core.framework import click, echo
from core.helpers import ns as _ns
from domains.logging import get_global
log = get_global()


def _db_default_dir() -> str:
    return str(Path.cwd().parent.parent.parent / "data" / "mogdb")


def _import_mogdb():
    """Import mogdb, adding its src dir to sys.path for this session."""
    _src = Path(__file__).resolve().parents[3] / "packages" / "mogdb" / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from mogdb import MogDB
    from mogdb.sync import sync_from_files, preview_from_files
    return MogDB, sync_from_files, preview_from_files


def register(cli):
    """Register db commands with the CLI group."""

    @cli.group(help="MogDB embedded database — migrate, sync, and status")
    def db():
        pass

    @db.command("migrate", help="Import a JSON/JSONL/CSV file into a MogDB collection")
    @click.option("--file", required=True, help="Path to the source file (.json, .jsonl, or .csv)")
    @click.option("--collection", help="Target collection name (default: from filename)")
    @click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
    @click.option("--key", required=True, help="Identity/key field used for dedupe")
    @click.option("--format", type=click.Choice(["json", "jsonl", "csv"]), help="Source format (default: auto-detect)")
    @click.option("--delete-missing", is_flag=True, help="Delete collection docs missing from the file")
    @click.option("--sync-dir", help="Write human-readable JSON sync files to this directory")
    @click.option("--dry-run", is_flag=True, help="Report what would change without writing")
    def db_migrate(file, collection, db, key, format, delete_missing, sync_dir, dry_run):
        _MogDB, _sync, _preview = _import_mogdb()
        path = Path(file)
        if not path.exists():
            log.error(f"source file not found: {file}")
            return
        if collection is None:
            collection = path.stem.replace("-", "_").replace(".", "_").lower()
        if db is None:
            db = _db_default_dir()
        _database = _MogDB(db, compact_on_close=not dry_run, sync_dir=sync_dir)
        try:
            _col = _database.collection(collection)
            result = (_preview if dry_run else _sync)(
                _col, str(path), key_field=key,
                delete_missing=delete_missing, file_format=format,
            )
        finally:
            _database.close()
        if dry_run:
            log.info(f"dry run for {collection}:")
        log.success(
            f"{collection}: +{result.inserted} ~{result.updated} -{result.deleted} ={result.unchanged}"
        )

    @db.command("sync", help="Force JSON sync for all collections")
    @click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
    @click.option("--sync-dir", required=True, help="Directory holding the JSON sync files")
    def db_sync(db, sync_dir):
        _MogDB, _, _ = _import_mogdb()
        if db is None:
            db = _db_default_dir()
        _database = _MogDB(db, compact_on_close=True, sync_dir=sync_dir)
        try:
            names = _database.list_collections()
            for name in names:
                _database.collection(name).sync()
        finally:
            _database.close()
        log.success(f"synced {len(names)} collections to {sync_dir}")

    @db.command("status", help="Show MogDB collection stats")
    @click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
    @click.option("--json", "as_json", is_flag=True, help="Print JSON output")
    def db_status(db, as_json):
        _MogDB, _, _ = _import_mogdb()
        if db is None:
            db = _db_default_dir()
        _database = _MogDB(db, compact_on_close=False)
        try:
            names = _database.list_collections()
            rows = []
            for name in names:
                rows.append({"collection": name, "count": _database.collection(name).count()})
        finally:
            _database.close()
        if as_json:
            echo(json.dumps(rows, indent=2, default=str))
        else:
            for row in rows:
                echo(f"{row['collection']}: {row['count']} docs")

    return db
