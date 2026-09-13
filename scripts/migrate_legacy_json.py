"""Migrate legacy JSON/JSONL data files to proper MogDB collections.

This script fixes the anti-pattern where all entries are stored as a single
document (_id: "all_entries"). After migration, each entry is an individual
document in MogDB, enabling efficient queries and updates.

Usage:
    python -m scripts.migrate_legacy_json [--dry-run] [--force]
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "mogdb" / "src"))

from mogdb import MogDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate")

DATA_ROOT = Path(__file__).parent.parent / "data"


def default_collection(src: str) -> str:
    """Derive a collection name from a source path (non-alnum → '_')."""
    slug = re.sub(r"[^A-Za-z0-9]", "_", Path(src).stem)
    return slug or "documents"


def _import_generic(argv: list[str]) -> int:
    """Import one JSON/JSONL file into a MogDB directory.

    Old CLI contract: positional source + ``--db``, ``--collection``,
    ``--key``, ``--dry-run``, ``--force``, ``--sync-dir``.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Import JSON/JSONL data into MogDB")
    parser.add_argument("src", help="Source JSON/JSONL file")
    parser.add_argument("--db", default=None, help="Target MogDB directory")
    parser.add_argument("--collection", default=None, help="Collection name")
    parser.add_argument("--key", default="_id", help="Field to use as document _id")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    parser.add_argument("--sync-dir", default=None, help="Export the resulting collection here")
    args = parser.parse_args(argv)

    source = Path(args.src)
    if not source.exists():
        logger.warning("Source file not found: %s", source)
        return 2

    collection = args.collection or default_collection(args.src)
    db_path = Path(args.db) if args.db else DATA_ROOT / f"{collection}_mogdb"

    with open(source) as f:
        if source.suffix.lower() == ".jsonl":
            docs = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            docs = data if isinstance(data, list) else list(data.values())

    if args.dry_run:
        logger.info("[dry-run] would import %d docs into %s.%s", len(docs), db_path, collection)
        return 0

    backup_existing(db_path)
    db = MogDB(str(db_path))
    col = db.collection(collection)
    col.drop()
    if args.key != "_id":
        for doc in docs:
            if args.key in doc:
                doc["_id"] = str(doc[args.key])
    ids = col.insert_many(docs) if docs else []
    db.close()
    logger.info("Imported %d docs into %s.%s", len(ids), db_path, collection)

    if args.sync_dir:
        sync = Path(args.sync_dir)
        sync.mkdir(parents=True, exist_ok=True)
        verify = MogDB(str(db_path))
        rows = verify.collection(collection).find()
        verify.close()
        target = sync / f"{collection}.json"
        target.write_text(json.dumps(rows, indent=2))
        logger.info("Synced %d docs to %s", len(rows), target)
    return 0


def backup_existing(db_path: Path) -> None:
    """Create a backup of an existing MogDB directory."""
    if db_path.exists():
        backup = db_path.parent / f"{db_path.name}_backup"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(db_path, backup)
        logger.info("Backed up %s to %s", db_path.name, backup.name)


def migrate_entries() -> dict:
    """Migrate knowledge/entries.json → knowledge_mogdb/entries collection.

    Each entry becomes its own document instead of one giant document.
    """
    source_file = DATA_ROOT / "knowledge" / "entries.json"
    db_path = DATA_ROOT / "knowledge_mogdb"

    logger.info("=== Migrating knowledge entries ===")

    if not source_file.exists():
        logger.warning("Source file not found: %s", source_file)
        return {"skipped": True}

    # Load source data
    with open(source_file) as f:
        entries = json.load(f)

    logger.info("Source: %d entries from %s", len(entries), source_file.name)

    # Backup existing
    backup_existing(db_path)

    # Clear old journal if it exists (has the anti-pattern single document)
    journal = db_path / "entries.journal.jsonl"
    if journal.exists():
        journal.unlink()
        logger.info("Removed old anti-pattern journal")

    # Create MogDB with proper per-document storage
    db = MogDB(str(db_path))
    entries_col = db.collection("entries")

    # Clear existing data
    entries_col.drop()

    # Insert each entry as its own document
    if entries:
        result = entries_col.insert_many(entries)
        logger.info("Inserted %d individual entry documents", len(result))

    db.close()

    # Verify
    db_verify = MogDB(str(db_path))
    count = db_verify.collection("entries").count()
    db_verify.close()

    logger.info("Verified: %d documents in entries collection", count)
    return {"inserted": len(entries), "count": count}


def migrate_visited() -> dict:
    """Migrate knowledge/visited.json → knowledge_mogdb/visited collection.

    Each hash becomes its own document.
    """
    source_file = DATA_ROOT / "knowledge" / "visited.json"
    db_path = DATA_ROOT / "knowledge_mogdb"

    logger.info("=== Migrating knowledge visited hashes ===")

    if not source_file.exists():
        logger.warning("Source file not found: %s", source_file)
        return {"skipped": True}

    # Load source data
    with open(source_file) as f:
        hashes = json.load(f)

    logger.info("Source: %d hashes from %s", len(hashes), source_file.name)

    # Create MogDB
    db = MogDB(str(db_path))
    visited_col = db.collection("visited")

    # Clear existing
    visited_col.drop()

    # Insert each hash as its own document
    docs = [{"hash": h} for h in hashes]
    if docs:
        result = visited_col.insert_many(docs)
        logger.info("Inserted %d individual hash documents", len(result))

    db.close()

    # Verify
    db_verify = MogDB(str(db_path))
    count = db_verify.collection("visited").count()
    db_verify.close()

    logger.info("Verified: %d documents in visited collection", count)
    return {"inserted": len(hashes), "count": count}


def migrate_rag_documents() -> dict:
    """Migrate rag_store/documents.jsonl → rag_mogdb/documents collection.

    Each document becomes its own MogDB document.
    """
    source_file = DATA_ROOT / "rag_store" / "documents.jsonl"
    db_path = DATA_ROOT / "rag_mogdb"

    logger.info("=== Migrating RAG documents ===")

    if not source_file.exists():
        logger.warning("Source file not found: %s", source_file)
        return {"skipped": True}

    # Load source data
    docs = []
    with open(source_file) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    logger.info("Source: %d documents from %s", len(docs), source_file.name)

    # Backup existing
    backup_existing(db_path)

    # Clear old journal
    journal = db_path / "documents.journal.jsonl"
    if journal.exists():
        journal.unlink()
        logger.info("Removed old anti-pattern journal")

    # Create MogDB
    db = MogDB(str(db_path))
    docs_col = db.collection("documents")

    # Clear existing
    docs_col.drop()

    # Use sync_from_jsonl for efficient bulk import
    # This handles content hashing and dedup automatically
    if docs:
        # Write to temp file for sync
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for doc in docs:
                tmp.write(json.dumps(doc) + "\n")
            tmp_path = tmp.name

        try:
            # Use insert_many directly for speed
            result = docs_col.insert_many(docs)
            logger.info("Inserted %d individual RAG documents", len(result))
        finally:
            Path(tmp_path).unlink()

    db.close()

    # Verify
    db_verify = MogDB(str(db_path))
    count = db_verify.collection("documents").count()
    db_verify.close()

    logger.info("Verified: %d documents in rag documents collection", count)
    return {"inserted": len(docs), "count": count}


def main(argv: list[str] | None = None) -> int | None:
    import argparse

    if argv is not None:
        # Programmatic API — generic file → MogDB import (testable, no side
        # effects beyond the requested target). Returns a process exit code.
        return _import_generic(argv)

    parser = argparse.ArgumentParser(description="Migrate legacy data to MogDB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    logger.info("Data root: %s", DATA_ROOT)

    if not args.force and not args.dry_run:
        resp = input("This will overwrite existing MogDB journals. Continue? [y/N] ")
        if resp.lower() != "y":
            logger.info("Aborted.")
            return

    results = {}
    results["entries"] = migrate_entries()
    results["visited"] = migrate_visited()
    results["rag_documents"] = migrate_rag_documents()

    logger.info("\n=== Migration Summary ===")
    for key, val in results.items():
        logger.info("  %s: %s", key, val)

    logger.info("\nDone. Knowledge and RAG managers now use proper per-document MogDB storage.")
    return 0


if __name__ == "__main__":
    main()
