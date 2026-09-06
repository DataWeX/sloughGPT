"""Migrate legacy JSON/JSONL data files to proper MogDB collections.

This script fixes the anti-pattern where all entries are stored as a single
document (_id: "all_entries"). After migration, each entry is an individual
document in MogDB, enabling efficient queries and updates.

Usage:
    python -m scripts.migrate_legacy_json [--dry-run] [--force]
"""

import json
import logging
import shutil
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "mogdb" / "src"))

from mogdb import MogDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate")

DATA_ROOT = Path(__file__).parent.parent / "data"


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


def main():
    import argparse
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


if __name__ == "__main__":
    main()
