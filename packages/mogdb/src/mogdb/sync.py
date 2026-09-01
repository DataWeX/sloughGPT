"""Diff-based file sync for MogDB collections.

Compares documents in a collection against external files (JSON, JSONL, CSV)
and applies the delta: inserts new documents, updates changed ones, and
optionally deletes documents that no longer exist in the source file.

Usage::

    from mogdb import MogDB
    from mogdb.sync import sync_from_json

    db = MogDB("data/mogdb")
    users = db.collection("users")

    # Sync from a JSON file, using "email" as the key field
    result = sync_from_json(users, "users.json", key_field="email")
    print(result)  # {"inserted": 3, "updated": 1, "deleted": 0, "unchanged": 5}
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mogdb.sync")


class SyncResult:
    """Result of a sync operation."""

    __slots__ = ("inserted", "updated", "deleted", "unchanged", "errors")

    def __init__(self) -> None:
        self.inserted: int = 0
        self.updated: int = 0
        self.deleted: int = 0
        self.unchanged: int = 0
        self.errors: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "deleted": self.deleted,
            "unchanged": self.unchanged,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"SyncResult(inserted={self.inserted}, updated={self.updated}, "
            f"deleted={self.deleted}, unchanged={self.unchanged}, "
            f"errors={len(self.errors)})"
        )


def _content_hash(doc: Dict[str, Any]) -> str:
    """SHA-256 of document content excluding metadata fields."""
    clean = {k: v for k, v in doc.items() if not k.startswith("_")}
    raw = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cast_csv_value(v: str) -> Any:
    """Best-effort cast of a CSV string value."""
    if v == "":
        return None
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _load_json(path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("root", "data", "items", "documents"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return data


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSONL file."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _load_csv(path: str) -> List[Dict[str, Any]]:
    """Load documents from a CSV file."""
    docs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append({k: _cast_csv_value(v) for k, v in row.items()})
    return docs


def sync_from_files(
    collection,
    file_path: str,
    key_field: str,
    *,
    delete_missing: bool = False,
    file_format: Optional[str] = None,
) -> SyncResult:
    """Sync a collection from an external file.

    Compares documents in *collection* against those in *file_path* using
    *key_field* as the identity. Documents whose content hash differs are
    updated. Documents in the file that don't exist in the collection are
    inserted. Optionally, documents in the collection that don't exist in
    the file are deleted.

    Parameters
    ----------
    collection:
        A MogDB Collection instance.
    file_path:
        Path to the source file (JSON, JSONL, or CSV).
    key_field:
        Field name used as the document identity / key.
    delete_missing:
        If True, remove collection documents whose key is not in the file.
    file_format:
        Explicit format ("json", "jsonl", "csv"). Auto-detected from
        extension when None.

    Returns a SyncResult with counts of inserted, updated, deleted,
    and unchanged documents.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    if file_format is None:
        ext = path.suffix.lower()
        format_map = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}
        file_format = format_map.get(ext)
        if file_format is None:
            raise ValueError(f"Cannot detect format from extension '{ext}'. Pass file_format explicitly.")

    loaders = {"json": _load_json, "jsonl": _load_jsonl, "csv": _load_csv}
    source_docs = loaders[file_format](str(path))

    # Build source index: key_value -> source doc
    source_by_key: Dict[Any, Dict[str, Any]] = {}
    for doc in source_docs:
        kv = doc.get(key_field)
        if kv is not None:
            source_by_key[kv] = doc

    # Build collection index: key_value -> collection doc
    existing = collection.find()
    existing_by_key: Dict[Any, Dict[str, Any]] = {}
    for doc in existing:
        kv = doc.get(key_field)
        if kv is not None:
            existing_by_key[kv] = doc

    result = SyncResult()

    # Insert or update
    for kv, source_doc in source_by_key.items():
        existing_doc = existing_by_key.get(kv)
        if existing_doc is None:
            # Insert new document
            collection.insert_one(source_doc)
            result.inserted += 1
        else:
            # Compare content hash
            old_hash = _content_hash(existing_doc)
            new_hash = _content_hash(source_doc)
            if old_hash != new_hash:
                # Update changed document
                update_ops = {k: v for k, v in source_doc.items() if not k.startswith("_")}
                collection.update_one(
                    {"_id": existing_doc["_id"]},
                    {"$set": update_ops},
                )
                result.updated += 1
            else:
                result.unchanged += 1

    # Delete missing documents
    if delete_missing:
        for kv, existing_doc in existing_by_key.items():
            if kv not in source_by_key:
                collection.delete_one({"_id": existing_doc["_id"]})
                result.deleted += 1

    logger.info(
        "sync complete: +%d ~%d -%d =%d (%d errors)",
        result.inserted, result.updated, result.deleted, result.unchanged,
        len(result.errors),
    )
    return result


# Convenience aliases
def sync_from_json(collection, file_path: str, key_field: str, **kwargs: Any) -> SyncResult:
    """Sync from a JSON file."""
    return sync_from_files(collection, file_path, key_field, file_format="json", **kwargs)


def sync_from_jsonl(collection, file_path: str, key_field: str, **kwargs: Any) -> SyncResult:
    """Sync from a JSONL file."""
    return sync_from_files(collection, file_path, key_field, file_format="jsonl", **kwargs)


def sync_from_csv(collection, file_path: str, key_field: str, **kwargs: Any) -> SyncResult:
    """Sync from a CSV file."""
    return sync_from_files(collection, file_path, key_field, file_format="csv", **kwargs)
