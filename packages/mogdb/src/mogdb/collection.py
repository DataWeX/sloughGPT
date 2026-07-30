"""Collection CRUD for MogDB.

Each collection manages an in-memory document map backed by an append-only
JSONL journal. Writes are journaled immediately; reads come from memory.
Supports MongoDB-style update operators ($set, $unset, $inc, $push, $pull,
$addToSet) with dot-notation for nested fields.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .document import Document
from .index import Index
from .query import _get_field, match_document

logger = logging.getLogger("slo.mogdb")


class Collection:
    """A named document collection within a MogDB database.

    Parameters
    ----------
    name:
        Collection name (used as the journal filename stem).
    db_path:
        Directory path for the journal file.
    """

    def __init__(self, name: str, db_path: Path):
        self.name = name
        self._db_path = db_path
        self._docs: Dict[str, Document] = {}
        self._lock = threading.Lock()
        self._journal_path = db_path / f"{name}.journal.jsonl"
        self._compacted_path = db_path / f"{name}.mogdb"
        self._dirty: bool = False
        self._indexes: Dict[str, Index] = {}

        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load documents from the most recent snapshot or journal."""
        data_path = self._compacted_path if self._compacted_path.exists() else self._journal_path
        if not data_path.exists():
            return

        count = 0
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "data" in entry:
                        entry = entry["data"]
                    doc = Document(entry)
                    self._docs[doc.id] = doc
                    count += 1
                except json.JSONDecodeError:
                    continue
        if count:
            logger.debug("loaded %d docs from %s", count, data_path.name)

    def _journal(self, op: str, data: Dict[str, Any]) -> None:
        """Append an operation to the journal."""
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"op": op, "data": data}
        with open(self._journal_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._dirty = True

    def compact(self) -> int:
        """Rewrite the journal as a compacted snapshot.

        Drops all tombstones (deleted docs) and history, keeping only the
        current state. The journal file is replaced by the compacted file.
        """
        with self._lock:
            entries = [dict(d) for d in self._docs.values()]
            self._compacted_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._compacted_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, default=str) + "\n")
            # Remove old journal
            if self._journal_path.exists():
                self._journal_path.unlink()
            self._dirty = False
        logger.info("compacted %s: %d docs", self.name, len(entries))
        return len(entries)

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def create_index(self, field: str, unique: bool = False) -> Index:
        """Create an index on *field*. Returns the Index object.

        Existing documents are indexed. Duplicate values raise ValueError
        if ``unique=True``.
        """
        if field in self._indexes:
            return self._indexes[field]
        idx = Index(field, unique=unique)
        self._indexes[field] = idx
        with self._lock:
            for doc in self._docs.values():
                val = _get_field(doc, field)
                idx.add(doc.id, val)
        return idx

    def drop_index(self, field: str) -> None:
        """Drop the index on *field*."""
        self._indexes.pop(field, None)

    def _index_insert(self, doc: Document) -> None:
        """Update all indexes after an insert."""
        for field, idx in self._indexes.items():
            val = _get_field(doc, field)
            idx.add(doc.id, val)

    def _index_update(self, doc: Document, old_vals: Dict[str, Any]) -> None:
        """Update all indexes after an update."""
        for field, idx in self._indexes.items():
            old_val = old_vals.get(field)
            new_val = _get_field(doc, field)
            idx.update(doc.id, old_val, new_val)

    def _index_remove(self, doc: Document) -> None:
        """Update all indexes after a delete."""
        for field, idx in self._indexes.items():
            val = _get_field(doc, field)
            idx.remove(doc.id, val)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert_one(self, doc: Dict[str, Any]) -> str:
        """Insert a single document. Returns its ``_id``."""
        d = Document(doc)
        with self._lock:
            self._docs[d.id] = d
            self._index_insert(d)
            self._journal("insert", dict(d))
        return d.id

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents. Returns their ``_id``s."""
        ids: List[str] = []
        with self._lock:
            for doc in docs:
                d = Document(doc)
                self._docs[d.id] = d
                self._index_insert(d)
                ids.append(d.id)
                self._journal("insert", dict(d))
        return ids

    def find(
        self,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find documents matching *query*.

        Parameters
        ----------
        query:
            MongoDB-style query dict. ``None`` or ``{}`` matches all.
        sort:
            List of ``(field, direction)`` tuples where direction is
            1 (ascending) or -1 (descending).
        limit:
            Maximum number of results.
        skip:
            Number of results to skip before returning.

        Returns a list of plain dict copies.
        """
        with self._lock:
            results = [
                dict(d)
                for d in self._docs.values()
                if not query or match_document(d, query)
            ]

        if sort:
            for field, direction in reversed(sort):
                results.sort(
                    key=lambda r, f=field: r.get(f) if f in r else "",
                    reverse=(direction == -1),
                )

        if skip:
            results = results[skip:]
        if limit is not None:
            results = results[:limit]

        return results

    def find_one(
        self, query: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the first document matching *query*, or ``None``."""
        results = self.find(query, limit=1)
        return results[0] if results else None

    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching *query*."""
        if not query:
            with self._lock:
                return len(self._docs)
        return len(self.find(query))

    def update_one(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
    ) -> int:
        """Update the first document matching *query*.

        Supports ``$set``, ``$unset``, ``$inc``, ``$push``, ``$pull``,
        ``$addToSet``, and ``$mul`` operators.
        Returns the number of modified documents (0 or 1).
        """
        with self._lock:
            for doc in self._docs.values():
                if not match_document(doc, query):
                    continue
                old_vals = {f: _get_field(doc, f) for f in self._indexes}
                self._apply_update(doc, update)
                doc["_updated"] = time.time()
                self._index_update(doc, old_vals)
                self._journal("update", {"_id": doc.id, "update": update})
                return 1
        return 0

    def update_many(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
    ) -> int:
        """Update all documents matching *query*."""
        count = 0
        with self._lock:
            for doc in self._docs.values():
                if not match_document(doc, query):
                    continue
                old_vals = {f: _get_field(doc, f) for f in self._indexes}
                self._apply_update(doc, update)
                doc["_updated"] = time.time()
                self._index_update(doc, old_vals)
                count += 1
            if count:
                self._journal("update_many", {"query": query, "update": update})
        return count

    def delete_one(self, query: Dict[str, Any]) -> int:
        """Delete the first document matching *query*."""
        with self._lock:
            for doc_id, doc in list(self._docs.items()):
                if match_document(doc, query):
                    self._index_remove(doc)
                    del self._docs[doc_id]
                    self._journal("delete", {"_id": doc_id})
                    return 1
        return 0

    def delete_many(self, query: Dict[str, Any]) -> int:
        """Delete all documents matching *query*."""
        count = 0
        with self._lock:
            for doc_id, doc in list(self._docs.items()):
                if match_document(doc, query):
                    self._index_remove(doc)
                    del self._docs[doc_id]
                    count += 1
            if count:
                self._journal("delete_many", {"query": query, "count": count})
        return count

    def drop(self) -> None:
        """Remove all documents and journal files."""
        with self._lock:
            self._docs.clear()
            for idx in self._indexes.values():
                idx.clear()
            self._dirty = False
            for p in [self._journal_path, self._compacted_path]:
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------------
    # update operators
    # ------------------------------------------------------------------

    @staticmethod
    def _set_nested(doc: Document, path: str, value: Any) -> None:
        """Set a value at a dot-separated path, creating intermediate dicts."""
        parts = path.split(".")
        current = doc
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    @staticmethod
    def _unset_nested(doc: Document, path: str) -> None:
        """Remove a value at a dot-separated path."""
        parts = path.split(".")
        current = doc
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                return
            current = current[part]
        if isinstance(current, dict):
            current.pop(parts[-1], None)

    @classmethod
    def _apply_update(cls, doc: Document, update: Dict[str, Any]) -> None:
        """Apply MongoDB-style update operators to a document.

        Supports: $set, $unset, $inc, $push, $pull, $addToSet, $mul.
        """
        for op, data in update.items():
            if op == "$set":
                if isinstance(data, dict):
                    for path, value in data.items():
                        cls._set_nested(doc, path, value)
            elif op == "$unset":
                if isinstance(data, dict):
                    for path in data:
                        cls._unset_nested(doc, path)
            elif op == "$inc":
                if isinstance(data, dict):
                    for path, value in data.items():
                        current = _get_field(doc, path)
                        if current is None:
                            current = 0
                        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                            cls._set_nested(doc, path, current + value)
            elif op == "$mul":
                if isinstance(data, dict):
                    for path, value in data.items():
                        current = _get_field(doc, path)
                        if current is None:
                            current = 0
                        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                            cls._set_nested(doc, path, current * value)
            elif op == "$push":
                if isinstance(data, dict):
                    for path, value in data.items():
                        current = _get_field(doc, path)
                        if not isinstance(current, list):
                            current = []
                        current.append(value)
                        cls._set_nested(doc, path, current)
            elif op == "$pull":
                if isinstance(data, dict):
                    for path, value in data.items():
                        current = _get_field(doc, path)
                        if isinstance(current, list):
                            current = [x for x in current if x != value]
                            cls._set_nested(doc, path, current)
            elif op == "$addToSet":
                if isinstance(data, dict):
                    for path, value in data.items():
                        current = _get_field(doc, path)
                        if not isinstance(current, list):
                            current = []
                        if value not in current:
                            current.append(value)
                            cls._set_nested(doc, path, current)
