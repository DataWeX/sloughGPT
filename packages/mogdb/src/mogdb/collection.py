"""Collection CRUD for MogDB.

Each collection manages an in-memory document map backed by an append-only
JSONL journal. Writes are journaled immediately; reads come from memory.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .document import Document
from .query import match_document

logger = logging.getLogger("man.mogdb")


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
    # CRUD
    # ------------------------------------------------------------------

    def insert_one(self, doc: Dict[str, Any]) -> str:
        """Insert a single document. Returns its ``_id``."""
        d = Document(doc)
        with self._lock:
            self._docs[d.id] = d
            self._journal("insert", dict(d))
        return d.id

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents. Returns their ``_id``s."""
        ids: List[str] = []
        for doc in docs:
            ids.append(self.insert_one(doc))
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

        Supports ``$set`` and ``$unset`` operators.
        Returns the number of modified documents (0 or 1).
        """
        with self._lock:
            for doc in self._docs.values():
                if not match_document(doc, query):
                    continue
                self._apply_update(doc, update)
                doc["_updated"] = __import__("time").time()
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
                self._apply_update(doc, update)
                doc["_updated"] = __import__("time").time()
                count += 1
            if count:
                self._journal("update_many", {"query": query, "update": update})
        return count

    def delete_one(self, query: Dict[str, Any]) -> int:
        """Delete the first document matching *query*."""
        with self._lock:
            for doc_id, doc in list(self._docs.items()):
                if match_document(doc, query):
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
                    del self._docs[doc_id]
                    count += 1
            if count:
                self._journal("delete_many", {"query": query, "count": count})
        return count

    def drop(self) -> None:
        """Remove all documents and journal files."""
        with self._lock:
            self._docs.clear()
            self._dirty = False
            for p in [self._journal_path, self._compacted_path]:
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_update(doc: Document, update: Dict[str, Any]) -> None:
        for op, data in update.items():
            if op == "$set":
                if isinstance(data, dict):
                    doc.update(data)
            elif op == "$unset":
                if isinstance(data, dict):
                    for key in data:
                        doc.pop(key, None)
