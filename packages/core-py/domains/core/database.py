"""
Minimal in-memory document store providing the ``get_db()`` interface
used by :mod:`domains.billing.token_service` and other modules.

Provides ``find(collection, query)``, ``upsert(collection, query, data)``,
and ``insert(collection, data)``.

This is an in-memory implementation.  Data is lost on process restart.
Replace with a real database (SQLite, PostgreSQL, etc.) when persistence
is required.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List


class _MemoryDB:
    """Thread-safe in-memory document store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collections: Dict[str, List[Dict[str, Any]]] = {}

    def find(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self._lock:
            docs = self._collections.get(collection, [])
        if not query:
            return list(docs)
        return [d for d in docs if all(d.get(k) == v for k, v in query.items())]

    def insert(self, collection: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._collections.setdefault(collection, []).append(dict(data))

    def upsert(self, collection: str, query: Dict[str, Any], data: Dict[str, Any]) -> None:
        with self._lock:
            docs = self._collections.setdefault(collection, [])
            for doc in docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    doc.update(data)
                    return
            merged = dict(query)
            merged.update(data)
            docs.append(merged)

    def count(self, collection: str) -> int:
        with self._lock:
            return len(self._collections.get(collection, []))


_db: _MemoryDB | None = None


def get_db() -> _MemoryDB:
    """Return the singleton in-memory database."""
    global _db
    if _db is None:
        _db = _MemoryDB()
    return _db
