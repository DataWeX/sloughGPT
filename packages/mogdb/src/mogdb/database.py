"""MogDB — top-level database interface.

A MogDB instance manages a directory of collection journals and provides
a factory for named collections.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from .collection import Collection

logger = logging.getLogger("man.mogdb")


class MogDB:
    """Document-oriented embedded database.

    Each database is a directory on disk containing journal files
    (``*.journal.jsonl``) and optional compacted snapshots (``*.mogdb``).

    Parameters
    ----------
    path:
        Filesystem path for this database. Created automatically.
    compact_on_close:
        If True, the journal is compacted when ``close()`` or the context
        manager exits.
    """

    def __init__(self, path: str, compact_on_close: bool = True):
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._compact_on_close = compact_on_close
        self._collections: Dict[str, Collection] = {}
        self._lock = threading.Lock()

    def collection(self, name: str) -> Collection:
        """Get or create a named collection."""
        with self._lock:
            if name not in self._collections:
                self._collections[name] = Collection(name, self._root)
            return self._collections[name]

    def drop_collection(self, name: str) -> None:
        """Drop a collection and all its data."""
        with self._lock:
            if name in self._collections:
                self._collections[name].drop()
                del self._collections[name]

    def list_collections(self) -> list:
        """Return list of collection names."""
        with self._lock:
            return list(self._collections.keys())

    def compact_all(self) -> int:
        """Compact all collections. Returns total document count."""
        total = 0
        with self._lock:
            for coll in self._collections.values():
                total += coll.compact()
        return total

    def close(self) -> None:
        """Close the database (compacts if configured)."""
        if self._compact_on_close:
            try:
                self.compact_all()
            except Exception as exc:
                logger.warning("compact on close failed: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
