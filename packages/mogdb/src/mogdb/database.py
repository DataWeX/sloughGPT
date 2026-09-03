"""MogDB — top-level database interface.

A MogDB instance manages a directory of collection journals and provides
a factory for named collections. Discovers collections on disk that may
not have been loaded in the current session.

When ``sync_dir`` is provided, all collections are automatically wrapped
with ``SyncableCollection`` which mirrors every write to a JSON file.
This gives you MogDB as the engine (append-only journal, compaction,
indexes, TTL) with JSON as a human-readable sync/backup.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

from .collection import Collection

logger = logging.getLogger("slo.mogdb")


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
    sync_dir:
        When set, all collections are wrapped with ``SyncableCollection``
        that auto-syncs writes to JSON files in this directory. The JSON
        files are named ``{collection_name}.json``. On startup, if the
        JSON file exists but the collection is empty, documents are
        loaded from JSON (bootstrap).
    """

    def __init__(
        self,
        path: str,
        compact_on_close: bool = True,
        sync_dir: Optional[Union[str, Path]] = None,
    ):
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._compact_on_close = compact_on_close
        self._sync_dir = Path(sync_dir) if sync_dir else None
        if self._sync_dir:
            self._sync_dir.mkdir(parents=True, exist_ok=True)
        self._collections: Dict[str, Collection] = {}
        self._synced_collections: Dict[str, object] = {}
        self._lock = threading.Lock()

    def _discover_collections(self) -> List[str]:
        """Find collection names from journal/compacted files on disk."""
        names = set()
        for f in self._root.iterdir():
            if f.suffix == ".jsonl" and f.stem.endswith(".journal"):
                names.add(f.stem.replace(".journal", ""))
            elif f.suffix == ".mogdb":
                names.add(f.stem)
        return sorted(names)

    def collection(
        self,
        name: str,
        max_size_bytes: Optional[int] = None,
        max_count: Optional[int] = None,
    ):
        """Get or create a named collection.

        When ``sync_dir`` is set, returns a ``SyncableCollection`` that
        auto-syncs all writes to ``{sync_dir}/{name}.json``.

        Parameters
        ----------
        name:
            Collection name.
        max_size_bytes:
            Maximum size in bytes for capped collections (None = uncapped).
        max_count:
            Maximum document count for capped collections (None = unlimited).
        """
        with self._lock:
            if self._sync_dir:
                if name not in self._synced_collections:
                    raw = Collection(
                        name, self._root,
                        max_size_bytes=max_size_bytes,
                        max_count=max_count,
                    )
                    from .json_sync import SyncableCollection
                    json_path = self._sync_dir / f"{name}.json"
                    self._synced_collections[name] = SyncableCollection(raw, json_path)
                return self._synced_collections[name]
            else:
                if name not in self._collections:
                    self._collections[name] = Collection(
                        name, self._root,
                        max_size_bytes=max_size_bytes,
                        max_count=max_count,
                    )
                return self._collections[name]

    def drop_collection(self, name: str) -> None:
        """Drop a collection and all its data."""
        with self._lock:
            if self._synced_collections:
                sc = self._synced_collections.pop(name, None)
                if sc:
                    sc.drop()
                    return
            if name in self._collections:
                self._collections[name].drop()
                del self._collections[name]
            else:
                journal = self._root / f"{name}.journal.jsonl"
                compacted = self._root / f"{name}.mogdb"
                for p in [journal, compacted]:
                    if p.exists():
                        p.unlink()

    def list_collections(self) -> List[str]:
        """Return list of collection names (including disk-only)."""
        with self._lock:
            in_memory = set(self._collections.keys()) | set(self._synced_collections.keys())
            on_disk = set(self._discover_collections())
            return sorted(in_memory | on_disk)

    def compact_all(self) -> int:
        """Compact all collections. Returns total document count."""
        total = 0
        with self._lock:
            for coll in self._collections.values():
                total += coll.compact()
        return total

    def sync_all(self) -> int:
        """Force JSON sync for all synced collections. Returns count."""
        count = 0
        with self._lock:
            for sc in self._synced_collections.values():
                sc.sync()
                count += 1
        return count

    def close(self) -> None:
        """Close the database (compacts if configured, syncs if configured)."""
        if self._sync_dir:
            try:
                self.sync_all()
            except Exception as exc:
                logger.warning("sync on close failed: %s", exc)
        if self._compact_on_close:
            try:
                self.compact_all()
            except Exception as exc:
                logger.warning("compact on close failed: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
