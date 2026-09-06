"""JSON sync layer for MogDB — auto-persists collections to JSON files.

Built into the MogDB engine itself. Every write operation (insert, update,
delete) immediately mirrors the collection state to a JSON file on disk.

Usage::

    from mogdb import MogDB

    # Enable JSON sync for all collections
    db = MogDB("data/mogdb", sync_dir="data/mogdb_json")

    users = db.collection("users")
    users.insert_one({"name": "Alice"})  # auto-writes to data/mogdb_json/users.json

    # Or wrap a single collection
    from mogdb.json_sync import SyncableCollection
    sync_users = SyncableCollection(users, "data/users.json")
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .collection import Collection

logger = logging.getLogger("slo.mogdb.json_sync")


class SyncableCollection:
    """Wraps a MogDB Collection and auto-syncs all writes to a JSON file.

    The JSON file is a flat list of documents. On every write mutation,
    the full collection state is dumped to the JSON file atomically
    (write-to-temp then rename for crash safety).

    Parameters
    ----------
    collection:
        The underlying MogDB Collection.
    json_path:
        Path to the JSON sync file.
    sync_mode:
        "full" — rewrite entire JSON on every write (simple, safe).
        "append" — append new docs, reload on read (faster for large sets).
        "lazy" — mark dirty and schedule background sync (best for high write volume).
    lazy_sync_interval:
        Minimum seconds between background syncs when sync_mode="lazy".
    """

    def __init__(
        self,
        collection: Collection,
        json_path: Union[str, Path],
        sync_mode: str = "full",
        lazy_sync_interval: float = 5.0,
    ):
        self._col = collection
        self._json_path = Path(json_path)
        self._sync_mode = sync_mode
        self._lock = threading.Lock()
        self._json_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy sync state
        self._dirty = False
        self._lazy_interval = lazy_sync_interval
        self._last_sync = 0.0
        self._lazy_timer: Optional[threading.Timer] = None
        self._shutdown = False
        self._batch_depth = 0  # Track nested batch() context depth
        self._batch_had_writes = False  # Track if batch had any writes

        # Bootstrap: if JSON exists but collection is empty, load from JSON
        self._bootstrap_from_json()

        # Initial sync: ensure JSON matches collection state
        if self._sync_mode != "lazy":
            self._sync_to_json()

    def _bootstrap_from_json(self) -> None:
        """Load documents from JSON file into collection if collection is empty."""
        if not self._json_path.exists():
            return
        if self._col.count() > 0:
            return
        try:
            # Try gzip first, then plain JSON
            if self._json_path.suffix == ".gz":
                import gzip
                with gzip.open(self._json_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(self._json_path) as f:
                    data = json.load(f)
            if isinstance(data, list) and data:
                self._col.insert_many(data)
                logger.debug("bootstrapped %d docs from %s", len(data), self._json_path.name)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("failed to bootstrap from %s: %s", self._json_path.name, e)

    def _sync_to_json(self) -> None:
        """Write all collection documents to the JSON file atomically."""
        docs = self._col.find()
        # Use .json.gz for compressed files, .json for plain
        if self._json_path.suffix == ".gz":
            tmp_path = self._json_path.with_suffix(".json.gz.tmp")
            try:
                import gzip
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2, default=str)
                tmp_path.replace(self._json_path)
            except OSError as e:
                logger.warning("failed to sync %s to JSON: %s", self._col.name, e)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
        else:
            tmp_path = self._json_path.with_suffix(".json.tmp")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(docs, f, indent=2, default=str)
                tmp_path.replace(self._json_path)
            except OSError as e:
                logger.warning("failed to sync %s to JSON: %s", self._col.name, e)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def _on_write(self) -> None:
        """Called after every write operation to sync to JSON.

        Skipped if inside a ``batch()`` context — sync happens on batch exit.
        """
        if self._batch_depth > 0:
            self._batch_had_writes = True
            return
        if self._sync_mode == "lazy":
            self._mark_dirty()
        else:
            with self._lock:
                self._sync_to_json()

    def _mark_dirty(self) -> None:
        """Mark collection as needing sync and schedule background sync if needed."""
        self._dirty = True
        now = time.monotonic()
        if now - self._last_sync >= self._lazy_interval:
            self._do_sync()

    def _do_sync(self) -> None:
        """Perform the actual sync and schedule next if still dirty."""
        if self._shutdown:
            return
        with self._lock:
            self._sync_to_json()
            self._last_sync = time.monotonic()
            self._dirty = False

        # Schedule next sync if still dirty
        if self._dirty and not self._shutdown:
            self._lazy_timer = threading.Timer(self._lazy_interval, self._do_sync)
            self._lazy_timer.daemon = True
            self._lazy_timer.start()

    # ------------------------------------------------------------------
    # Proxied CRUD — all writes trigger JSON sync
    # ------------------------------------------------------------------

    def insert_one(self, doc: Dict[str, Any]) -> str:
        doc_id = self._col.insert_one(doc)
        self._on_write()
        return doc_id

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        ids = self._col.insert_many(docs)
        self._on_write()
        return ids

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        count = self._col.update_one(query, update)
        if count:
            self._on_write()
        return count

    def update_many(self, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        count = self._col.update_many(query, update)
        if count:
            self._on_write()
        return count

    def delete_one(self, query: Dict[str, Any]) -> int:
        count = self._col.delete_one(query)
        if count:
            self._on_write()
        return count

    def delete_many(self, query: Dict[str, Any]) -> int:
        count = self._col.delete_many(query)
        if count:
            self._on_write()
        return count

    def find_one_and_update(
        self, query: Dict[str, Any], update: Dict[str, Any], return_document: str = "before"
    ) -> Optional[Dict[str, Any]]:
        result = self._col.find_one_and_update(query, update, return_document)
        if result is not None:
            self._on_write()
        return result

    def find_one_and_replace(
        self, query: Dict[str, Any], replacement: Dict[str, Any], return_document: str = "before"
    ) -> Optional[Dict[str, Any]]:
        result = self._col.find_one_and_replace(query, replacement, return_document)
        if result is not None:
            self._on_write()
        return result

    def find_one_and_delete(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = self._col.find_one_and_delete(query)
        if result is not None:
            self._on_write()
        return result

    def drop(self) -> None:
        self._col.drop()
        self._on_write()

    # ------------------------------------------------------------------
    # Batch writes — defer sync until context exits
    # ------------------------------------------------------------------

    def batch(self):
        """Context manager that defers JSON sync until the block exits.

        All writes inside the block are applied to the in-memory collection
        immediately, but JSON persistence happens only once on exit.

        Usage::

            with sync_col.batch():
                for item in items:
                    sync_col.insert_one(item)
        """
        return _BatchContext(self)

    # ------------------------------------------------------------------
    # Read-only proxied methods — no sync needed
    # ------------------------------------------------------------------

    def find(self, query=None, sort=None, limit=None, skip=0, projection=None) -> List[Dict[str, Any]]:
        return self._col.find(query, sort=sort, limit=limit, skip=skip, projection=projection)

    def find_one(self, query=None, projection=None) -> Optional[Dict[str, Any]]:
        return self._col.find_one(query, projection=projection)

    def count(self, query=None) -> int:
        return self._col.count(query)

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._col.aggregate(pipeline)

    # ------------------------------------------------------------------
    # Indexing — proxied, no JSON sync needed
    # ------------------------------------------------------------------

    def create_index(self, field: str, unique: bool = False):
        return self._col.create_index(field, unique=unique)

    def create_sorted_index(self, field: str):
        return self._col.create_sorted_index(field)

    def create_ttl_index(self, field: str, expire_after_seconds: int) -> None:
        self._col.create_ttl_index(field, expire_after_seconds)

    def drop_index(self, field: str) -> None:
        self._col.drop_index(field)

    # ------------------------------------------------------------------
    # Manual sync
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Force a manual sync to JSON."""
        with self._lock:
            self._sync_to_json()
            self._last_sync = time.monotonic()
            self._dirty = False

    def close(self) -> None:
        """Shut down lazy sync timer and flush any pending writes."""
        self._shutdown = True
        if self._lazy_timer:
            self._lazy_timer.cancel()
            self._lazy_timer = None
        if self._dirty:
            self._sync_to_json()
            self._dirty = False

    def reload(self) -> None:
        """Force reload from JSON file into collection."""
        with self._lock:
            self._col.drop()
            self._bootstrap_from_json()

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def underlying(self) -> Collection:
        """Access the raw MogDB Collection."""
        return self._col


class _BatchContext:
    """Context manager for batch writes — defers JSON sync until exit."""

    def __init__(self, syncable: SyncableCollection):
        self._syncable = syncable

    def __enter__(self):
        self._syncable._batch_depth += 1
        self._syncable._batch_had_writes = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._syncable._batch_depth -= 1
        if exc_type is None and self._syncable._batch_had_writes:
            self._syncable._batch_had_writes = False
            self._syncable._on_write()
        return False
