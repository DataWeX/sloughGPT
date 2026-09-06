"""Collection CRUD for MogDB.

Each collection manages an in-memory document map backed by an append-only
JSONL journal. Writes are journaled immediately; reads come from memory.
Supports MongoDB-style update operators ($set, $unset, $inc, $push, $pull,
$addToSet, $mul) with dot-notation for nested fields.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .document import Document
from .index import Index, SortedIndex
from .query import _get_field, match_document

logger = logging.getLogger("slo.mogdb")

ASCENDING = 1
DESCENDING = -1


class Collection:
    """A named document collection within a MogDB database.

    Parameters
    ----------
    name:
        Collection name (used as the journal filename stem).
    db_path:
        Directory path for the journal file.
    max_size_bytes:
        Maximum size in bytes for capped collections (None = uncapped).
    max_count:
        Maximum document count for capped collections (None = unlimited).
    """

    def __init__(
        self,
        name: str,
        db_path: Path,
        max_size_bytes: Optional[int] = None,
        max_count: Optional[int] = None,
    ):
        self.name = name
        self._db_path = db_path
        self._docs: Dict[str, Document] = {}
        self._lock = threading.Lock()
        self._journal_path = db_path / f"{name}.journal.jsonl"
        self._compacted_path = db_path / f"{name}.mogdb"
        self._dirty: bool = False
        self._indexes: Dict[str, Index] = {}
        self._sorted_indexes: Dict[str, SortedIndex] = {}
        self._ttl_index: Optional[str] = None
        self._ttl_seconds: Optional[int] = None
        self._max_size_bytes = max_size_bytes
        self._max_count = max_count
        self._last_expire_check: float = 0

        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load documents from the most recent snapshot or journal.

        Compacted snapshots are plain documents written one per line. Journal
        files are an operation log and are replayed in order so that
        ``update``/``delete`` entries apply to previously inserted documents
        instead of overwriting them with the raw op payload.
        """
        data_path = self._compacted_path if self._compacted_path.exists() else self._journal_path
        if not data_path.exists():
            return

        is_journal = data_path.name.endswith(".journal.jsonl")
        count = 0
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not is_journal:
                    doc = Document(entry)
                    self._docs[doc.id] = doc
                    count += 1
                    continue

                op = entry.get("op")
                data = entry.get("data")
                if not isinstance(data, dict):
                    continue
                if op == "insert":
                    doc = Document(data)
                    self._docs[doc.id] = doc
                    count += 1
                elif op == "update":
                    doc = self._docs.get(data.get("_id"))
                    if doc is None:
                        continue
                    self._apply_update(doc, data.get("update") or {})
                    doc["_updated"] = time.time()
                    count += 1
                elif op == "update_many":
                    update = data.get("update") or {}
                    query = data.get("query") or {}
                    for doc in self._docs.values():
                        if match_document(doc, query):
                            self._apply_update(doc, update)
                            doc["_updated"] = time.time()
                            count += 1
                elif op == "delete":
                    self._docs.pop(data.get("_id"), None)
                elif op == "delete_many":
                    query = data.get("query") or {}
                    for doc_id in [
                        d.id for d in self._docs.values() if match_document(d, query)
                    ]:
                        self._docs.pop(doc_id, None)
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
        """Create a hash index on *field*. Returns the Index object.

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

    def create_sorted_index(self, field: str) -> SortedIndex:
        """Create a sorted index on *field* for range queries.

        Existing documents are indexed. Returns the SortedIndex object.
        """
        if field in self._sorted_indexes:
            return self._sorted_indexes[field]
        idx = SortedIndex(field)
        self._sorted_indexes[field] = idx
        with self._lock:
            for doc in self._docs.values():
                val = _get_field(doc, field)
                if val is not None:
                    idx.add(doc.id, val)
        return idx

    def create_ttl_index(self, field: str, expire_after_seconds: int) -> None:
        """Create a TTL index that auto-expires documents.

        Documents whose *field* value plus *expire_after_seconds* is in the
        past are removed on access. Only one TTL index is supported per
        collection.
        """
        self._ttl_index = field
        self._ttl_seconds = expire_after_seconds
        self.create_index(field)

    def drop_index(self, field: str) -> None:
        """Drop the index on *field*."""
        self._indexes.pop(field, None)
        self._sorted_indexes.pop(field, None)
        if self._ttl_index == field:
            self._ttl_index = None
            self._ttl_seconds = None

    def _index_insert(self, doc: Document) -> None:
        """Update all indexes after an insert."""
        for field, idx in self._indexes.items():
            val = _get_field(doc, field)
            idx.add(doc.id, val)
        for field, idx in self._sorted_indexes.items():
            val = _get_field(doc, field)
            if val is not None:
                idx.add(doc.id, val)

    def _index_update(self, doc: Document, old_vals: Dict[str, Any]) -> None:
        """Update all indexes after an update."""
        for field, idx in self._indexes.items():
            old_val = old_vals.get(field)
            new_val = _get_field(doc, field)
            idx.update(doc.id, old_val, new_val)
        for field, idx in self._sorted_indexes.items():
            old_val = old_vals.get(field)
            new_val = _get_field(doc, field)
            if old_val is not None:
                idx.remove(doc.id, old_val)
            if new_val is not None:
                idx.add(doc.id, new_val)

    def _index_remove(self, doc: Document) -> None:
        """Update all indexes after a delete."""
        for field, idx in self._indexes.items():
            val = _get_field(doc, field)
            idx.remove(doc.id, val)
        for field, idx in self._sorted_indexes.items():
            val = _get_field(doc, field)
            if val is not None:
                idx.remove(doc.id, val)

    # ------------------------------------------------------------------
    # TTL expiration
    # ------------------------------------------------------------------

    def _expire_documents(self) -> None:
        """Remove expired documents based on TTL index.

        Checks at most once per 60 seconds.
        """
        if not self._ttl_index or self._ttl_seconds is None:
            return
        now = time.time()
        if now - self._last_expire_check < 60:
            return
        self._last_expire_check = now
        expired_ids: List[str] = []
        with self._lock:
            for doc in self._docs.values():
                ts = _get_field(doc, self._ttl_index)
                if isinstance(ts, (int, float)) and (now - ts) > self._ttl_seconds:
                    expired_ids.append(doc.id)
            for doc_id in expired_ids:
                doc = self._docs.pop(doc_id, None)
                if doc:
                    self._index_remove(doc)
        if expired_ids:
            self._journal("delete_many", {"query": {}, "count": len(expired_ids)})
            logger.debug("expired %d documents in %s", len(expired_ids), self.name)

    # ------------------------------------------------------------------
    # capped collection
    # ------------------------------------------------------------------

    def _cap_if_needed(self) -> None:
        """Enforce capped collection limits after insert."""
        if self._max_count is not None:
            while len(self._docs) > self._max_count:
                # Remove oldest document (lowest _created)
                oldest_id = min(self._docs, key=lambda k: self._docs[k].get("_created", 0))
                doc = self._docs.pop(oldest_id)
                self._index_remove(doc)
                self._journal("delete", {"_id": oldest_id})
        if self._max_size_bytes is not None:
            try:
                size = self._journal_path.stat().st_size if self._journal_path.exists() else 0
                compacted_size = self._compacted_path.stat().st_size if self._compacted_path.exists() else 0
                total = size + compacted_size
                while total > self._max_size_bytes and len(self._docs) > 1:
                    oldest_id = min(self._docs, key=lambda k: self._docs[k].get("_created", 0))
                    doc = self._docs.pop(oldest_id)
                    self._index_remove(doc)
                    self._journal("delete", {"_id": oldest_id})
                    size = self._journal_path.stat().st_size if self._journal_path.exists() else 0
                    compacted_size = self._compacted_path.stat().st_size if self._compacted_path.exists() else 0
                    total = size + compacted_size
            except OSError:
                pass

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_projection(doc: Dict[str, Any], projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply projection to a document.

        Projection modes:
        - None: return all fields
        - Inclusion: {"field": 1, ...} — return only listed fields + _id
        - Exclusion: {"field": 0, ...} — return all fields except listed
        - Mixed not allowed (MongoDB behavior).
        """
        if not projection:
            return dict(doc)

        inclusion = {k for k, v in projection.items() if v == 1}
        exclusion = {k for k, v in projection.items() if v == 0}

        if inclusion:
            # Inclusion mode: return only specified fields + _id
            result = {"_id": doc.get("_id")}
            for field in inclusion:
                if field == "_id":
                    continue
                val = _get_field(doc, field)
                if val is not None or field in doc:
                    result[field] = val
            return result
        elif exclusion:
            # Exclusion mode: return all except specified
            return {k: v for k, v in doc.items() if k not in exclusion}
        return dict(doc)

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
        self._cap_if_needed()
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
        self._cap_if_needed()
        return ids

    def find(
        self,
        query: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
        projection: Optional[Dict[str, Any]] = None,
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
        projection:
            Fields to include/exclude. ``{"name": 1}`` returns only name
            (+ _id). ``{"secret": 0}`` returns all except secret.

        Returns a list of plain dict copies.
        """
        self._expire_documents()

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

        if projection:
            results = [self._apply_projection(r, projection) for r in results]

        return results

    def find_one(
        self,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the first document matching *query*, or ``None``."""
        results = self.find(query, limit=1, projection=projection)
        return results[0] if results else None

    def find_by_ids(
        self,
        ids: List[str],
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-fetch documents by their ``_id`` values.

        More efficient than N individual ``find_one`` calls — acquires
        the lock once and does a single pass over the document map.
        """
        if not ids:
            return []
        id_set = set(ids)
        self._expire_documents()
        with self._lock:
            results = [dict(d) for d in self._docs.values() if d.id in id_set]
        if projection:
            results = [self._apply_projection(r, projection) for r in results]
        return results

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
            for idx in self._sorted_indexes.values():
                idx.clear()
            self._dirty = False
            for p in [self._journal_path, self._compacted_path]:
                if p.exists():
                    p.unlink()

    # ------------------------------------------------------------------
    # atomic find-and-modify
    # ------------------------------------------------------------------

    def find_one_and_update(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
        return_document: str = "before",
    ) -> Optional[Dict[str, Any]]:
        """Find the first matching document, apply update, return it.

        Parameters
        ----------
        query:
            MongoDB-style query dict.
        update:
            Update operators ($set, $inc, etc.).
        return_document:
            ``"before"`` returns the document before modification.
            ``"after"`` returns the document after modification.

        Returns the document or None if no match.
        """
        with self._lock:
            for doc in self._docs.values():
                if not match_document(doc, query):
                    continue
                old_vals = {f: _get_field(doc, f) for f in self._indexes}
                snapshot = dict(doc) if return_document == "before" else None
                self._apply_update(doc, update)
                doc["_updated"] = time.time()
                self._index_update(doc, old_vals)
                self._journal("update", {"_id": doc.id, "update": update})
                if return_document == "before":
                    return snapshot
                return dict(doc)
        return None

    def find_one_and_replace(
        self,
        query: Dict[str, Any],
        replacement: Dict[str, Any],
        return_document: str = "before",
    ) -> Optional[Dict[str, Any]]:
        """Find the first matching document, replace it entirely, return it.

        Parameters
        ----------
        query:
            MongoDB-style query dict.
        replacement:
            The new document (must not contain update operators).
        return_document:
            ``"before"`` returns the document before replacement.
            ``"after"`` returns the document after replacement.

        Returns the document or None if no match.
        """
        with self._lock:
            for doc_id, doc in list(self._docs.items()):
                if not match_document(doc, query):
                    continue
                old_vals = {f: _get_field(doc, f) for f in self._indexes}
                snapshot = dict(doc) if return_document == "before" else None
                # Replace: keep _id and _created, overwrite everything else
                new_data = {k: v for k, v in replacement.items() if not k.startswith("_")}
                for key in list(doc.keys()):
                    if key not in ("_id", "_created"):
                        del doc[key]
                doc.update(new_data)
                doc["_updated"] = time.time()
                self._index_update(doc, old_vals)
                self._journal("update", {"_id": doc_id, "update": {"$set": new_data}})
                if return_document == "before":
                    return snapshot
                return dict(doc)
        return None

    def find_one_and_delete(
        self,
        query: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find the first matching document, delete it, return the deleted doc.

        Returns the deleted document or None if no match.
        """
        with self._lock:
            for doc_id, doc in list(self._docs.items()):
                if match_document(doc, query):
                    snapshot = dict(doc)
                    self._index_remove(doc)
                    del self._docs[doc_id]
                    self._journal("delete", {"_id": doc_id})
                    return snapshot
        return None

    # ------------------------------------------------------------------
    # aggregation pipeline
    # ------------------------------------------------------------------

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run an aggregation pipeline.

        Supported stages:
        - ``$match``: filter documents (same syntax as find query)
        - ``$project``: include/exclude/compute fields
        - ``$group``: group by _id with accumulator operators
        - ``$sort``: sort by fields
        - ``$skip``: skip N documents
        - ``$limit``: limit to N documents
        - ``$unwind``: deconstruct an array field

        Parameters
        ----------
        pipeline:
            List of stage dicts.

        Returns a list of result documents.
        """
        self._expire_documents()

        with self._lock:
            docs = [dict(d) for d in self._docs.values()]

        for stage in pipeline:
            if "$match" in stage:
                query = stage["$match"]
                docs = [d for d in docs if match_document(d, query)]

            elif "$project" in stage:
                proj = stage["$project"]
                docs = [self._apply_projection(d, proj) for d in docs]

            elif "$sort" in stage:
                sort_spec = stage["$sort"]
                for field, direction in reversed(
                    list(sort_spec.items()) if isinstance(sort_spec, dict) else sort_spec
                ):
                    docs.sort(
                        key=lambda r, f=field: r.get(f) if f in r else "",
                        reverse=(direction == -1),
                    )

            elif "$skip" in stage:
                n = stage["$skip"]
                docs = docs[n:]

            elif "$limit" in stage:
                n = stage["$limit"]
                docs = docs[:n]

            elif "$unwind" in stage:
                field = stage["$unwind"]
                if isinstance(field, str):
                    field = field.lstrip("$")
                expanded: List[Dict[str, Any]] = []
                for d in docs:
                    arr = d.get(field)
                    if isinstance(arr, list):
                        if not arr:
                            # Preserve document with null for empty arrays
                            copy = {k: v for k, v in d.items()}
                            copy[field] = None
                            expanded.append(copy)
                        else:
                            for val in arr:
                                copy = {k: v for k, v in d.items()}
                                copy[field] = val
                                expanded.append(copy)
                    else:
                        expanded.append(d)
                docs = expanded

            elif "$group" in stage:
                group_spec = stage["$group"]
                group_id = group_spec.get("_id")
                # Group documents
                groups: Dict[Any, List[Dict[str, Any]]] = {}
                for d in docs:
                    if group_id is None:
                        key = None
                    elif isinstance(group_id, str) and group_id.startswith("$"):
                        key = d.get(group_id[1:])
                    else:
                        key = group_id
                    groups.setdefault(key, []).append(d)

                result: List[Dict[str, Any]] = []
                for key, group_docs in groups.items():
                    out: Dict[str, Any] = {"_id": key}
                    for accum_field, accum_spec in group_spec.items():
                        if accum_field == "_id":
                            continue
                        if not isinstance(accum_spec, dict):
                            continue
                        op = list(accum_spec.keys())[0] if accum_spec else None
                        val_field = accum_spec[op] if op else None

                        if op == "$sum":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                out[accum_field] = sum(
                                    (d.get(field_name, 0) or 0) for d in group_docs
                                    if isinstance(d.get(field_name), (int, float))
                                )
                            else:
                                # Literal number or no field = count
                                out[accum_field] = len(group_docs)
                        elif op == "$avg":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                nums = [
                                    d.get(field_name, 0)
                                    for d in group_docs
                                    if isinstance(d.get(field_name), (int, float))
                                ]
                                out[accum_field] = sum(nums) / len(nums) if nums else 0
                            else:
                                out[accum_field] = 0
                        elif op == "$min":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                vals = [d.get(field_name) for d in group_docs if d.get(field_name) is not None]
                                out[accum_field] = min(vals) if vals else None
                            else:
                                out[accum_field] = None
                        elif op == "$max":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                vals = [d.get(field_name) for d in group_docs if d.get(field_name) is not None]
                                out[accum_field] = max(vals) if vals else None
                            else:
                                out[accum_field] = None
                        elif op == "$first":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                out[accum_field] = group_docs[0].get(field_name) if group_docs else None
                            else:
                                out[accum_field] = group_docs[0].get(val_field) if group_docs else None
                        elif op == "$last":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                out[accum_field] = group_docs[-1].get(field_name) if group_docs else None
                            else:
                                out[accum_field] = group_docs[-1].get(val_field) if group_docs else None
                        elif op == "$push":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                out[accum_field] = [d.get(field_name) for d in group_docs]
                            else:
                                out[accum_field] = list(group_docs)
                        elif op == "$addToSet":
                            if isinstance(val_field, str) and val_field.startswith("$"):
                                field_name = val_field[1:]
                                seen_set: list = []
                                for d in group_docs:
                                    v = d.get(field_name)
                                    if v not in seen_set:
                                        seen_set.append(v)
                                out[accum_field] = seen_set
                            else:
                                out[accum_field] = list({d.get("_id") for d in group_docs})
                    result.append(out)
                docs = result

        return docs

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
