"""MogDB — lightweight document-oriented embedded database engine.

Built-in replacement for external DB dependencies. Uses an append-only
JSONL journal with periodic compaction for persistence and durability.

Quick start::

    from mogdb import MogDB

    db = MogDB("data/mogdb")
    users = db.collection("users")
    users.insert_one({"name": "Alice", "age": 30})
    result = users.find({"age": {"$gt": 25}})

With JSON sync (MogDB engine + JSON backup)::

    db = MogDB("data/mogdb", sync_dir="data/mogdb_json")
    users = db.collection("users")
    users.insert_one({"name": "Bob"})
    # -> data/mogdb/users.journal.jsonl (engine)
    # -> data/mogdb_json/users.json (human-readable sync)
"""

from .database import MogDB
from .collection import Collection, ASCENDING, DESCENDING
from .document import Document, ObjectId
from .query import match_document
from .index import Index, SortedIndex
from .json_sync import SyncableCollection

__all__ = [
    "MogDB",
    "Collection",
    "SyncableCollection",
    "Document",
    "ObjectId",
    "match_document",
    "Index",
    "SortedIndex",
    "ASCENDING",
    "DESCENDING",
]
