"""MogDB — lightweight document-oriented embedded database engine.

Built-in replacement for external DB dependencies. Uses an append-only
JSONL journal with periodic compaction for persistence and durability.

Quick start (embedded)::

    from mogdb import MogDB

    db = MogDB("data/mogdb")
    users = db.collection("users")
    users.insert_one({"name": "Alice", "age": 30})

Quick start (client-server)::

    from mogdb import MogDBClient

    client = MogDBClient("localhost", 27017)
    client.connect()
    client.auth("password")
    users = client.collection("users")
    result = users.find({"age": {"$gt": 25}})
    client.close()
"""

from .database import MogDB
from .collection import Collection
from .document import Document, ObjectId
from .query import match_document
from .index import Index, SortedIndex
from .client import MogDBClient, MogDBError, RemoteCollection
from .server import MogDBServer

__all__ = [
    "MogDB",
    "Collection",
    "Document",
    "ObjectId",
    "match_document",
    "Index",
    "SortedIndex",
    "MogDBClient",
    "MogDBError",
    "RemoteCollection",
    "MogDBServer",
]
