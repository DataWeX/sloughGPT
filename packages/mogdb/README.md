# mogdb — Lightweight Document Database

MongoDB-compatible embedded document database for Python. JSONL journaling, TCP server, client library, and CLI tools.

## Quick Start

```python
from mogdb import MogDB

db = MogDB("data/mogdb")
users = db.collection("users")
users.insert_one({"name": "Alice", "age": 30})
users.find({"age": {"$gt": 25}})
```

## Features

- Document model with MongoDB-style query operators (`$gt`, `$lt`, `$in`, `$regex`, etc.)
- JSONL persistence with journaling and compaction
- In-memory + hash/sorted indexes
- Thread-safe concurrent access
- TCP server with auth, rate-limiting, CLI tools
- Zero external dependencies

## CLI

```bash
mogdb-server --port 27017 --dbpath data/mogdb
mogdb-client --host localhost --port 27017
```
