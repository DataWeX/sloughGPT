# mogdb — Lightweight Document Database

Embedded document database for Python. Zero dependencies. JSONL journaling with compaction.

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
- Zero external dependencies
