# MogDB Usage Guide

## Overview

MogDB is the embedded document database for SloughGPT. It stores data as JSON documents with JSONL journaling and periodic compaction.

## Quick Start

```python
from mogdb import MogDB

# Open or create a database
db = MogDB("data/my_collection")

# Get a collection
col = db.collection("items")

# Insert documents
col.insert_one({"name": "item1", "value": 42})
col.insert_many([{"name": "item2"}, {"name": "item3"}])

# Query documents
doc = col.find_one({"name": "item1"})
docs = col.find({"value": {"$gte": 10}})

# Update documents
col.update_one({"name": "item1"}, {"$set": {"value": 100}})

# Delete documents
col.delete_one({"name": "item1"})

# Aggregate
results = col.aggregate([
    {"$group": {"_id": "$status", "count": {"$sum": 1}}},
])
```

## SyncableCollection

For data that needs human-readable JSON backup:

```python
from mogdb.json_sync import SyncableCollection

# Create with JSON sync
col = SyncableCollection(
    name="my_data",
    sync_dir="data/my_data_json",  # JSON files written here
    lazy=True,  # Background sync (default)
)

# Same API as regular Collection
col.insert_one({"key": "value"})
docs = col.find()

# Force sync
col.sync()

# Batch operations (more efficient)
with col.batch() as batch:
    for i in range(1000):
        batch.insert_one({"index": i})
```

## TTL Indexes

Auto-expire documents after a time period:

```python
# Create TTL index (documents expire after 30 days)
col.create_ttl_index("created_at", expire_after_seconds=30 * 24 * 3600)

# Note: TTL field must be numeric (epoch seconds), not ISO string
import time
col.insert_one({"data": "value", "created_at": time.time()})
```

## Query Cache

Cache frequently accessed queries:

```python
from mogdb.cache import QueryCache

cache = QueryCache(default_ttl=5.0, max_size=100)

# Cache a query result
result = cache.get_or_set(
    "my_query",
    lambda: col.find({"status": "active"}),
    ttl=10.0,
)

# Invalidate cache
cache.invalidate("my_query")
cache.invalidate_pattern("my_*")  # Pattern matching
```

## Batch Lookups

Efficiently fetch multiple documents by ID:

```python
# Instead of N queries:
# for id in ids:
#     col.find_one({"_id": id})

# Use batch lookup:
docs = col.find_by_ids(ids)
```

## Best Practices

1. **Use `find_by_ids()` for bulk reads** - Avoids N+1 query patterns
2. **Use `batch()` context for bulk writes** - Groups inserts for efficiency
3. **Use `lazy=True` for SyncableCollection** - Background sync prevents blocking
4. **Set TTL indexes on numeric fields** - ISO strings won't expire
5. **Use QueryCache for hot paths** - Reduces repeated queries
6. **Handle exceptions gracefully** - MogDB falls back to JSONL on errors

## Migration from JSON Files

Use the migration script:

```bash
# Migrate all legacy JSON files
python scripts/migrate_legacy_json.py

# Or use CLI commands
slo db status      # Show collection stats
slo db sync        # Force JSON sync
slo db migrate     # Run migration
```

## Performance

Based on benchmarks (1K documents):
- Insert: ~6K docs/sec
- Find all: ~1.5M docs/sec
- Find one: ~700 ops/sec
- Update: ~2.8K ops/sec
- Aggregation: ~2ms

Gzip compression reduces JSON sync file size to ~30% of plain JSON.
