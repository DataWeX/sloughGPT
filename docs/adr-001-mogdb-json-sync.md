# ADR: MogDB + JSON Sync for Data Persistence

## Status

Accepted

## Context

SloughGPT needs embedded data persistence across modules (feedback, knowledge, RAG, notifications, agents, etc.). Previous approaches:

1. **Raw JSON files** — simple but no querying, race conditions on concurrent writes, entire-file rewrite on every update.
2. **SQLite** — overkill for embedded use, external dependency, async complexity.
3. **ChromaDB** — only for vector data, separate persistence needed for everything else.

MogDB provides a middle ground: embedded document database with JSONL journaling, query support, indexes, and optional JSON file sync for backward compatibility.

## Decision

Use MogDB as the primary embedded persistence layer with the following pattern:

```python
from mogdb import MogDB

class MyService:
    def __init__(self, db_path=None):
        self._db = MogDB(db_path or DEFAULT_DB_PATH)
        self._col = self._db.collection("my_data")

    def save(self, doc):
        existing = self._col.find_one({"_id": doc["id"]})
        if existing:
            self._col.update_one({"_id": doc["id"]}, {"$set": doc})
        else:
            self._col.insert_one(doc)

    def load(self, doc_id):
        return self._col.find_one({"_id": doc_id})
```

### JSON Fallback Pattern

For backward compatibility and data recovery, modules dual-write to both MogDB and legacy JSON:

```python
def _save(self):
    # MogDB (primary)
    self._col.drop()
    self._col.insert_many(data)
    # JSON (fallback)
    LEGACY_PATH.write_text(json.dumps(data))
```

### Test Isolation Pattern

Tests use `set_mogdb_path(tmp_path)` and `reset_mogdb()` to isolate from production data:

```python
@pytest.fixture
def store(tmp_path):
    set_mogdb_path(str(tmp_path / "mogdb"))
    yield MyService(db_path=str(tmp_path / "mogdb"))
    reset_mogdb()
```

## Consequences

**Positive:**
- Single dependency for all embedded persistence
- Query support (find, sort, filter) without external services
- Atomic upserts reduce race conditions
- JSONL journal enables crash recovery
- JSON fallback preserves data if MogDB is removed

**Negative:**
- Module-level singletons need explicit reset in tests
- Dual-write adds slight overhead per save
- New `_created`/`_updated` metadata fields require stripping when loading into dataclasses

## Migrated Modules

| Module | Collections | JSON Fallback |
|--------|-------------|---------------|
| `feedback/database.py` | conversations, messages, feedback, user_meta_weights | No |
| `shell/state.py` | shell_state | Yes (SyncableCollection) |
| `learner/knowledge.py` | entries, visited | Yes |
| `cognitive/rag_service.py` | documents | Yes (JSONL) |
| `mobile/notifications.py` | devices, history | No |
| `agents/run_history.py` | runs | No |
| `infrastructure/model_catalog.py` | models | Yes |

## Alternatives Considered

1. **SQLite + aiosqlite** — Rejected: async complexity, external dependency, no document model.
2. **TinyDB** — Rejected: no compaction, slower for large datasets.
3. **Pure JSON with file locking** — Rejected: no query support, full-file rewrite on every update.
