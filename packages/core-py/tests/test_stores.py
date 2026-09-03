"""Tests for domains.collections.stores — MemoryStore, FileStore, CallbackStore, etc."""

from __future__ import annotations

import pytest
from pathlib import Path

from domains.collections.sources import Record
from domains.collections.stores import (
    MemoryStore,
    FileStore,
    CallbackStore,
    ChainedStore,
    StatsStore,
)


# ── MemoryStore ───────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_write_and_read(self):
        store = MemoryStore()
        store.write(Record(content="hello"))
        records = list(store.read_all())
        assert len(records) == 1
        assert records[0].content == "hello"

    def test_count(self):
        store = MemoryStore()
        assert store.count() == 0
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        assert store.count() == 2

    def test_max_size_eviction(self):
        store = MemoryStore(max_size=3)
        for i in range(5):
            store.write(Record(content=str(i)))
        assert store.count() == 3
        records = list(store.read_all())
        assert records[0].content == "2"  # oldest kept

    def test_take(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        taken = store.take(1)
        assert len(taken) == 1
        assert taken[0].content == "a"
        assert store.count() == 1

    def test_take_more_than_available(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        taken = store.take(10)
        assert len(taken) == 1

    def test_peek(self):
        store = MemoryStore()
        for i in range(5):
            store.write(Record(content=str(i)))
        peeked = store.peek(2)
        assert len(peeked) == 2
        assert peeked[0].content == "0"
        assert store.count() == 5  # peek doesn't remove

    def test_clear(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.clear()
        assert store.count() == 0


# ── FileStore ─────────────────────────────────────────────────────────────────

class TestFileStore:
    def test_write_and_read(self, tmp_path):
        store = FileStore(str(tmp_path / "test.jsonl"))
        store.write(Record(content="hello"))
        records = list(store.read_all())
        assert len(records) == 1
        assert records[0].content == "hello"

    def test_count(self, tmp_path):
        store = FileStore(str(tmp_path / "test.jsonl"))
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        assert store.count() == 2

    def test_clear(self, tmp_path):
        store = FileStore(str(tmp_path / "test.jsonl"))
        store.write(Record(content="a"))
        store.clear()
        assert store.count() == 0

    def test_read_nonexistent(self, tmp_path):
        store = FileStore(str(tmp_path / "missing.jsonl"))
        records = list(store.read_all())
        assert records == []

    def test_metadata_persisted(self, tmp_path):
        store = FileStore(str(tmp_path / "test.jsonl"))
        store.write(Record(content="data", metadata={"source": "test"}))
        records = list(store.read_all())
        assert records[0].metadata.get("source") == "test"


# ── CallbackStore ─────────────────────────────────────────────────────────────

class TestCallbackStore:
    def test_write_calls_callback(self):
        received = []
        store = CallbackStore(lambda r: received.append(r))
        store.write(Record(content="hello"))
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_count(self):
        store = CallbackStore(lambda r: None)
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        assert store.count() == 2

    def test_read_all_empty(self):
        store = CallbackStore(lambda r: None)
        assert list(store.read_all()) == []


# ── ChainedStore ──────────────────────────────────────────────────────────────

class TestChainedStore:
    def test_write_to_all(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        chain = ChainedStore([s1, s2])
        chain.write(Record(content="hello"))
        assert s1.count() == 1
        assert s2.count() == 1

    def test_read_all(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        s1.write(Record(content="a"))
        s2.write(Record(content="b"))
        chain = ChainedStore([s1, s2])
        records = list(chain.read_all())
        assert len(records) == 2

    def test_count(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        s1.write(Record(content="a"))
        s2.write(Record(content="b"))
        chain = ChainedStore([s1, s2])
        assert chain.count() == 2


# ── StatsStore ────────────────────────────────────────────────────────────────

class TestStatsStore:
    def test_write_stats(self):
        inner = MemoryStore()
        store = StatsStore(inner)
        store.write(Record(content="hello"))
        store.write(Record(content="world!"))
        stats = store.stats()
        assert stats["total_written"] == 2
        assert stats["total_bytes"] == 11
        assert stats["avg_bytes"] == 5.5

    def test_by_source(self):
        inner = MemoryStore()
        store = StatsStore(inner)
        store.write(Record(content="a", metadata={"source": "api"}))
        store.write(Record(content="b", metadata={"source": "api"}))
        store.write(Record(content="c", metadata={"source": "web"}))
        stats = store.stats()
        assert stats["by_source"] == {"api": 2, "web": 1}

    def test_delegates_read(self):
        inner = MemoryStore()
        inner.write(Record(content="hello"))
        store = StatsStore(inner)
        records = list(store.read_all())
        assert len(records) == 1

    def test_delegates_count(self):
        inner = MemoryStore()
        inner.write(Record(content="hello"))
        store = StatsStore(inner)
        assert store.count() == 1
