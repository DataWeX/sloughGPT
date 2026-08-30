"""Tests for domains.collections.stores — pure logic, no network."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from domains.collections.sources import Record
from domains.collections.stores import (
    CallbackStore,
    ChainedStore,
    FileStore,
    MemoryStore,
    StatsStore,
    Store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(content: str, **meta) -> Record:
    return Record(content=content, metadata=meta)


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------

class TestStoreProtocol:
    def test_memory_store_is_store(self):
        assert isinstance(MemoryStore(), Store)

    def test_file_store_is_store(self):
        with tempfile.TemporaryDirectory() as td:
            assert isinstance(FileStore(os.path.join(td, "f.jsonl")), Store)

    def test_callback_store_is_store(self):
        assert isinstance(CallbackStore(lambda r: None), Store)

    def test_chained_store_is_store(self):
        assert isinstance(ChainedStore([MemoryStore()]), Store)

    def test_stats_store_is_store(self):
        assert isinstance(StatsStore(MemoryStore()), Store)


# ---------------------------------------------------------------------------
# FileStore
# ---------------------------------------------------------------------------

class TestFileStore:
    def test_init_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "store.jsonl")
            fs = FileStore(path)
            assert fs.path.parent.exists()

    def test_default_name(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "data.jsonl"))
            assert fs.name == "file:data.jsonl"

    def test_custom_name(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "x.jsonl"), name="custom")
            assert fs.name == "custom"

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "s.jsonl"))
            fs.write(_r("hello", source="test"))
            fs.write(_r("world", source="test"))
            records = list(fs.read_all())
            assert len(records) == 2
            assert records[0].content == "hello"
            assert records[1].content == "world"

    def test_count(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "c.jsonl"))
            assert fs.count() == 0
            fs.write(_r("a"))
            fs.write(_r("b"))
            assert fs.count() == 2

    def test_count_nonexistent(self):
        fs = FileStore("/tmp/_nonexistent_store_test_.jsonl")
        assert fs.count() == 0

    def test_read_all_nonexistent(self):
        fs = FileStore("/tmp/_nonexistent_store_test_.jsonl")
        assert list(fs.read_all()) == []

    def test_clear(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "cl.jsonl"))
            fs.write(_r("data"))
            fs.clear()
            assert fs.count() == 0
            assert list(fs.read_all()) == []

    def test_clear_nonexistent(self):
        fs = FileStore("/tmp/_nonexistent_clear_test_.jsonl")
        fs.clear()

    def test_read_all_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"content": "ok"}) + "\n")
                f.write("NOT JSON\n")
                f.write(json.dumps({"content": "ok2"}) + "\n")
            fs = FileStore(p)
            records = list(fs.read_all())
            assert len(records) == 2

    def test_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            fs = FileStore(os.path.join(td, "meta.jsonl"))
            fs.write(_r("content", key="val"))
            records = list(fs.read_all())
            assert records[0].metadata["key"] == "val"

    def test_empty_line_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "empty.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"content": "a"}) + "\n")
                f.write("\n")
                f.write(json.dumps({"content": "b"}) + "\n")
            assert FileStore(p).count() == 2


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_write_and_read(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        ms.write(_r("b"))
        records = list(ms.read_all())
        assert len(records) == 2
        assert records[0].content == "a"
        assert records[1].content == "b"

    def test_count(self):
        ms = MemoryStore()
        assert ms.count() == 0
        ms.write(_r("x"))
        assert ms.count() == 1

    def test_max_size_eviction(self):
        ms = MemoryStore(max_size=3)
        for i in range(5):
            ms.write(_r(str(i)))
        assert ms.count() == 3
        records = list(ms.read_all())
        assert records[0].content == "2"
        assert records[2].content == "4"

    def test_take_single(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        ms.write(_r("b"))
        taken = ms.take(1)
        assert len(taken) == 1
        assert taken[0].content == "a"
        assert ms.count() == 1

    def test_take_all(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        ms.write(_r("b"))
        taken = ms.take(10)
        assert len(taken) == 2
        assert ms.count() == 0

    def test_take_empty(self):
        ms = MemoryStore()
        assert ms.take() == []

    def test_take_zero(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        assert ms.take(0) == []
        assert ms.count() == 1

    def test_peek(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        ms.write(_r("b"))
        ms.write(_r("c"))
        peeked = ms.peek(2)
        assert len(peeked) == 2
        assert ms.count() == 3

    def test_peek_empty(self):
        ms = MemoryStore()
        assert ms.peek() == []

    def test_clear(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        ms.clear()
        assert ms.count() == 0

    def test_default_name(self):
        assert MemoryStore().name == "memory"

    def test_custom_name(self):
        assert MemoryStore(name="my_store").name == "my_store"

    def test_read_all_returns_copy(self):
        ms = MemoryStore()
        ms.write(_r("a"))
        list1 = list(ms.read_all())
        list2 = list(ms.read_all())
        assert len(list1) == len(list2) == 1

    def test_thread_safety(self):
        import threading
        ms = MemoryStore()
        def writer(n):
            for i in range(100):
                ms.write(_r(f"{n}-{i}"))
        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ms.count() == 400


# ---------------------------------------------------------------------------
# CallbackStore
# ---------------------------------------------------------------------------

class TestCallbackStore:
    def test_callback_called(self):
        received = []
        cs = CallbackStore(lambda r: received.append(r))
        cs.write(_r("hello"))
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_count(self):
        cs = CallbackStore(lambda r: None)
        cs.write(_r("a"))
        cs.write(_r("b"))
        assert cs.count() == 2

    def test_read_all_empty(self):
        cs = CallbackStore(lambda r: None)
        assert list(cs.read_all()) == []

    def test_default_name(self):
        assert CallbackStore(lambda r: None).name == "callback"

    def test_custom_name(self):
        assert CallbackStore(lambda r: None, name="cb").name == "cb"


# ---------------------------------------------------------------------------
# ChainedStore
# ---------------------------------------------------------------------------

class TestChainedStore:
    def test_write_to_all_stores(self):
        m1 = MemoryStore()
        m2 = MemoryStore()
        cs = ChainedStore([m1, m2])
        cs.write(_r("x"))
        assert m1.count() == 1
        assert m2.count() == 1

    def test_read_all_deduplicates_by_content_and_store(self):
        m1 = MemoryStore()
        m2 = MemoryStore()
        cs = ChainedStore([m1, m2])
        m1.write(_r("same"))
        m2.write(_r("same"))
        records = list(cs.read_all())
        assert len(records) == 2

    def test_count_sums(self):
        m1 = MemoryStore()
        m2 = MemoryStore()
        cs = ChainedStore([m1, m2])
        m1.write(_r("a"))
        m2.write(_r("b"))
        m2.write(_r("c"))
        assert cs.count() == 3

    def test_default_name(self):
        assert ChainedStore([]).name == "chained"

    def test_custom_name(self):
        assert ChainedStore([], name="ch").name == "ch"

    def test_empty_stores(self):
        cs = ChainedStore([])
        assert cs.count() == 0
        assert list(cs.read_all()) == []


# ---------------------------------------------------------------------------
# StatsStore
# ---------------------------------------------------------------------------

class TestStatsStore:
    def test_delegates_write(self):
        inner = MemoryStore()
        ss = StatsStore(inner)
        ss.write(_r("hello"))
        assert inner.count() == 1

    def test_delegates_read_all(self):
        inner = MemoryStore()
        ss = StatsStore(inner)
        inner.write(_r("x"))
        records = list(ss.read_all())
        assert len(records) == 1

    def test_delegates_count(self):
        inner = MemoryStore()
        ss = StatsStore(inner)
        inner.write(_r("a"))
        inner.write(_r("b"))
        assert ss.count() == 2

    def test_stats_tracking(self):
        inner = MemoryStore()
        ss = StatsStore(inner)
        ss.write(_r("abc", source="src1"))
        ss.write(_r("de", source="src2"))
        ss.write(_r("f", source="src1"))
        stats = ss.stats()
        assert stats["total_written"] == 3
        assert stats["total_bytes"] == 6
        assert stats["avg_bytes"] == 2.0
        assert stats["by_source"] == {"src1": 2, "src2": 1}
        assert stats["inner_count"] == 3

    def test_stats_avg_bytes_zero_writes(self):
        ss = StatsStore(MemoryStore())
        stats = ss.stats()
        assert stats["avg_bytes"] == 0

    def test_default_name(self):
        inner = MemoryStore()
        inner.name = "mem"
        ss = StatsStore(inner)
        assert ss.name == "stats:mem"

    def test_custom_name(self):
        ss = StatsStore(MemoryStore(), name="s")
        assert ss.name == "s"

    def test_unknown_source(self):
        inner = MemoryStore()
        ss = StatsStore(inner)
        ss.write(_r("x"))
        assert ss.stats()["by_source"] == {"unknown": 1}
