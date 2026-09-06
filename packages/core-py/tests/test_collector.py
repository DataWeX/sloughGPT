"""Tests for collections.collector — Collector, ParallelCollector, BatchCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from domains.collections.collector import Collector, ParallelCollector, BatchCollector
from domains.collections.sources import Record
from domains.collections.filters import LengthFilter, DedupFilter
from domains.collections.stores import Store


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_records(n: int, prefix: str = "rec") -> list[Record]:
    return [Record(content=f"{prefix}-{i}", metadata={"i": i}) for i in range(n)]


class ListStore:
    """In-memory store for testing."""

    def __init__(self):
        self._items: list[Record] = []

    def write(self, record: Record):
        self._items.append(record)

    def count(self) -> int:
        return len(self._items)

    def read(self):
        return list(self._items)


class NameSource:
    """Source with a name for ParallelCollector stats."""

    def __init__(self, name: str, records: list[Record]):
        self.name = name
        self._records = records

    def read(self):
        return iter(self._records)


class FailingStore:
    def write(self, record: Record):
        raise RuntimeError("disk full")

    def count(self) -> int:
        return 0

    def read_all(self):
        return iter([])


# ── Collector ─────────────────────────────────────────────────────────────


class TestCollector:

    def test_collect_writes_to_store(self):
        src = NameSource("s", _make_records(3))
        store = ListStore()
        c = Collector(src, store)
        added = c.collect()
        assert added == 3
        assert store.count() == 3

    def test_collect_filters_out(self):
        src = NameSource("s", _make_records(10))
        store = ListStore()
        c = Collector(src, store, filters=[LengthFilter(min_length=100)])
        added = c.collect()
        assert added == 0
        assert c.stats["filtered"] == 10

    def test_collect_dedup(self):
        records = _make_records(5) + _make_records(5)
        src = NameSource("s", records)
        store = ListStore()
        c = Collector(src, store, filters=[DedupFilter()])
        added = c.collect()
        assert added == 5

    def test_collect_error_incremented(self):
        src = NameSource("s", _make_records(3))
        store = ListStore()
        store.write = MagicMock(side_effect=RuntimeError("boom"))
        c = Collector(src, store)
        added = c.collect()
        assert added == 0
        assert c.stats["errors"] == 3

    def test_read_yields_filtered(self):
        records = _make_records(5)
        src = NameSource("s", records)
        c = Collector(src, ListStore(), filters=[LengthFilter(min_length=100)])
        result = list(c.read())
        assert result == []

    def test_read_yields_matching(self):
        records = _make_records(3)
        src = NameSource("s", records)
        c = Collector(src, ListStore())
        result = list(c.read())
        assert len(result) == 3

    def test_collect_continuous_max_rounds(self):
        src = NameSource("s", [])
        store = ListStore()
        c = Collector(src, store)
        call_count = 0
        original_collect = c.collect

        def counting_collect():
            nonlocal call_count
            call_count += 1
            return original_collect()

        c.collect = counting_collect
        c.collect_continuous(interval=0, max_rounds=3)
        assert call_count == 3


# ── ParallelCollector ─────────────────────────────────────────────────────


class TestParallelCollector:

    def test_collects_from_multiple(self):
        c1 = Collector(NameSource("s1", _make_records(3)), ListStore())
        c2 = Collector(NameSource("s2", _make_records(5)), ListStore())
        pc = ParallelCollector([c1, c2])
        total = pc.collect()
        assert total == 8
        assert pc.stats["total_collected"] == 8

    def test_collect_threaded(self):
        c1 = Collector(NameSource("s1", _make_records(3)), ListStore())
        c2 = Collector(NameSource("s2", _make_records(5)), ListStore())
        pc = ParallelCollector([c1, c2])
        total = pc.collect_threaded()
        assert total == 8

    def test_store_returns_first_collector_store(self):
        store = ListStore()
        c1 = Collector(NameSource("s1", []), store)
        pc = ParallelCollector([c1])
        assert pc.store is store

    def test_store_none_when_empty(self):
        pc = ParallelCollector([])
        assert pc.store is None

    def test_collect_continuous_max_rounds(self):
        c1 = Collector(NameSource("s1", []), ListStore())
        pc = ParallelCollector([c1])
        call_count = 0
        original_collect = pc.collect_threaded

        def counting():
            nonlocal call_count
            call_count += 1
            return original_collect()

        pc.collect_threaded = counting
        pc.collect_continuous(interval=0, max_rounds=2)
        assert call_count == 2


# ── BatchCollector ────────────────────────────────────────────────────────


class TestBatchCollector:

    def test_collects_in_batches(self):
        src = NameSource("s", _make_records(25))
        store = ListStore()
        bc = BatchCollector(src, store, batch_size=10)
        added = bc.collect()
        assert added == 25
        assert store.count() == 25
        assert bc.stats["batches"] == 3

    def test_collects_exact_batch(self):
        src = NameSource("s", _make_records(20))
        store = ListStore()
        bc = BatchCollector(src, store, batch_size=10)
        added = bc.collect()
        assert added == 20
        assert bc.stats["batches"] == 2

    def test_filters_in_batches(self):
        records = _make_records(10)
        src = NameSource("s", records)
        store = ListStore()
        bc = BatchCollector(src, store, filters=[LengthFilter(min_length=100)], batch_size=5)
        added = bc.collect()
        assert added == 0
        assert bc.stats["filtered"] == 10

    def test_retry_on_write_failure(self):
        src = NameSource("s", _make_records(3))
        store = ListStore()
        fail_count = 0

        def flaky_write(record):
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                raise RuntimeError("transient")
            store._items.append(record)

        store.write = flaky_write
        bc = BatchCollector(src, store, batch_size=100, max_retries=3, retry_delay=0)
        added = bc.collect()
        assert added == 3
        assert store.count() == 3

    def test_stats_tracked(self):
        src = NameSource("s", _make_records(5))
        store = ListStore()
        bc = BatchCollector(src, store, batch_size=100)
        bc.collect()
        assert bc.stats["collected"] == 5
        assert bc.stats["batches"] == 1
