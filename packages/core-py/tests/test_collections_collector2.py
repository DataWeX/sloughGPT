"""Tests for domains.collections.collector — pure logic, no network."""
from __future__ import annotations

import pytest

from domains.collections.collector import BatchCollector, Collector, ParallelCollector
from domains.collections.filters import (
    DedupFilter,
    FilterChain,
    KeywordFilter,
    LengthFilter,
)
from domains.collections.sources import GeneratorSource, Record
from domains.collections.stores import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(records: list[Record]):
    return GeneratorSource(lambda: iter(records))


def _store():
    return MemoryStore()


def _r(content: str, **meta) -> Record:
    return Record(content=content, metadata=meta)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TestCollector:
    def test_collect_returns_count(self):
        c = Collector(_source([_r("a"), _r("b")]), _store())
        assert c.collect() == 2

    def test_collect_writes_to_store(self):
        store = _store()
        c = Collector(_source([_r("x"), _r("y")]), store)
        c.collect()
        assert store.count() == 2

    def test_collect_stats(self):
        c = Collector(_source([_r("a"), _r("b")]), _store())
        c.collect()
        assert c.stats["collected"] == 2
        assert c.stats["filtered"] == 0
        assert c.stats["errors"] == 0

    def test_collect_with_filter_rejects(self):
        filt = LengthFilter(min_length=10)
        c = Collector(_source([_r("short"), _r("a" * 15)]), _store(), filters=[filt])
        c.collect()
        assert c.stats["collected"] == 1
        assert c.stats["filtered"] == 1

    def test_collect_with_no_filters(self):
        c = Collector(_source([_r("a")]), _store(), filters=None)
        c.collect()
        assert c.stats["collected"] == 1

    def test_collect_empty_source(self):
        c = Collector(_source([]), _store())
        assert c.collect() == 0

    def test_collect_returns_delta(self):
        store = _store()
        c = Collector(_source([_r("a"), _r("b")]), store)
        assert c.collect() == 2
        assert c.collect() == 2  # same source, same count again

    def test_read_yields_filtered_records(self):
        filt = KeywordFilter(keywords=["hello"])
        c = Collector(_source([_r("hello world"), _r("goodbye")]), _store(), filters=[filt])
        results = list(c.read())
        assert len(results) == 1
        assert results[0].content == "hello world"

    def test_read_no_filter(self):
        c = Collector(_source([_r("a"), _r("b")]), _store())
        results = list(c.read())
        assert len(results) == 2

    def test_collect_continuous_max_rounds(self):
        call_count = 0
        orig_collect = Collector.collect

        def counting_collect(self_inner):
            nonlocal call_count
            call_count += 1
            return orig_collect(self_inner)

        c = Collector(_source([_r("a")]), _store())
        c.collect = lambda: counting_collect(c)
        import unittest.mock
        with unittest.mock.patch("domains.collections.collector.time.sleep"):
            c.collect_continuous(interval=0, max_rounds=3)
        assert call_count == 3

    def test_stats_initialized(self):
        c = Collector(_source([]), _store())
        assert c.stats == {"collected": 0, "filtered": 0, "errors": 0}

    def test_multiple_collect_calls_accumulate_stats(self):
        c = Collector(_source([_r("a")]), _store())
        c.collect()
        c.collect()
        assert c.stats["collected"] == 2

    def test_store_count_growth(self):
        store = _store()
        c = Collector(_source([_r("a"), _r("b"), _r("c")]), store)
        c.collect()
        assert store.count() == 3


# ---------------------------------------------------------------------------
# Collector — error handling
# ---------------------------------------------------------------------------

class TestCollectorErrorHandling:
    def test_error_during_write_increments_error_stat(self):
        class BrokenStore:
            def write(self, record):
                raise RuntimeError("disk full")
            def read_all(self):
                return iter([])
            def count(self):
                return 0

        c = Collector(_source([_r("a")]), BrokenStore())
        c.collect()
        assert c.stats["errors"] == 1
        assert c.stats["collected"] == 0

    def test_error_does_not_stop_iteration(self):
        call_count = 0

        class FailOnceStore:
            def __init__(self):
                self.n = 0
            def write(self, record):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("boom")
            def read_all(self):
                return iter([])
            def count(self):
                return self.n - 1 if self.n > 0 else 0

        store = FailOnceStore()
        c = Collector(_source([_r("a"), _r("b")]), store)
        c.collect()
        assert c.stats["errors"] == 1
        assert c.stats["collected"] == 1


# ---------------------------------------------------------------------------
# ParallelCollector
# ---------------------------------------------------------------------------

class TestParallelCollector:
    def test_collect_sums(self):
        c1 = Collector(_source([_r("a"), _r("b")]), _store())
        c2 = Collector(_source([_r("c")]), _store())
        pc = ParallelCollector([c1, c2])
        assert pc.collect() == 3

    def test_collect_populates_stats(self):
        c1 = Collector(_source([_r("a")]), _store())
        pc = ParallelCollector([c1])
        pc.collect()
        assert pc.stats["total_collected"] == 1
        assert "sources" in pc.stats

    def test_collect_threaded(self):
        c1 = Collector(_source([_r("a"), _r("b")]), _store())
        c2 = Collector(_source([_r("c")]), _store())
        pc = ParallelCollector([c1, c2])
        assert pc.collect_threaded() == 3
        assert pc.stats["total_collected"] == 3

    def test_collect_threaded_stats_populated(self):
        c1 = Collector(_source([_r("x")]), _store())
        pc = ParallelCollector([c1])
        pc.collect_threaded()
        assert pc.stats["sources"][c1.source.name]["collected"] == 1

    def test_store_property(self):
        store = _store()
        c1 = Collector(_source([]), store)
        pc = ParallelCollector([c1])
        assert pc.store is store

    def test_store_property_empty(self):
        pc = ParallelCollector([])
        assert pc.store is None

    def test_empty_collectors(self):
        pc = ParallelCollector([])
        assert pc.collect() == 0

    def test_collect_continuous_max_rounds(self):
        call_count = 0
        orig_collect = ParallelCollector.collect_threaded

        def counting_collect(self_inner):
            nonlocal call_count
            call_count += 1
            return orig_collect(self_inner)

        c1 = Collector(_source([_r("a")]), _store())
        pc = ParallelCollector([c1])
        pc.collect_threaded = lambda: counting_collect(pc)
        import unittest.mock
        with unittest.mock.patch("domains.collections.collector.time.sleep"):
            pc.collect_continuous(interval=0, max_rounds=2)
        assert call_count == 2

    def test_multiple_collect_calls(self):
        store = _store()
        c1 = Collector(GeneratorSource(lambda: [_r("a")]), store)
        pc = ParallelCollector([c1])
        pc.collect()
        # Second collect overwrites total_collected with current call's total
        pc.collect()
        assert pc.stats["total_collected"] == 1
        # Store accumulates records across calls
        assert store.count() == 2

    def test_source_names_in_stats(self):
        c1 = Collector(_source([_r("a")]), _store())
        c1.source.name = "src1"
        c2 = Collector(_source([_r("b")]), _store())
        c2.source.name = "src2"
        pc = ParallelCollector([c1, c2])
        pc.collect()
        assert "src1" in pc.stats["sources"]
        assert "src2" in pc.stats["sources"]


# ---------------------------------------------------------------------------
# BatchCollector
# ---------------------------------------------------------------------------

class TestBatchCollector:
    def test_collect_returns_count(self):
        bc = BatchCollector(_source([_r("a"), _r("b"), _r("c")]), _store(), batch_size=2)
        assert bc.collect() == 3

    def test_batch_size_triggers_flush(self):
        store = _store()
        bc = BatchCollector(
            _source([_r(str(i)) for i in range(5)]),
            store,
            batch_size=2,
        )
        bc.collect()
        assert store.count() == 5

    def test_batch_stats(self):
        bc = BatchCollector(_source([_r("a"), _r("b")]), _store(), batch_size=10)
        bc.collect()
        assert bc.stats["collected"] == 2
        assert bc.stats["batches"] == 1
        assert bc.stats["retries"] == 0
        assert bc.stats["errors"] == 0

    def test_batch_with_filter(self):
        filt = LengthFilter(min_length=10)
        bc = BatchCollector(
            _source([_r("short"), _r("a" * 15), _r("b" * 20)]),
            _store(),
            filters=[filt],
            batch_size=10,
        )
        bc.collect()
        assert bc.stats["collected"] == 2
        assert bc.stats["filtered"] == 1

    def test_batch_exact_size(self):
        store = _store()
        bc = BatchCollector(
            _source([_r(str(i)) for i in range(4)]),
            store,
            batch_size=2,
        )
        bc.collect()
        assert bc.stats["batches"] == 2

    def test_batch_remainder(self):
        store = _store()
        bc = BatchCollector(
            _source([_r(str(i)) for i in range(5)]),
            store,
            batch_size=2,
        )
        bc.collect()
        assert bc.stats["batches"] == 3  # 2+2+1

    def test_empty_source(self):
        bc = BatchCollector(_source([]), _store(), batch_size=10)
        assert bc.collect() == 0
        assert bc.stats["batches"] == 0

    def test_single_record(self):
        store = _store()
        bc = BatchCollector(_source([_r("only")]), store, batch_size=100)
        bc.collect()
        assert store.count() == 1
        assert bc.stats["batches"] == 1

    def test_default_params(self):
        bc = BatchCollector(_source([]), _store())
        assert bc.batch_size == 100
        assert bc.max_retries == 3
        assert bc.retry_delay == 1.0

    def test_custom_params(self):
        bc = BatchCollector(_source([]), _store(), batch_size=50, max_retries=5, retry_delay=2.0)
        assert bc.batch_size == 50
        assert bc.max_retries == 5
        assert bc.retry_delay == 2.0


# ---------------------------------------------------------------------------
# BatchCollector — write retry logic
# ---------------------------------------------------------------------------

class TestBatchCollectorRetry:
    def test_write_batch_retries_on_failure(self):
        call_count = 0

        class FailFirstStore:
            def __init__(self):
                self.written = []
                self.n = 0
            def write(self, record):
                self.n += 1
                if self.n <= 1:
                    raise RuntimeError("transient failure")
                self.written.append(record)
            def read_all(self):
                return iter([])
            def count(self):
                return len(self.written)

        store = FailFirstStore()
        bc = BatchCollector(_source([_r("a")]), store, batch_size=10, max_retries=3, retry_delay=0)
        bc.collect()
        assert bc.stats["retries"] == 1
        assert bc.stats["collected"] == 1
        assert len(store.written) == 1

    def test_write_batch_exhausts_retries(self):
        class AlwaysFailStore:
            def write(self, record):
                raise RuntimeError("permanent failure")
            def read_all(self):
                return iter([])
            def count(self):
                return 0

        store = AlwaysFailStore()
        bc = BatchCollector(
            _source([_r("a"), _r("b")]),
            store,
            batch_size=10,
            max_retries=2,
            retry_delay=0,
        )
        bc.collect()
        assert bc.stats["errors"] == 2
        assert bc.stats["retries"] == 1  # 1 retry before final failure


# ---------------------------------------------------------------------------
# Integration: Collector + FilterChain
# ---------------------------------------------------------------------------

class TestCollectorIntegration:
    def test_dedup_filter(self):
        c = Collector(
            _source([_r("dup"), _r("dup"), _r("unique")]),
            _store(),
            filters=[DedupFilter()],
        )
        c.collect()
        assert c.stats["collected"] == 2

    def test_keyword_include(self):
        c = Collector(
            _source([_r("hello world"), _r("goodbye"), _r("hello again")]),
            _store(),
            filters=[KeywordFilter(keywords=["hello"], mode="include")],
        )
        c.collect()
        assert c.stats["collected"] == 2

    def test_keyword_exclude(self):
        c = Collector(
            _source([_r("hello world"), _r("goodbye"), _r("hello again")]),
            _store(),
            filters=[KeywordFilter(keywords=["hello"], mode="exclude")],
        )
        c.collect()
        assert c.stats["collected"] == 1

    def test_length_filter(self):
        c = Collector(
            _source([_r("short"), _r("a" * 20)]),
            _store(),
            filters=[LengthFilter(min_length=10)],
        )
        c.collect()
        assert c.stats["collected"] == 1

    def test_chained_filters(self):
        chain = FilterChain([LengthFilter(min_length=5), KeywordFilter(keywords=["test"])])
        c = Collector(
            _source([_r("test"), _r("nope"), _r("test long enough")]),
            _store(),
            filters=[LengthFilter(min_length=5), KeywordFilter(keywords=["test"])],
        )
        c.collect()
        assert c.stats["collected"] == 1
