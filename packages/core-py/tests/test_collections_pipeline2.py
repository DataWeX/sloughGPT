from __future__ import annotations

import pytest
from collections.abc import Iterator

from domains.collections.pipeline import CollectionPipeline
from domains.collections.sources import Record, Source
from domains.collections.stores import MemoryStore
from domains.collections.filters import LengthFilter, KeywordFilter


class StubSource:
    def __init__(self, records: list[Record], name: str = "stub_src"):
        self.name = name
        self._records = records

    def read(self) -> Iterator[Record]:
        yield from self._records


class FailingSource:
    """Always raises on read."""
    name = "failing_src"

    def read(self) -> Iterator[Record]:
        raise RuntimeError("source exploded")
        yield  # make it a generator


class RecordCounter:
    """Counts how many records were yielded across calls."""
    def __init__(self, records: list[Record], name: str = "counter_src"):
        self.name = name
        self._records = records
        self.call_count = 0

    def read(self) -> Iterator[Record]:
        self.call_count += 1
        yield from self._records


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_default_name(self):
        src = StubSource([], name="Alpha")
        store = MemoryStore(name="Beta")
        p = CollectionPipeline(src, store)
        assert p.name == "Alpha->Beta"

    def test_custom_name(self):
        src = StubSource([], name="X")
        store = MemoryStore(name="Y")
        p = CollectionPipeline(src, store, name="my-pipeline")
        assert p.name == "my-pipeline"

    def test_filters_default_empty(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        assert p.filters == []

    def test_filters_passed_through(self):
        src = StubSource([])
        store = MemoryStore()
        flt = [LengthFilter(min_length=5)]
        p = CollectionPipeline(src, store, filters=flt)
        assert p.filters is flt

    def test_stores_source_and_store_references(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        assert p.source is src
        assert p.store is store

    def test_internal_collector_created(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        assert p._collector.source is src
        assert p._collector.store is store


# ---------------------------------------------------------------------------
# collect()
# ---------------------------------------------------------------------------

class TestPipelineCollect:
    def test_collect_empty_source(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        assert p.collect() == 0
        assert store.count() == 0

    def test_collect_single_record(self):
        rec = Record(content="hello world")
        src = StubSource([rec])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        added = p.collect()
        assert added == 1
        assert store.count() == 1

    def test_collect_multiple_records(self):
        recs = [Record(content=f"item {i}") for i in range(5)]
        src = StubSource(recs)
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        added = p.collect()
        assert added == 5
        assert store.count() == 5

    def test_collect_accumulates_across_calls(self):
        recs = [Record(content="a"), Record(content="b")]
        src = StubSource(recs)
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect()
        p.collect()
        assert store.count() == 4

    def test_collect_returns_store_delta(self):
        src = StubSource([Record(content="x"), Record(content="y")])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        delta = p.collect()
        assert delta == 2

    def test_collect_with_filter_rejects(self):
        src = StubSource([Record(content="short"), Record(content="a" * 50)])
        store = MemoryStore()
        flt = LengthFilter(min_length=10)
        p = CollectionPipeline(src, store, filters=[flt])
        delta = p.collect()
        assert delta == 1
        assert store.count() == 1

    def test_collect_with_multiple_filters(self):
        recs = [
            Record(content="hello world"),           # passes length + keyword
            Record(content="goodbye world"),          # passes length, fails keyword
            Record(content="hi"),                     # fails length
            Record(content="hello there you"),        # passes both
        ]
        src = StubSource(recs)
        store = MemoryStore()
        p = CollectionPipeline(
            src, store,
            filters=[LengthFilter(min_length=5), KeywordFilter(keywords=["hello"], mode="include")]
        )
        delta = p.collect()
        assert delta == 2

    def test_collect_all_filtered_returns_zero(self):
        src = StubSource([Record(content="x")])
        store = MemoryStore()
        p = CollectionPipeline(src, store, filters=[LengthFilter(min_length=100)])
        assert p.collect() == 0
        assert store.count() == 0

    def test_collect_preserves_content(self):
        rec = Record(content="keep me")
        src = StubSource([rec])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect()
        stored = list(store.read_all())
        assert len(stored) == 1
        assert stored[0].content == "keep me"


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------

class TestPipelineRead:
    def test_read_empty(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        results = list(p.read())
        assert results == []

    def test_read_returns_filtered_records(self):
        recs = [Record(content="short"), Record(content="a" * 20)]
        src = StubSource(recs)
        store = MemoryStore()
        p = CollectionPipeline(src, store, filters=[LengthFilter(min_length=10)])
        results = list(p.read())
        assert len(results) == 1
        assert results[0].content == "a" * 20

    def test_read_does_not_write_to_store(self):
        src = StubSource([Record(content="only read, no write")])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        list(p.read())
        assert store.count() == 0

    def test_read_yields_all_when_no_filters(self):
        recs = [Record(content=f"r{i}") for i in range(3)]
        src = StubSource(recs)
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        results = list(p.read())
        assert len(results) == 3


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_keys(self):
        src = StubSource([Record(content="data")], name="src")
        store = MemoryStore(name="mem")
        p = CollectionPipeline(src, store, name="pipe1")
        s = p.stats
        assert "source" in s
        assert "store" in s
        assert "pipeline" in s
        assert "collector" in s
        assert "filter_chain" in s
        assert "store_count" in s

    def test_stats_values_before_collect(self):
        src = StubSource([], name="my_src")
        store = MemoryStore(name="my_store")
        p = CollectionPipeline(src, store, name="my_pipe")
        s = p.stats
        assert s["source"] == "my_src"
        assert s["store"] == "my_store"
        assert s["pipeline"] == "my_pipe"
        assert s["store_count"] == 0

    def test_stats_after_collect(self):
        src = StubSource([Record(content="a"), Record(content="b")])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect()
        s = p.stats
        assert s["collector"]["collected"] == 2
        assert s["store_count"] == 2

    def test_stats_filter_chain_rejected(self):
        src = StubSource([Record(content="x")])
        store = MemoryStore()
        p = CollectionPipeline(src, store, filters=[LengthFilter(min_length=100)])
        p.collect()
        fc = p.stats["filter_chain"]
        assert fc["rejected"] >= 1
        assert fc["accepted"] == 0

    def test_stats_filter_chain_accepted(self):
        src = StubSource([Record(content="a" * 20)])
        store = MemoryStore()
        p = CollectionPipeline(src, store, filters=[LengthFilter(min_length=5)])
        p.collect()
        fc = p.stats["filter_chain"]
        assert fc["accepted"] == 1
        assert fc["rejected"] == 0

    def test_stats_returns_new_dict_each_call(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        s1 = p.stats
        s2 = p.stats
        assert s1 is not s2


# ---------------------------------------------------------------------------
# collect_continuous (mock-free: just verify it calls collect the right number)
# ---------------------------------------------------------------------------

class TestPipelineCollectContinuous:
    def test_collect_continuous_max_rounds(self):
        src = StubSource([Record(content="r")])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect_continuous(interval=0.0, max_rounds=3)
        # Each round collects 1 record; 3 rounds = 3
        assert store.count() == 3

    def test_collect_continuous_one_round(self):
        src = StubSource([Record(content="once")])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect_continuous(interval=0.0, max_rounds=1)
        assert store.count() == 1

    def test_collect_continuous_with_filters(self):
        src = StubSource([Record(content="x"), Record(content="a" * 50)])
        store = MemoryStore()
        p = CollectionPipeline(src, store, filters=[LengthFilter(min_length=10)])
        p.collect_continuous(interval=0.0, max_rounds=2)
        # Each round: 1 record passes (the long one), 2 rounds = 2
        assert store.count() == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    def test_collect_source_that_yields_nothing(self):
        src = StubSource([])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        assert p.collect() == 0
        assert p.stats["collector"]["collected"] == 0

    def test_pipeline_with_many_filters_all_reject(self):
        src = StubSource([Record(content="hi")])
        store = MemoryStore()
        filters = [
            LengthFilter(min_length=100),
            KeywordFilter(keywords=["nonexistent"]),
        ]
        p = CollectionPipeline(src, store, filters=filters)
        assert p.collect() == 0
        assert p.stats["filter_chain"]["rejected"] == 1

    def test_pipeline_preserves_metadata(self):
        rec = Record(content="meta", metadata={"key": "val"})
        src = StubSource([rec])
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect()
        stored = list(store.read_all())
        assert stored[0].metadata.get("key") == "val"

    def test_source_call_count_preserved(self):
        src = RecordCounter([Record(content="c")], name="ctr")
        store = MemoryStore()
        p = CollectionPipeline(src, store)
        p.collect()
        assert src.call_count == 1
        p.collect()
        assert src.call_count == 2
