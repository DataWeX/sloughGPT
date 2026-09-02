"""Tests for domains.collections.pipeline — CollectionPipeline."""

from __future__ import annotations

import pytest

from domains.collections.pipeline import CollectionPipeline
from domains.collections.sources import Source, Record
from domains.collections.stores import MemoryStore
from domains.collections.filters import Filter


class DummySource(Source):
    def __init__(self, records=None, name="test_source"):
        self.name = name
        self._records = records or [Record(content="test")]
    def collect(self):
        return self._records
    def read(self):
        return iter(self._records)


class DummyFilter(Filter):
    def accept(self, record):
        return True


class TestCollectionPipeline:
    def test_init_default_name(self):
        src = DummySource()
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store)
        assert pipeline.name == "test_source->memory"

    def test_init_custom_name(self):
        src = DummySource()
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store, name="mypipeline")
        assert pipeline.name == "mypipeline"

    def test_collect(self):
        src = DummySource([Record(content="hello")])
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store)
        count = pipeline.collect()
        assert count == 1
        assert store.count() == 1

    def test_collect_with_filters(self):
        src = DummySource([Record(content="hello")])
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store, filters=[DummyFilter()])
        count = pipeline.collect()
        assert count == 1

    def test_read(self):
        src = DummySource([Record(content="hello")])
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store)
        records = list(pipeline.read())
        assert len(records) == 1
        assert records[0].content == "hello"

    def test_stats(self):
        src = DummySource(name="mysource")
        store = MemoryStore()
        pipeline = CollectionPipeline(src, store, name="mypipeline")
        stats = pipeline.stats
        assert stats["source"] == "mysource"
        assert stats["store"] == "memory"
        assert stats["pipeline"] == "mypipeline"
        assert stats["store_count"] == 0
