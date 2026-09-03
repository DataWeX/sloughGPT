"""Tests for domains.collections.registry — CollectionRegistry."""

from __future__ import annotations

import pytest

from domains.collections.registry import CollectionRegistry, get_registry
from domains.collections.sources import Source, Record
from domains.collections.stores import MemoryStore
from domains.collections.filters import Filter


# ── Helpers ───────────────────────────────────────────────────────────────────

class DummySource(Source):
    def __init__(self, records=None, name="test_source"):
        self.name = name
        self._records = records or [Record(content="test")]
    def collect(self):
        return self._records
    def read(self):
        return iter(self._records)


class DummyFilter(Filter):
    def apply(self, record):
        return record


# ── CollectionRegistry ────────────────────────────────────────────────────────

class TestCollectionRegistry:
    def test_register_source(self):
        reg = CollectionRegistry()
        src = DummySource()
        reg.register_source("test", src)
        assert reg.get_source("test") is src

    def test_register_store(self):
        reg = CollectionRegistry()
        store = MemoryStore()
        reg.register_store("test", store)
        assert reg.get_store("test") is store

    def test_register_filter(self):
        reg = CollectionRegistry()
        f = DummyFilter()
        reg.register_filter("test", f)
        assert reg.get_filter("test") is f

    def test_get_missing(self):
        reg = CollectionRegistry()
        assert reg.get_source("missing") is None
        assert reg.get_store("missing") is None
        assert reg.get_filter("missing") is None

    def test_create_pipeline(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        reg.register_store("store", MemoryStore())
        pipeline = reg.create_pipeline("p1", "src", "store")
        assert pipeline is not None
        assert reg.get_pipeline("p1") is pipeline

    def test_create_pipeline_missing_source(self):
        reg = CollectionRegistry()
        reg.register_store("store", MemoryStore())
        pipeline = reg.create_pipeline("p1", "src", "store")
        assert pipeline is None

    def test_create_pipeline_missing_store(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        pipeline = reg.create_pipeline("p1", "src", "store")
        assert pipeline is None

    def test_create_pipeline_with_filters(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        reg.register_store("store", MemoryStore())
        reg.register_filter("f1", DummyFilter())
        pipeline = reg.create_pipeline("p1", "src", "store", ["f1"])
        assert pipeline is not None

    def test_remove_pipeline(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        reg.register_store("store", MemoryStore())
        reg.create_pipeline("p1", "src", "store")
        assert reg.remove_pipeline("p1") is True
        assert reg.get_pipeline("p1") is None

    def test_remove_pipeline_not_found(self):
        reg = CollectionRegistry()
        assert reg.remove_pipeline("missing") is False

    def test_collect(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource([Record(content="hello")]))
        reg.register_store("store", MemoryStore())
        reg.create_pipeline("p1", "src", "store")
        count = reg.collect("p1")
        assert count == 1
        assert reg.get_store("store").count() == 1

    def test_collect_missing_pipeline(self):
        reg = CollectionRegistry()
        assert reg.collect("missing") == 0

    def test_list_sources(self):
        reg = CollectionRegistry()
        reg.register_source("a", DummySource())
        reg.register_source("b", DummySource())
        assert sorted(reg.list_sources()) == ["a", "b"]

    def test_list_stores(self):
        reg = CollectionRegistry()
        reg.register_store("a", MemoryStore())
        assert reg.list_stores() == ["a"]

    def test_list_filters(self):
        reg = CollectionRegistry()
        reg.register_filter("a", DummyFilter())
        assert reg.list_filters() == ["a"]

    def test_list_pipelines(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        reg.register_store("store", MemoryStore())
        reg.create_pipeline("p1", "src", "store")
        assert reg.list_pipelines() == ["p1"]

    def test_stats(self):
        reg = CollectionRegistry()
        reg.register_source("src", DummySource())
        reg.register_store("store", MemoryStore())
        reg.create_pipeline("p1", "src", "store")
        stats = reg.stats()
        assert "src" in stats["sources"]
        assert "store" in stats["stores"]
        assert "p1" in stats["pipelines"]


# ── get_registry ──────────────────────────────────────────────────────────────

class TestGetRegistry:
    def test_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
