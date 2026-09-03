"""Tests for CollectionRegistry — source/store/filter/pipeline registry."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from domains.collections.registry import CollectionRegistry, get_registry
from domains.collections.sources import Record


@runtime_checkable
class FakeSource(Protocol):
    name: str
    def fetch(self) -> Iterator[Record]: ...


@runtime_checkable
class FakeStore(Protocol):
    name: str
    def write(self, record: Record) -> None: ...
    def read_all(self) -> Iterator[Record]: ...


class _FakeSource:
    def __init__(self, name: str = "test_source"):
        self.name = name
    def fetch(self) -> Iterator[Record]:
        yield Record(content="hello")


class _FakeStore:
    def __init__(self, name: str = "test_store"):
        self.name = name
        self.records = []
    def write(self, record: Record) -> None:
        self.records.append(record)
    def read_all(self) -> Iterator[Record]:
        return iter(self.records)


class _FakeFilter:
    def accept(self, record: Record) -> bool:
        return True


class TestCollectionRegistry:
    def test_register_source(self):
        reg = CollectionRegistry()
        src = _FakeSource("src1")
        reg.register_source("src1", src)
        assert reg.get_source("src1") is src

    def test_register_store(self):
        reg = CollectionRegistry()
        store = _FakeStore("st1")
        reg.register_store("st1", store)
        assert reg.get_store("st1") is store

    def test_register_filter(self):
        reg = CollectionRegistry()
        filt = _FakeFilter()
        reg.register_filter("f1", filt)
        assert reg.get_filter("f1") is filt

    def test_get_nonexistent(self):
        reg = CollectionRegistry()
        assert reg.get_source("nope") is None
        assert reg.get_store("nope") is None
        assert reg.get_filter("nope") is None

    def test_create_pipeline(self):
        reg = CollectionRegistry()
        src = _FakeSource("src")
        store = _FakeStore("st")
        reg.register_source("src", src)
        reg.register_store("st", store)
        pipe = reg.create_pipeline("p1", "src", "st")
        assert pipe is not None
        assert pipe.name == "p1"

    def test_create_pipeline_missing_source(self):
        reg = CollectionRegistry()
        store = _FakeStore("st")
        reg.register_store("st", store)
        pipe = reg.create_pipeline("p1", "missing", "st")
        assert pipe is None

    def test_remove_pipeline(self):
        reg = CollectionRegistry()
        src = _FakeSource()
        store = _FakeStore()
        reg.register_source("s", src)
        reg.register_store("s", store)
        reg.create_pipeline("p", "s", "s")
        assert reg.remove_pipeline("p") is True
        assert reg.get_pipeline("p") is None
        assert reg.remove_pipeline("p") is False

    def test_list_methods(self):
        reg = CollectionRegistry()
        reg.register_source("s1", _FakeSource("s1"))
        reg.register_store("st1", _FakeStore("st1"))
        reg.register_filter("f1", _FakeFilter())
        assert "s1" in reg.list_sources()
        assert "st1" in reg.list_stores()
        assert "f1" in reg.list_filters()

    def test_stats(self):
        reg = CollectionRegistry()
        reg.register_source("s", _FakeSource("s"))
        stats = reg.stats()
        assert "sources" in stats
        assert "s" in stats["sources"]

    def test_singleton(self):
        a = get_registry()
        b = get_registry()
        assert a is b
