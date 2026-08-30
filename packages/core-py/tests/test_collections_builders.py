"""Tests for domains.collections.builders — pure logic, no network."""
from __future__ import annotations

import os
import tempfile

import pytest

from domains.collections.builders import (
    CollectorBuilder,
    DataSink,
    DataSource,
    DataTransformer,
)
from domains.collections.filters import (
    KeywordFilter,
    LengthFilter,
    RegexFilter,
)
from domains.collections.sources import (
    FileSource,
    GeneratorSource,
    Record,
    Source,
)
from domains.collections.stores import (
    CallbackStore,
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


def _source(records: list[Record]):
    return GeneratorSource(lambda: iter(records))


# ---------------------------------------------------------------------------
# CollectorBuilder
# ---------------------------------------------------------------------------

class TestCollectorBuilder:
    def test_build_requires_source(self):
        with pytest.raises(ValueError, match="Source is required"):
            CollectorBuilder().build()

    def test_build_with_source_and_memory_store(self):
        b = CollectorBuilder()
        b.source(_source([_r("a")]))
        c = b.build()
        assert c.collect() == 1

    def test_build_with_file_store(self):
        with tempfile.TemporaryDirectory() as td:
            b = CollectorBuilder()
            b.source(_source([_r("x")]))
            b.file_store(os.path.join(td, "out.jsonl"))
            c = b.build()
            c.collect()
            assert c.store.count() == 1

    def test_build_with_memory_store(self):
        b = CollectorBuilder()
        b.source(_source([_r("a")]))
        b.memory_store(max_size=50)
        c = b.build()
        assert c.collect() == 1

    def test_build_with_callback_store(self):
        received = []
        b = CollectorBuilder()
        b.source(_source([_r("cb")]))
        b.callback_store(lambda r: received.append(r))
        c = b.build()
        c.collect()
        assert len(received) == 1

    def test_build_with_stats_store(self):
        b = CollectorBuilder()
        b.source(_source([_r("s")]))
        b.memory_store()
        b.stats_store()
        c = b.build()
        c.collect()
        assert isinstance(c.store, StatsStore)

    def test_build_with_name(self):
        b = CollectorBuilder().name("test").source(_source([]))
        c = b.build()
        assert c is not None

    def test_build_with_filters(self):
        b = CollectorBuilder()
        b.source(_source([_r("short"), _r("a" * 20)]))
        b.length_filter(min_length=10)
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_dedup_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("dup"), _r("dup")]))
        b.dedup_filter()
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_keyword_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("hello"), _r("bye")]))
        b.keyword_filter(["hello"])
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_regex_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("abc123"), _r("xyz")]))
        b.regex_filter(r"\d+")
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_language_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("hello world"), _r("\u00e9\u00e8\u00ea")]))
        b.language_filter(allowed_chars_ratio=0.8)
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_sampler_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r(str(i)) for i in range(100)]))
        b.sampler_filter(rate=0.5)
        c = b.build()
        count = c.collect()
        assert 0 < count < 100

    def test_build_truncate_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("a" * 2000)]))
        b.truncate_filter(max_length=10)
        c = b.build()
        c.collect()
        store = c.store
        records = list(store.read_all())
        assert len(records[0].content) == 10

    def test_build_prefix_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("world")]))
        b.prefix_filter("hello ")
        c = b.build()
        c.collect()
        records = list(c.store.read_all())
        assert records[0].content == "hello world"

    def test_build_metadata_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("a", tag="keep"), _r("b", tag="skip")]))
        b.metadata_filter("tag", ["keep"])
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_generic_filter(self):
        b = CollectorBuilder()
        b.source(_source([_r("a"), _r("b")]))
        b.filter(KeywordFilter(keywords=["a"]))
        c = b.build()
        c.collect()
        assert c.stats["collected"] == 1

    def test_build_batch(self):
        b = CollectorBuilder()
        b.source(_source([_r(str(i)) for i in range(10)]))
        b.batch(batch_size=3, max_retries=2)
        c = b.build()
        from domains.collections.collector import BatchCollector
        assert isinstance(c, BatchCollector)
        assert c.collect() == 10

    def test_build_default_store_is_memory(self):
        b = CollectorBuilder()
        b.source(_source([_r("a")]))
        c = b.build()
        assert isinstance(c.store, MemoryStore)

    def test_build_file_source(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "src.txt")
            with open(p, "w") as f:
                f.write("line1\nline2\n")
            b = CollectorBuilder()
            b.file_source(p)
            c = b.build()
            count = c.collect()
            assert count == 2

    def test_build_build_parallel(self):
        b1 = CollectorBuilder().source(_source([_r("a")]))
        b2 = CollectorBuilder().source(_source([_r("b")]))
        from domains.collections.collector import ParallelCollector
        pc = CollectorBuilder().build_parallel([b1, b2])
        assert isinstance(pc, ParallelCollector)
        assert pc.collect() == 2

    def test_chaining(self):
        result = (
            CollectorBuilder()
            .name("chain")
            .source(_source([_r("a")]))
            .memory_store()
            .length_filter()
            .dedup_filter()
        )
        assert result is not None


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class TestDataSource:
    def test_add_source(self):
        ds = DataSource()
        src = _source([_r("a")])
        ds.add(src)
        assert ds.count() == 1

    def test_add_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("hello\n")
            ds = DataSource()
            ds.add_file(p)
            records = list(ds.read_all())
            assert len(records) == 1

    def test_read_all_multiple_sources(self):
        ds = DataSource()
        ds.add(_source([_r("a")]))
        ds.add(_source([_r("b")]))
        records = list(ds.read_all())
        assert len(records) == 2

    def test_read_specific_source(self):
        ds = DataSource()
        ds.add(_source([_r("a")]))
        ds.add(_source([_r("b")]))
        records = list(ds.read(source_index=1))
        assert len(records) == 1
        assert records[0].content == "b"

    def test_read_invalid_index(self):
        ds = DataSource()
        ds.add(_source([_r("a")]))
        records = list(ds.read(source_index=99))
        assert records == []

    def test_list_sources(self):
        ds = DataSource()
        s1 = _source([_r("a")])
        s1.name = "src1"
        s2 = _source([_r("b")])
        s2.name = "src2"
        ds.add(s1).add(s2)
        assert ds.list_sources() == ["src1", "src2"]

    def test_count(self):
        ds = DataSource()
        assert ds.count() == 0
        ds.add(_source([]))
        assert ds.count() == 1

    def test_fluent_add(self):
        ds = DataSource()
        result = ds.add(_source([_r("a")]))
        assert result is ds

    def test_init_with_sources(self):
        ds = DataSource([_source([_r("a")])])
        assert ds.count() == 1

    def test_add_url(self):
        ds = DataSource()
        ds.add_url("http://example.com")
        assert ds.count() == 1
        assert isinstance(ds._sources[0], Source)


# ---------------------------------------------------------------------------
# DataSink
# ---------------------------------------------------------------------------

class TestDataSink:
    def test_add_memory(self):
        ds = DataSink()
        ds.add_memory()
        assert ds.count() == 0

    def test_add_file(self):
        with tempfile.TemporaryDirectory() as td:
            ds = DataSink()
            ds.add_file(os.path.join(td, "out.jsonl"))
            ds.write(_r("hello"))
            assert ds.count() == 1

    def test_add_callback(self):
        received = []
        ds = DataSink()
        ds.add_callback(lambda r: received.append(r))
        ds.write(_r("x"))
        assert len(received) == 1

    def test_write_all(self):
        ds = DataSink()
        ms = MemoryStore()
        ds.add(ms)
        count = ds.write_all([_r("a"), _r("b"), _r("c")])
        assert count == 3
        assert ms.count() == 3

    def test_list_stores(self):
        ds = DataSink()
        m1 = MemoryStore(name="m1")
        m2 = MemoryStore(name="m2")
        ds.add(m1).add(m2)
        assert ds.list_stores() == ["m1", "m2"]

    def test_count_sums(self):
        ds = DataSink()
        m1 = MemoryStore()
        m2 = MemoryStore()
        ds.add(m1).add(m2)
        m1.write(_r("a"))
        m2.write(_r("b"))
        m2.write(_r("c"))
        assert ds.count() == 3

    def test_flush(self):
        ds = DataSink()
        ds.flush()

    def test_write_to_multiple_stores(self):
        ds = DataSink()
        m1 = MemoryStore()
        m2 = MemoryStore()
        ds.add(m1).add(m2)
        ds.write(_r("shared"))
        assert m1.count() == 1
        assert m2.count() == 1

    def test_init_with_stores(self):
        ms = MemoryStore()
        ds = DataSink([ms])
        assert ds.count() == 0

    def test_fluent_add(self):
        ds = DataSink()
        result = ds.add(MemoryStore())
        assert result is ds


# ---------------------------------------------------------------------------
# DataTransformer
# ---------------------------------------------------------------------------

class TestDataTransformer:
    def test_no_transforms(self):
        dt = DataTransformer()
        r = _r("hello")
        result = dt.transform(r)
        assert result.content == "hello"

    def test_add_transform(self):
        def upper(r):
            r.content = r.content.upper()
            return r
        dt = DataTransformer().add(upper)
        result = dt.transform(_r("hello"))
        assert result.content == "HELLO"

    def test_add_field(self):
        dt = DataTransformer().add_field("tag", "v1")
        result = dt.transform(_r("x"))
        assert result.metadata["tag"] == "v1"

    def test_add_field_fn(self):
        dt = DataTransformer().add_field_fn("length", lambda r: len(r.content))
        result = dt.transform(_r("hello"))
        assert result.metadata["length"] == 5

    def test_add_content_transform(self):
        dt = DataTransformer().add_content_transform(str.upper)
        result = dt.transform(_r("hello"))
        assert result.content == "HELLO"

    def test_transform_stats(self):
        dt = DataTransformer().add(lambda r: r)
        dt.transform(_r("a"))
        dt.transform(_r("b"))
        assert dt.stats["transformed"] == 2
        assert dt.stats["errors"] == 0

    def test_transform_error_counted(self):
        def bad(r):
            raise ValueError("oops")
        dt = DataTransformer().add(bad)
        result = dt.transform(_r("x"))
        assert result.content == "x"
        assert dt.stats["errors"] == 1

    def test_transform_all(self):
        dt = DataTransformer().add_content_transform(str.upper)
        records = [_r("a"), _r("b")]
        results = dt.transform_all(records)
        assert results[0].content == "A"
        assert results[1].content == "B"

    def test_reset_stats(self):
        dt = DataTransformer().add(lambda r: r)
        dt.transform(_r("x"))
        dt.reset_stats()
        assert dt.stats == {"transformed": 0, "errors": 0}

    def test_chained_transforms(self):
        dt = (
            DataTransformer()
            .add_content_transform(str.upper)
            .add_content_transform(lambda s: s + "!")
        )
        result = dt.transform(_r("hello"))
        assert result.content == "HELLO!"

    def test_init_with_transforms(self):
        fn = lambda r: r
        dt = DataTransformer(transforms=[fn])
        result = dt.transform(_r("x"))
        assert result.content == "x"
