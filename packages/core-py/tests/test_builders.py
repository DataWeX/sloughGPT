"""Meaningful tests for CollectorBuilder, DataSource, DataSink, DataTransformer — builder pattern, source/sink wiring, transforms."""

import pytest
from pathlib import Path
from domains.collections.builders import CollectorBuilder, DataSource, DataSink, DataTransformer
from domains.collections.sources import Record, GeneratorSource
from domains.collections.stores import MemoryStore, StatsStore


# ── CollectorBuilder ───────────────────────────────────────────────────

class TestCollectorBuilder:
    def test_build_requires_source(self):
        with pytest.raises(ValueError, match="Source is required"):
            CollectorBuilder().build()

    def test_build_default_memory_store(self):
        def gen():
            yield Record("hello")

        collector = CollectorBuilder().generator_source(gen).build()
        assert collector is not None
        count = collector.collect()
        assert count == 1

    def test_build_with_file_store(self, tmp_path):
        def gen():
            yield Record("data")

        path = str(tmp_path / "out.jsonl")
        # FileStore doesn't accept 'append' kwarg — use store() directly
        from domains.collections.stores import FileStore
        store = FileStore(path)
        collector = CollectorBuilder().generator_source(gen).store(store).build()
        collector.collect()
        assert Path(path).exists()

    def test_build_with_memory_store(self):
        def gen():
            yield Record("x")

        collector = CollectorBuilder().generator_source(gen).memory_store().build()
        collector.collect()
        assert collector.store.count() == 1

    def test_build_with_callback_store(self):
        captured = []
        def gen():
            yield Record("cb")

        collector = CollectorBuilder().generator_source(gen).callback_store(lambda r: captured.append(r)).build()
        collector.collect()
        assert len(captured) == 1
        assert captured[0].content == "cb"

    def test_build_with_length_filter(self):
        def gen():
            yield Record("short")
            yield Record("this is a longer record with enough content")

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .length_filter(min_length=10)
            .build()
        )
        count = collector.collect()
        assert count == 1

    def test_build_batch(self):
        def gen():
            for i in range(5):
                yield Record(f"r{i}")

        collector = CollectorBuilder().generator_source(gen).memory_store().batch(batch_size=2).build()
        assert collector.batch_size == 2

    def test_stats_store_wraps(self):
        def gen():
            yield Record("s")

        builder = CollectorBuilder().generator_source(gen).memory_store().stats_store()
        collector = builder.build()
        assert isinstance(collector.store, StatsStore)

    def test_fluent_chaining(self):
        def gen():
            yield Record("chain")

        result = (
            CollectorBuilder()
            .name("test")
            .generator_source(gen)
            .memory_store()
            .dedup_filter()
            .build()
        )
        assert result is not None

    def test_keyword_filter(self):
        def gen():
            yield Record("python is great")
            yield Record("rust is fast")

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .keyword_filter(["python"], mode="include")
            .build()
        )
        count = collector.collect()
        assert count == 1

    def test_regex_filter(self):
        def gen():
            yield Record("hello world")
            yield Record("goodbye moon")

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .regex_filter(r"^hello", mode="include")
            .build()
        )
        count = collector.collect()
        assert count == 1

    def test_transform_filter(self):
        def gen():
            yield Record("original")

        def upper(r):
            r.content = r.content.upper()
            return r

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .transform_filter(upper)
            .build()
        )
        count = collector.collect()
        assert count == 1
        assert collector.store.count() == 1

    def test_truncate_filter(self):
        def gen():
            yield Record("a" * 500)

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .truncate_filter(max_length=10)
            .build()
        )
        collector.collect()
        records = list(collector.store.read_all())
        assert len(records[0].content) == 10

    def test_prefix_filter(self):
        def gen():
            yield Record("hello")

        collector = (
            CollectorBuilder()
            .generator_source(gen)
            .memory_store()
            .prefix_filter(">>> ")
            .build()
        )
        collector.collect()
        records = list(collector.store.read_all())
        assert records[0].content.startswith(">>> ")

    def test_build_parallel(self):
        def gen():
            yield Record("p")

        b1 = CollectorBuilder().generator_source(gen).memory_store()
        b2 = CollectorBuilder().generator_source(gen).memory_store()
        parallel = CollectorBuilder().build_parallel([b1, b2])
        assert parallel is not None


# ── DataSource ─────────────────────────────────────────────────────────

class TestDataSource:
    def test_add_source(self):
        def gen():
            yield Record("s1")
            yield Record("s2")

        ds = DataSource().add(GeneratorSource(gen))
        assert ds.count() == 1

    def test_read_all(self):
        def gen1():
            yield Record("a")

        def gen2():
            yield Record("b")

        ds = DataSource().add(GeneratorSource(gen1)).add(GeneratorSource(gen2))
        records = list(ds.read_all())
        assert len(records) == 2

    def test_read_single_source(self):
        def gen():
            yield Record("only")

        ds = DataSource().add(GeneratorSource(gen))
        records = list(ds.read(0))
        assert len(records) == 1

    def test_read_out_of_bounds(self):
        ds = DataSource()
        records = list(ds.read(99))
        assert records == []

    def test_list_sources(self):
        def gen():
            yield Record("x")

        src = GeneratorSource(gen)
        ds = DataSource().add(src)
        names = ds.list_sources()
        assert len(names) == 1

    def test_count(self):
        ds = DataSource()
        assert ds.count() == 0


# ── DataSink ───────────────────────────────────────────────────────────

class TestDataSink:
    def test_write_single(self):
        store = MemoryStore()
        sink = DataSink().add(store)
        sink.write(Record("hello"))
        assert store.count() == 1

    def test_write_all(self):
        store = MemoryStore()
        sink = DataSink().add(store)
        count = sink.write_all([Record("a"), Record("b"), Record("c")])
        assert count == 3
        assert store.count() == 3

    def test_add_memory(self):
        sink = DataSink().add_memory(max_size=5)
        assert sink.count() == 0

    def test_add_callback(self):
        captured = []
        sink = DataSink().add_callback(lambda r: captured.append(r))
        sink.write(Record("cb"))
        assert len(captured) == 1

    def test_list_stores(self):
        sink = DataSink().add_memory()
        assert len(sink.list_stores()) == 1

    def test_flush_noop(self):
        sink = DataSink()
        sink.flush()  # should not raise


# ── DataTransformer ────────────────────────────────────────────────────

class TestDataTransformer:
    def test_transform_single(self):
        dt = DataTransformer()
        dt.add(lambda r: Record(r.content.upper(), r.metadata))
        result = dt.transform(Record("hello"))
        assert result.content == "HELLO"

    def test_transform_all(self):
        dt = DataTransformer()
        dt.add(lambda r: Record(r.content + "!", r.metadata))
        results = dt.transform_all([Record("a"), Record("b")])
        assert [r.content for r in results] == ["a!", "b!"]

    def test_add_field(self):
        dt = DataTransformer()
        dt.add_field("source", "test")
        result = dt.transform(Record("x"))
        assert result.metadata["source"] == "test"

    def test_add_field_fn(self):
        dt = DataTransformer()
        dt.add_field_fn("length", lambda r: len(r.content))
        result = dt.transform(Record("hello"))
        assert result.metadata["length"] == 5

    def test_add_content_transform(self):
        dt = DataTransformer()
        dt.add_content_transform(str.upper)
        result = dt.transform(Record("hello"))
        assert result.content == "HELLO"

    def test_stats_track(self):
        dt = DataTransformer()
        dt.add(lambda r: Record(r.content, r.metadata))
        dt.transform(Record("a"))
        assert dt.stats["transformed"] == 1
        assert dt.stats["errors"] == 0

    def test_stats_errors(self):
        dt = DataTransformer()
        dt.add(lambda r: 1 / 0)  # type: ignore
        dt.transform(Record("a"))
        assert dt.stats["errors"] == 1

    def test_reset_stats(self):
        dt = DataTransformer()
        dt.add(lambda r: r)
        dt.transform(Record("a"))
        dt.reset_stats()
        assert dt.stats == {"transformed": 0, "errors": 0}

    def test_chained_transforms(self):
        dt = DataTransformer()
        dt.add_content_transform(str.upper)
        dt.add_content_transform(lambda s: s + "!")
        result = dt.transform(Record("hello"))
        assert result.content == "HELLO!"
