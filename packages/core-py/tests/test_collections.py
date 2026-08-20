import json
import os
import tempfile

from domains.collections import (
    Record, FileSource, MemoryStore, CallbackStore, Collector,
    LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
    LanguageFilter, FilterChain, CollectionPipeline, CollectionRegistry,
    ChainedStore, StatsStore, GeneratorSource, WatchSource,
    ParallelCollector, BatchCollector, SamplerFilter, TransformFilter,
    TruncateFilter, PrefixFilter, MetadataFilter,
)


class TestRecord:
    def test_default_timestamp(self):
        r = Record(content="hello")
        assert "timestamp" in r.metadata

    def test_custom_metadata(self):
        r = Record(content="hello", metadata={"source": "test"})
        assert r.metadata["source"] == "test"


class TestFileSource:
    def test_read_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"content": "line1"}\n{"content": "line2"}\n')
        src = FileSource(str(p))
        records = list(src.read())
        assert len(records) == 2
        assert records[0].content == "line1"
        assert records[1].content == "line2"

    def test_read_text(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello\nworld\n")
        src = FileSource(str(p))
        records = list(src.read())
        assert len(records) == 2
        assert records[0].content == "hello"
        assert records[1].content == "world"

    def test_read_json_list(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps(["a", "b", "c"]))
        src = FileSource(str(p))
        records = list(src.read())
        assert len(records) == 3
        assert records[0].content == "a"

    def test_read_nonexistent(self, tmp_path):
        src = FileSource(str(tmp_path / "nope.txt"))
        records = list(src.read())
        assert len(records) == 0

    def test_name(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text("")
        src = FileSource(str(p))
        assert "test.jsonl" in src.name

    def test_custom_name(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text("")
        src = FileSource(str(p), name="my_source")
        assert src.name == "my_source"


class TestMemoryStore:
    def test_write_and_count(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        assert store.count() == 2

    def test_read_all(self):
        store = MemoryStore()
        store.write(Record(content="x"))
        store.write(Record(content="y"))
        records = list(store.read_all())
        assert len(records) == 2
        assert records[0].content == "x"

    def test_max_size(self):
        store = MemoryStore(max_size=3)
        for i in range(5):
            store.write(Record(content=str(i)))
        assert store.count() == 3
        records = list(store.read_all())
        assert records[0].content == "2"

    def test_take(self):
        store = MemoryStore()
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        taken = store.take(1)
        assert len(taken) == 1
        assert taken[0].content == "a"
        assert store.count() == 1

    def test_peek(self):
        store = MemoryStore()
        store.write(Record(content="x"))
        peeked = store.peek(5)
        assert len(peeked) == 1
        assert store.count() == 1

    def test_clear(self):
        store = MemoryStore()
        store.write(Record(content="x"))
        store.clear()
        assert store.count() == 0


class TestCallbackStore:
    def test_callback_receives_record(self):
        received = []
        store = CallbackStore(lambda r: received.append(r))
        store.write(Record(content="test"))
        assert len(received) == 1
        assert received[0].content == "test"

    def test_count(self):
        store = CallbackStore(lambda r: None)
        store.write(Record(content="a"))
        store.write(Record(content="b"))
        assert store.count() == 2


class TestFilters:
    def test_length_filter_accept(self):
        f = LengthFilter(min_length=3, max_length=10)
        assert f.accept(Record(content="hello"))
        assert not f.accept(Record(content="hi"))
        assert not f.accept(Record(content="x" * 11))

    def test_dedup_filter(self):
        f = DedupFilter()
        assert f.accept(Record(content="hello"))
        assert not f.accept(Record(content="hello"))
        assert f.accept(Record(content="world"))

    def test_dedup_reset(self):
        f = DedupFilter()
        f.accept(Record(content="hello"))
        f.reset()
        assert f.accept(Record(content="hello"))

    def test_keyword_filter_include(self):
        f = KeywordFilter(keywords=["python", "code"])
        assert f.accept(Record(content="I love python"))
        assert not f.accept(Record(content="I love java"))

    def test_keyword_filter_exclude(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert f.accept(Record(content="hello world"))
        assert not f.accept(Record(content="spam message"))

    def test_regex_filter(self):
        f = RegexFilter(pattern=r"\d{3}-\d{4}")
        assert f.accept(Record(content="call 555-1234 now"))
        assert not f.accept(Record(content="no numbers here"))

    def test_language_filter(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(Record(content="hello world"))
        assert not f.accept(Record(content="\xff\xfe\xfd"))

    def test_filter_chain(self):
        chain = FilterChain([LengthFilter(min_length=3)])
        assert chain.accept(Record(content="hello"))
        assert not chain.accept(Record(content="hi"))
        assert chain.stats["accepted"] == 1
        assert chain.stats["rejected"] == 1


class TestCollector:
    def test_collect_file_to_memory(self, tmp_path):
        src_file = tmp_path / "input.jsonl"
        src_file.write_text('{"content": "a"}\n{"content": "b"}\n')
        source = FileSource(str(src_file))
        store = MemoryStore()
        collector = Collector(source, store)
        count = collector.collect()
        assert count == 2
        assert store.count() == 2

    def test_collect_with_filter(self, tmp_path):
        src_file = tmp_path / "input.txt"
        src_file.write_text("short\nthis is a longer line that passes the filter\n")
        source = FileSource(str(src_file))
        store = MemoryStore()
        collector = Collector(source, store, [LengthFilter(min_length=20)])
        count = collector.collect()
        assert count == 1

    def test_collect_with_dedup(self, tmp_path):
        src_file = tmp_path / "input.txt"
        src_file.write_text("hello\nhello\nworld\n")
        source = FileSource(str(src_file))
        store = MemoryStore()
        collector = Collector(source, store, [DedupFilter()])
        count = collector.collect()
        assert count == 2


class TestCollectionPipeline:
    def test_pipeline_collect(self, tmp_path):
        src_file = tmp_path / "input.jsonl"
        src_file.write_text('{"content": "alpha"}\n{"content": "beta"}\n')
        pipeline = CollectionPipeline(
            source=FileSource(str(src_file)),
            store=MemoryStore(),
        )
        count = pipeline.collect()
        assert count == 2
        assert pipeline.stats["collector"]["collected"] == 2

    def test_pipeline_read(self, tmp_path):
        src_file = tmp_path / "input.txt"
        src_file.write_text("line1\nline2\n")
        pipeline = CollectionPipeline(
            source=FileSource(str(src_file)),
            store=MemoryStore(),
        )
        records = list(pipeline.read())
        assert len(records) == 2


class TestCollectionRegistry:
    def test_register_and_get(self):
        reg = CollectionRegistry()
        src = MemoryStore()
        reg.register_store("mem", src)
        assert reg.get_store("mem") is src
        assert reg.get_store("nope") is None

    def test_create_pipeline(self, tmp_path):
        reg = CollectionRegistry()
        src_file = tmp_path / "in.txt"
        src_file.write_text("hello\n")
        reg.register_source("f", FileSource(str(src_file)))
        reg.register_store("m", MemoryStore())
        pipeline = reg.create_pipeline("test", "f", "m")
        assert pipeline is not None
        count = pipeline.collect()
        assert count == 1

    def test_list(self):
        reg = CollectionRegistry()
        reg.register_source("s1", None)
        reg.register_store("st1", None)
        assert "s1" in reg.list_sources()
        assert "st1" in reg.list_stores()


class TestChainedStore:
    def test_write_to_all(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        chained = ChainedStore([s1, s2])
        chained.write(Record(content="hello"))
        assert s1.count() == 1
        assert s2.count() == 1

    def test_read_all_merges(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        s1.write(Record(content="a"))
        s2.write(Record(content="b"))
        chained = ChainedStore([s1, s2])
        records = list(chained.read_all())
        assert len(records) == 2

    def test_count_sums(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        s1.write(Record(content="a"))
        s2.write(Record(content="b"))
        s2.write(Record(content="c"))
        chained = ChainedStore([s1, s2])
        assert chained.count() == 3


class TestStatsStore:
    def test_stats_tracking(self):
        inner = MemoryStore()
        stats = StatsStore(inner)
        stats.write(Record(content="hello", metadata={"source": "test"}))
        stats.write(Record(content="world", metadata={"source": "test"}))
        s = stats.stats()
        assert s["total_written"] == 2
        assert s["total_bytes"] == 10
        assert s["by_source"]["test"] == 2

    def test_delegates_to_inner(self):
        inner = MemoryStore()
        stats = StatsStore(inner)
        stats.write(Record(content="x"))
        assert inner.count() == 1
        assert stats.count() == 1


class TestGeneratorSource:
    def test_generator_fn(self):
        def gen():
            yield "hello"
            yield "world"
        src = GeneratorSource(gen)
        records = list(src.read())
        assert len(records) == 2
        assert records[0].content == "hello"

    def test_generator_with_records(self):
        def gen():
            yield Record(content="a", metadata={"x": 1})
        src = GeneratorSource(gen)
        records = list(src.read())
        assert len(records) == 1
        assert records[0].metadata["x"] == 1


class TestWatchSource:
    def test_watches_new_files(self, tmp_path):
        src = WatchSource(str(tmp_path), patterns=["*.txt"])
        records = list(src.read())
        assert len(records) == 0

        (tmp_path / "test.txt").write_text("hello")
        records = list(src.read())
        assert len(records) == 1
        assert records[0].content == "hello"

    def test_watches_modified_files(self, tmp_path):
        (tmp_path / "test.txt").write_text("v1")
        src = WatchSource(str(tmp_path), patterns=["*.txt"])
        list(src.read())

        (tmp_path / "test.txt").write_text("v2")
        records = list(src.read())
        assert len(records) == 1
        assert records[0].content == "v2"

    def test_reset(self, tmp_path):
        (tmp_path / "test.txt").write_text("hello")
        src = WatchSource(str(tmp_path), patterns=["*.txt"])
        list(src.read())
        src.reset()
        records = list(src.read())
        assert len(records) == 1


class TestParallelCollector:
    def test_collects_from_all(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha\nbeta\n")
        f2.write_text("gamma\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        c2 = Collector(FileSource(str(f2)), store)
        pc = ParallelCollector([c1, c2])
        count = pc.collect_threaded()
        assert count == 3
        assert store.count() == 3

    def test_stats_populated(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        pc = ParallelCollector([c1])
        pc.collect_threaded()
        assert "file:a.txt" in pc.stats["sources"]


class TestBatchCollector:
    def test_collects_all(self, tmp_path):
        src_file = tmp_path / "data.txt"
        lines = "\n".join(f"line{i}" for i in range(25))
        src_file.write_text(lines + "\n")
        store = MemoryStore()
        bc = BatchCollector(FileSource(str(src_file)), store, batch_size=10)
        count = bc.collect()
        assert count == 25
        assert store.count() == 25
        assert bc.stats["batches"] == 3


class TestSamplerFilter:
    def test_samples_subset(self):
        f = SamplerFilter(rate=0.5)
        results = [f.accept(Record(content="x")) for _ in range(100)]
        sampled = sum(results)
        assert 20 < sampled < 80


class TestTransformFilter:
    def test_transform(self):
        tf = TransformFilter(transform_fn=lambda r: Record(
            content=r.content.upper(), metadata=r.metadata
        ))
        r = Record(content="hello")
        assert tf.accept(r) is True
        result = tf.transform(r)
        assert result.content == "HELLO"


class TestTruncateFilter:
    def test_truncates(self):
        f = TruncateFilter(max_length=5)
        r = Record(content="hello world")
        f.accept(r)
        assert r.content == "hello"


class TestPrefixFilter:
    def test_prefixes(self):
        f = PrefixFilter(prefix="[PREF] ")
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "[PREF] hello"


class TestMetadataFilter:
    def test_include_mode(self):
        f = MetadataFilter(key="source", values=["rss", "api"])
        assert f.accept(Record(content="x", metadata={"source": "rss"}))
        assert not f.accept(Record(content="x", metadata={"source": "file"}))

    def test_exclude_mode(self):
        f = MetadataFilter(key="source", values=["spam"], mode="exclude")
        assert f.accept(Record(content="x", metadata={"source": "news"}))
        assert not f.accept(Record(content="x", metadata={"source": "spam"}))
