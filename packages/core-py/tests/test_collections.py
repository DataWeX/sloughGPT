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
    Schema, DataValidator, DataEnricher, EnrichmentRule, RateLimiter,
    CallableSource, CallableStore, CollectorRunner,
    JobConfig, JobScheduler, CollectorMonitor, CollectorExporter,
    CollectorBuilder, DataSource, DataSink, DataTransformer,
    WorldFeedConfig, RecordToWorldMapper, WorldGridBridge,
    WorldGridSource, WorldStoreAdapter, CollectionWorldPipeline,
    TrainingDataConfig, TrainingDataAdapter, TrainingDatasetBuilder,
    CollectorTrainingBridge,
    collect_file, collect_records,
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
        store1 = MemoryStore()
        store2 = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store1)
        c2 = Collector(FileSource(str(f2)), store2)
        pc = ParallelCollector([c1, c2])
        count = pc.collect_threaded()
        assert count == 3
        assert store1.count() == 2
        assert store2.count() == 1

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


class TestSchema:
    def test_valid_record(self):
        s = Schema(required_fields=["source"], field_types={"source": str})
        r = Record(content="hello", metadata={"source": "test"})
        valid, _ = s.validate(r)
        assert valid

    def test_missing_required_field(self):
        s = Schema(required_fields=["source"])
        r = Record(content="hello")
        valid, error = s.validate(r)
        assert not valid
        assert "source" in error

    def test_wrong_field_type(self):
        s = Schema(field_types={"count": int})
        r = Record(content="hello", metadata={"count": "not_int"})
        valid, error = s.validate(r)
        assert not valid
        assert "wrong type" in error

    def test_content_length_limit(self):
        s = Schema(max_content_length=5)
        r = Record(content="hello world")
        valid, error = s.validate(r)
        assert not valid
        assert "too long" in error


class TestDataValidator:
    def test_validates_all(self):
        schema = Schema(required_fields=["source"])
        validator = DataValidator(schema)
        r1 = Record(content="hello", metadata={"source": "a"})
        r2 = Record(content="world")
        results = validator.validate_all([r1, r2])
        assert len(results) == 1
        assert validator.stats["valid"] == 1
        assert validator.stats["invalid"] == 1


class TestDataEnricher:
    def test_enriches(self):
        enricher = DataEnricher([
            EnrichmentRule(key="tag", value="important"),
            EnrichmentRule(key="length", value_fn=lambda r: len(r.content)),
        ])
        r = Record(content="hello")
        result = enricher.enrich(r)
        assert result.metadata["tag"] == "important"
        assert result.metadata["length"] == 5
        assert enricher.stats["enriched"] == 1

    def test_no_overwrite(self):
        enricher = DataEnricher([EnrichmentRule(key="tag", value="new")])
        r = Record(content="hello", metadata={"tag": "old"})
        enricher.enrich(r)
        assert r.metadata["tag"] == "old"
        assert enricher.stats["skipped"] == 1


class TestRateLimiter:
    def test_allows_within_rate(self):
        limiter = RateLimiter(max_per_second=10, burst_size=5)
        assert limiter.acquire()
        assert limiter.acquire()
        assert limiter.stats["allowed"] == 2

    def test_rate_limiting(self):
        limiter = RateLimiter(max_per_second=1, burst_size=1)
        assert limiter.acquire()
        assert not limiter.acquire()
        assert limiter.stats["delayed"] == 1


class TestCallableSource:
    def test_callable_source(self):
        def gen():
            yield Record(content="hello")
        src = CallableSource(gen)
        records = list(src.read())
        assert len(records) == 1
        assert records[0].content == "hello"


class TestCallableStore:
    def test_callable_store(self):
        stored = []
        def store_fn(r):
            stored.append(r)
        store = CallableStore(store_fn)
        store.write(Record(content="hello"))
        assert len(stored) == 1
        assert store.count() == 1


class TestCollectorRunner:
    def test_runs_collectors(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha\n")
        f2.write_text("beta\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        c2 = Collector(FileSource(str(f2)), store)
        runner = CollectorRunner()
        runner.add("src1", c1)
        runner.add("src2", c2)
        results = runner.run_all()
        assert results["src1"] == 1
        assert results["src2"] == 1
        assert store.count() == 2
        stats = runner.stats()
        assert stats["src1"]["runs"] == 1
        assert stats["src1"]["total_collected"] == 1

    def test_remove(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        runner = CollectorRunner()
        runner.add("src", c1)
        assert runner.remove("src")
        assert not runner.remove("nonexistent")
        assert len(runner.list()) == 0


class TestJobScheduler:
    def test_add_and_list(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        scheduler = JobScheduler()
        config = JobConfig(name="job1", interval=0.1, max_runs=1)
        scheduler.add_job(config, c1)
        assert "job1" in scheduler.list_jobs()
        assert scheduler.get_collector("job1") is c1

    def test_start_and_stop(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        scheduler = JobScheduler()
        config = JobConfig(name="job1", interval=0.1, max_runs=2)
        scheduler.add_job(config, c1)
        assert scheduler.start_job("job1")
        import time
        time.sleep(0.5)
        scheduler.stop_job("job1")
        stats = scheduler.job_stats("job1")
        assert stats["runs"] >= 1

    def test_remove_job(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello\n")
        store = MemoryStore()
        c1 = Collector(FileSource(str(f1)), store)
        scheduler = JobScheduler()
        config = JobConfig(name="job1", interval=1.0)
        scheduler.add_job(config, c1)
        assert scheduler.remove_job("job1")
        assert not scheduler.remove_job("nonexistent")


class TestCollectorMonitor:
    def test_overview(self):
        monitor = CollectorMonitor()
        overview = monitor.get_overview()
        assert "timestamp" in overview
        assert overview["healthy"]

    def test_health_checks(self):
        monitor = CollectorMonitor()
        monitor.add_health_check("test", lambda: True)
        health = monitor.check_health()
        assert health["test"]

    def test_format_report(self):
        monitor = CollectorMonitor()
        report = monitor.format_report()
        assert "Collection Monitor Report" in report


class TestCollectorExporter:
    def test_export_jsonl(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="hello"))
        store.write(Record(content="world"))
        exporter = CollectorExporter(store)
        out_path = str(tmp_path / "out.jsonl")
        count = exporter.to_jsonl(out_path)
        assert count == 2
        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_export_text(self, tmp_path):
        store = MemoryStore()
        store.write(Record(content="hello"))
        store.write(Record(content="world"))
        exporter = CollectorExporter(store)
        out_path = str(tmp_path / "out.txt")
        count = exporter.to_text(out_path)
        assert count == 2

    def test_to_dicts(self):
        store = MemoryStore()
        store.write(Record(content="hello"))
        exporter = CollectorExporter(store)
        dicts = exporter.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["content"] == "hello"

    def test_summary(self):
        store = MemoryStore()
        store.write(Record(content="hello", metadata={"source": "test"}))
        exporter = CollectorExporter(store)
        s = exporter.summary()
        assert s["count"] == 1
        assert s["total_bytes"] == 5
        assert s["sources"]["test"] == 1


class TestConvenienceAPI:
    def test_collect_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\n")
        count = collect_file(str(f))
        assert count == 2

    def test_collect_records(self):
        records = [Record(content="a"), Record(content="b")]
        count = collect_records(records)
        assert count == 2


class TestCollectorBuilder:
    def test_build_from_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\n")
        builder = CollectorBuilder()
        collector = builder.file_source(str(f)).memory_store().build()
        assert isinstance(collector, Collector)
        count = collector.collect()
        assert count == 2

    def test_build_with_filters(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("short\nthis is a longer line that passes\n")
        builder = CollectorBuilder()
        collector = builder.file_source(str(f)).memory_store().length_filter(min_length=20).build()
        count = collector.collect()
        assert count == 1

    def test_build_batch(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\n" * 25)
        builder = CollectorBuilder()
        collector = builder.file_source(str(f)).memory_store().batch(batch_size=10).build()
        assert isinstance(collector, BatchCollector)
        count = collector.collect()
        assert count == 25

    def test_build_parallel(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha\n")
        f2.write_text("beta\n")
        b1 = CollectorBuilder().file_source(str(f1)).memory_store()
        b2 = CollectorBuilder().file_source(str(f2)).memory_store()
        pc = b1.build_parallel([b1, b2])
        assert isinstance(pc, ParallelCollector)

    def test_fluent_chaining(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\n")
        collector = (CollectorBuilder()
            .file_source(str(f))
            .memory_store()
            .dedup_filter()
            .length_filter(min_length=1)
            .build())
        count = collector.collect()
        assert count == 1


class TestDataSource:
    def test_add_and_read(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha\n")
        f2.write_text("beta\n")
        ds = DataSource()
        ds.add_file(str(f1)).add_file(str(f2))
        records = list(ds.read_all())
        assert len(records) == 2
        assert ds.count() == 2

    def test_read_single_source(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("alpha\n")
        ds = DataSource().add_file(str(f1))
        records = list(ds.read(0))
        assert len(records) == 1

    def test_list_sources(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("")
        ds = DataSource().add_file(str(f1))
        names = ds.list_sources()
        assert len(names) == 1


class TestDataSink:
    def test_write_to_all(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        ds = DataSink().add(s1).add(s2)
        ds.write(Record(content="hello"))
        assert s1.count() == 1
        assert s2.count() == 1

    def test_write_all(self):
        s1 = MemoryStore()
        ds = DataSink().add(s1)
        records = [Record(content="a"), Record(content="b")]
        count = ds.write_all(records)
        assert count == 2
        assert s1.count() == 2

    def test_list_stores(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        ds = DataSink().add(s1).add(s2)
        names = ds.list_stores()
        assert len(names) == 2

    def test_count(self):
        s1 = MemoryStore()
        s2 = MemoryStore()
        ds = DataSink().add(s1).add(s2)
        ds.write(Record(content="a"))
        ds.write(Record(content="b"))
        assert ds.count() == 2


class TestDataTransformer:
    def test_transform(self):
        dt = DataTransformer()
        dt.add_content_transform(lambda s: s.upper())
        r = Record(content="hello")
        result = dt.transform(r)
        assert result.content == "HELLO"

    def test_add_field(self):
        dt = DataTransformer()
        dt.add_field("tag", "test")
        r = Record(content="hello")
        result = dt.transform(r)
        assert result.metadata["tag"] == "test"

    def test_add_field_fn(self):
        dt = DataTransformer()
        dt.add_field_fn("length", lambda r: len(r.content))
        r = Record(content="hello")
        result = dt.transform(r)
        assert result.metadata["length"] == 5

    def test_transform_all(self):
        dt = DataTransformer()
        dt.add_content_transform(lambda s: s.upper())
        records = [Record(content="a"), Record(content="b")]
        results = dt.transform_all(records)
        assert results[0].content == "A"
        assert results[1].content == "B"

    def test_stats(self):
        dt = DataTransformer()
        dt.add_content_transform(lambda s: s.upper())
        dt.transform(Record(content="hello"))
        dt.transform(Record(content="world"))
        assert dt.stats["transformed"] == 2
        assert dt.stats["errors"] == 0


class TestRecordToWorldMapper:
    def test_record_to_cell_signal(self):
        mapper = RecordToWorldMapper()
        r = Record(content="test data", metadata={"energy": 0.5})
        cell = mapper.record_to_cell_signal(r)
        assert "energy" in cell
        assert "temperature" in cell
        assert "signal" in cell
        assert cell["energy"] == 0.5

    def test_record_to_position(self):
        mapper = RecordToWorldMapper()
        r = Record(content="test", metadata={"position": [10, 5, 20]})
        pos = mapper.record_to_position(r, 0)
        assert pos == (10, 5, 20)

    def test_record_to_position_hash(self):
        mapper = RecordToWorldMapper()
        r = Record(content="test data for hashing")
        pos = mapper.record_to_position(r, 0)
        assert 0 <= pos[0] < 64
        assert 0 <= pos[1] < 32
        assert 0 <= pos[2] < 64

    def test_records_to_world_ops(self):
        mapper = RecordToWorldMapper()
        records = [Record(content="a"), Record(content="b")]
        ops = mapper.records_to_world_ops(records)
        assert len(ops) == 2
        assert ops[0]["type"] == "place_cell"
        assert "x" in ops[0]


class TestWorldFeedConfig:
    def test_defaults(self):
        config = WorldFeedConfig()
        assert config.grid_size == (64, 32, 64)
        assert config.energy_scale == 1.0
        assert config.feed_radius == 5

    def test_custom(self):
        config = WorldFeedConfig(grid_size=(32, 16, 32), energy_scale=2.0)
        assert config.grid_size == (32, 16, 32)
        assert config.energy_scale == 2.0


class TestWorldGridBridge:
    def test_inject_records(self):
        bridge = WorldGridBridge()
        records = [Record(content="hello"), Record(content="world")]
        count = bridge.inject_records(records)
        assert count == 2
        assert bridge.stats["injected"] == 2

    def test_inject_with_position(self):
        bridge = WorldGridBridge()
        records = [Record(content="test", metadata={"position": [10, 5, 20]})]
        count = bridge.inject_records(records)
        assert count == 1

    def test_read_grid_as_records(self):
        bridge = WorldGridBridge()
        records = bridge.read_grid_as_records(center=(32, 16, 32), radius=2)
        assert isinstance(records, list)

    def test_grid_to_source(self):
        bridge = WorldGridBridge()
        source = bridge.grid_to_source(center=(32, 16, 32), radius=2)
        assert source.name == "world_grid"
        records = list(source.read())
        assert isinstance(records, list)

    def test_world_store_adapter(self):
        bridge = WorldGridBridge()
        adapter = WorldStoreAdapter(bridge)
        adapter.write(Record(content="hello"))
        assert adapter.count() == 1
        assert bridge.stats["injected"] == 1


class TestCollectionWorldPipeline:
    def test_run(self):
        pipeline = CollectionWorldPipeline(
            source=GeneratorSource(lambda: iter([
                Record(content="a"), Record(content="b")
            ])),
        )
        count = pipeline.run()
        assert count == 2
        assert pipeline.stats["total_collected"] == 2
        assert pipeline.stats["total_injected"] == 2

    def test_run_with_filter(self):
        pipeline = CollectionWorldPipeline(
            source=GeneratorSource(lambda: iter([
                Record(content="short"),
                Record(content="this is a much longer record"),
            ])),
            filters=[LengthFilter(min_length=10)],
        )
        count = pipeline.run()
        assert count == 1
        assert pipeline.stats["filtered"] == 1


class TestTrainingDataAdapter:
    def test_records_to_text(self):
        adapter = TrainingDataAdapter()
        records = [Record(content="hello"), Record(content="world")]
        text = adapter.records_to_text(records)
        assert text == "hello\nworld"
        assert adapter.stats["accepted"] == 2

    def test_deduplication(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(deduplicate=True))
        records = [Record(content="hello"), Record(content="hello"), Record(content="world")]
        text = adapter.records_to_text(records)
        assert text == "hello\nworld"
        assert adapter.stats["deduplicated"] == 1

    def test_min_length_filter(self):
        adapter = TrainingDataAdapter(TrainingDataConfig(min_length=5))
        records = [Record(content="hi"), Record(content="hello world")]
        text = adapter.records_to_text(records)
        assert text == "hello world"
        assert adapter.stats["too_short"] == 1

    def test_records_to_numpy(self):
        adapter = TrainingDataAdapter()
        records = [Record(content="hello"), Record(content="world")]
        data, vocab_size = adapter.records_to_numpy(records)
        assert len(data) > 0
        assert vocab_size > 0

    def test_records_to_text_file(self, tmp_path):
        adapter = TrainingDataAdapter()
        records = [Record(content="hello"), Record(content="world")]
        path = str(tmp_path / "train.txt")
        count = adapter.records_to_text_file(records, path)
        assert count > 0
        assert (tmp_path / "train.txt").exists()

    def test_reset(self):
        adapter = TrainingDataAdapter()
        adapter.records_to_text([Record(content="hello")])
        assert adapter.stats["accepted"] == 1
        adapter.reset()
        assert adapter.stats["accepted"] == 0


class TestTrainingDatasetBuilder:
    def test_add_records(self):
        builder = TrainingDatasetBuilder()
        builder.add_records([Record(content="hello"), Record(content="world")])
        assert builder.record_count == 2

    def test_build_text(self):
        builder = TrainingDatasetBuilder()
        builder.add_records([Record(content="hello"), Record(content="world")])
        text = builder.build_text()
        assert text == "hello\nworld"

    def test_build_numpy(self):
        builder = TrainingDatasetBuilder()
        builder.add_records([Record(content="hello"), Record(content="world")])
        data, vocab_size = builder.build_numpy()
        assert len(data) > 0
        assert vocab_size > 0

    def test_build_dataset(self):
        builder = TrainingDatasetBuilder()
        builder.add_records([Record(content="hello"), Record(content="world")])
        data, stoi, itos = builder.build_dataset()
        assert len(data) > 0
        assert len(stoi) > 0
        assert len(itos) > 0

    def test_save_text(self, tmp_path):
        builder = TrainingDatasetBuilder()
        builder.add_records([Record(content="hello"), Record(content="world")])
        path = str(tmp_path / "train.txt")
        count = builder.save_text(path)
        assert count > 0
        assert (tmp_path / "train.txt").exists()

    def test_add_from_text(self):
        builder = TrainingDatasetBuilder()
        builder.add_from_text("hello\nworld\nfoo")
        assert builder.record_count == 3

    def test_add_from_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\n")
        builder = TrainingDatasetBuilder()
        builder.add_from_file(str(f))
        assert builder.record_count == 2


class TestCollectorTrainingBridge:
    def test_collect_and_prepare(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\nfoo bar\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector)
        data, vocab_size = bridge.collect_and_prepare()
        assert len(data) > 0
        assert vocab_size > 0
        assert bridge.stats["collected"] == 2

    def test_collect_and_save_text(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("hello\nworld\n")
        out = tmp_path / "output.txt"
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector)
        count = bridge.collect_and_save_text(str(out))
        assert count > 0
        assert out.exists()

    def test_get_text(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\nworld\n")
        collector = Collector(FileSource(str(f)), MemoryStore())
        bridge = CollectorTrainingBridge(collector)
        bridge.collect_and_prepare()
        text = bridge.get_text()
        assert "hello" in text
        assert "world" in text
