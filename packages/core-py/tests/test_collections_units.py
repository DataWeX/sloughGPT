from domains.collections.sources import Record
from domains.collections.filters import (
    LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
    LanguageFilter, FilterChain, SamplerFilter, TransformFilter,
    TruncateFilter, PrefixFilter, MetadataFilter,
)
from domains.collections.stores import MemoryStore, FileStore
from domains.collections.pipeline import CollectionPipeline
from domains.collections.registry import CollectionRegistry, get_registry


# ── Record ─────────────────────────────────────────────────────────

class TestRecord:
    def test_create(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_to_dict(self):
        r = Record(content="test", metadata={"a": 1})
        d = r.to_dict()
        assert d["content"] == "test"
        assert d["metadata"]["a"] == 1

    def test_preserves_existing_timestamp(self):
        r = Record(content="x", metadata={"timestamp": 123})
        assert r.metadata["timestamp"] == 123


# ── Filters ────────────────────────────────────────────────────────

class TestLengthFilter:
    def test_within_range(self):
        f = LengthFilter(min_length=3, max_length=10)
        assert f.accept(Record(content="hello"))

    def test_too_short(self):
        f = LengthFilter(min_length=5)
        assert not f.accept(Record(content="hi"))

    def test_too_long(self):
        f = LengthFilter(max_length=3)
        assert not f.accept(Record(content="toolong"))


class TestDedupFilter:
    def test_first_accepted(self):
        f = DedupFilter()
        assert f.accept(Record(content="hello"))

    def test_duplicate_rejected(self):
        f = DedupFilter()
        f.accept(Record(content="hello"))
        assert not f.accept(Record(content="hello"))

    def test_reset(self):
        f = DedupFilter()
        f.accept(Record(content="hello"))
        f.reset()
        assert f.accept(Record(content="hello"))


class TestKeywordFilter:
    def test_include_mode_match(self):
        f = KeywordFilter(keywords=["python", "rust"], mode="include")
        assert f.accept(Record(content="I love Python"))

    def test_include_mode_no_match(self):
        f = KeywordFilter(keywords=["java"], mode="include")
        assert not f.accept(Record(content="I love Python"))

    def test_exclude_mode(self):
        f = KeywordFilter(keywords=["spam"], mode="exclude")
        assert not f.accept(Record(content="this is spam"))

    def test_empty_keywords_accept_all(self):
        f = KeywordFilter(keywords=[])
        assert f.accept(Record(content="anything"))


class TestRegexFilter:
    def test_include_match(self):
        f = RegexFilter(pattern=r"\d{3}", mode="include")
        assert f.accept(Record(content="code 123"))

    def test_include_no_match(self):
        f = RegexFilter(pattern=r"\d{3}", mode="include")
        assert not f.accept(Record(content="no numbers"))

    def test_exclude_mode(self):
        f = RegexFilter(pattern=r"error", mode="exclude")
        assert not f.accept(Record(content="an error occurred"))

    def test_empty_pattern_accepts_all(self):
        f = RegexFilter(pattern="")
        assert f.accept(Record(content="anything"))


class TestLanguageFilter:
    def test_english_text(self):
        f = LanguageFilter(allowed_chars_ratio=0.8)
        assert f.accept(Record(content="Hello world"))

    def test_non_english(self):
        f = LanguageFilter(allowed_chars_ratio=0.9)
        assert not f.accept(Record(content="こんにちは世界"))

    def test_empty_content(self):
        f = LanguageFilter()
        assert not f.accept(Record(content=""))


class TestFilterChain:
    def test_empty_chain_accepts(self):
        chain = FilterChain()
        assert chain.accept(Record(content="anything"))

    def test_reject_on_first_fail(self):
        chain = FilterChain([LengthFilter(min_length=100)])
        assert not chain.accept(Record(content="short"))

    def test_stats_tracking(self):
        chain = FilterChain([LengthFilter(min_length=3)])
        chain.accept(Record(content="hello"))
        chain.accept(Record(content="hi"))
        assert chain.stats["accepted"] == 1
        assert chain.stats["rejected"] == 1

    def test_filter_records(self):
        chain = FilterChain([LengthFilter(min_length=3)])
        records = [Record(content="ab"), Record(content="abcde")]
        result = chain.filter_records(records)
        assert len(result) == 1
        assert result[0].content == "abcde"

    def test_add_method(self):
        chain = FilterChain()
        chain.add(LengthFilter(min_length=5))
        assert len(chain.filters) == 1


class TestSamplerFilter:
    def test_rate_zero_rejects_all(self):
        f = SamplerFilter(rate=0)
        for _ in range(20):
            assert not f.accept(Record(content="x"))

    def test_rate_one_accepts_all(self):
        f = SamplerFilter(rate=1.0)
        for _ in range(20):
            assert f.accept(Record(content="x"))


class TestTransformFilter:
    def test_accept_always(self):
        f = TransformFilter()
        assert f.accept(Record(content="x"))

    def test_transform(self):
        f = TransformFilter(transform_fn=lambda r: Record(content=r.content.upper(), metadata=r.metadata))
        r = f.transform(Record(content="hello"))
        assert r.content == "HELLO"

    def test_transform_non_record(self):
        f = TransformFilter(transform_fn=lambda r: "transformed")
        r = f.transform(Record(content="x"))
        assert r.content == "transformed"


class TestTruncateFilter:
    def test_within_limit(self):
        f = TruncateFilter(max_length=10)
        r = Record(content="short")
        f.accept(r)
        assert r.content == "short"

    def test_truncates(self):
        f = TruncateFilter(max_length=5)
        r = Record(content="toolongcontent")
        f.accept(r)
        assert r.content == "toolo"


class TestPrefixFilter:
    def test_adds_prefix(self):
        f = PrefixFilter(prefix="[SYS] ")
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "[SYS] hello"

    def test_empty_prefix_noop(self):
        f = PrefixFilter(prefix="")
        r = Record(content="hello")
        f.accept(r)
        assert r.content == "hello"


class TestMetadataFilter:
    def test_include_match(self):
        f = MetadataFilter(key="source", values=["api", "web"], mode="include")
        assert f.accept(Record(content="x", metadata={"source": "api"}))

    def test_include_no_match(self):
        f = MetadataFilter(key="source", values=["api"], mode="include")
        assert not f.accept(Record(content="x", metadata={"source": "cli"}))

    def test_exclude_match(self):
        f = MetadataFilter(key="type", values=["spam"], mode="exclude")
        assert not f.accept(Record(content="x", metadata={"type": "spam"}))

    def test_empty_key_accepts(self):
        f = MetadataFilter(key="", values=["x"])
        assert f.accept(Record(content="x"))


# ── Stores ─────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_write_and_count(self):
        s = MemoryStore()
        s.write(Record(content="a"))
        s.write(Record(content="b"))
        assert s.count() == 2

    def test_read_all(self):
        s = MemoryStore()
        s.write(Record(content="a"))
        s.write(Record(content="b"))
        items = list(s.read_all())
        assert len(items) == 2

    def test_empty(self):
        s = MemoryStore()
        assert s.count() == 0
        assert list(s.read_all()) == []

    def test_name_default(self):
        assert MemoryStore().name == "memory"

    def test_name_custom(self):
        assert MemoryStore(name="my-store").name == "my-store"


class TestFileStore:
    def test_write_and_read(self, tmp_path):
        p = str(tmp_path / "test.jsonl")
        s = FileStore(p)
        s.write(Record(content="hello"))
        s.write(Record(content="world"))
        items = list(s.read_all())
        assert len(items) == 2
        assert items[0].content == "hello"

    def test_count(self, tmp_path):
        p = str(tmp_path / "test.jsonl")
        s = FileStore(p)
        s.write(Record(content="a"))
        assert s.count() == 1

    def test_empty_file(self, tmp_path):
        p = str(tmp_path / "empty.jsonl")
        s = FileStore(p)
        assert s.count() == 0
        assert list(s.read_all()) == []

    def test_name_derived(self, tmp_path):
        s = FileStore(str(tmp_path / "data.jsonl"))
        assert s.name == "file:data.jsonl"


# ── Pipeline & Registry ────────────────────────────────────────────

class TestCollectionPipeline:
    def test_name_default(self):
        class FakeSource:
            name = "src"
            def read(self): return iter([])
        class FakeStore:
            name = "st"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        p = CollectionPipeline(FakeSource(), FakeStore())
        assert p.name == "src->st"

    def test_name_custom(self):
        class FakeSource:
            name = "src"
            def read(self): return iter([])
        class FakeStore:
            name = "st"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        p = CollectionPipeline(FakeSource(), FakeStore(), name="my-pipeline")
        assert p.name == "my-pipeline"

    def test_stats(self):
        class FakeSource:
            name = "src"
            def read(self): return iter([])
        class FakeStore:
            name = "st"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        p = CollectionPipeline(FakeSource(), FakeStore())
        stats = p.stats
        assert stats["source"] == "src"
        assert stats["store"] == "st"
        assert "collector" in stats
        assert "filter_chain" in stats


class TestCollectionRegistry:
    def test_register_and_get_source(self):
        reg = CollectionRegistry()
        class S:
            name = "s1"
            def read(self): return iter([])
        reg.register_source("s1", S())
        assert reg.get_source("s1") is not None
        assert reg.get_source("nope") is None

    def test_register_and_get_store(self):
        reg = CollectionRegistry()
        class St:
            name = "st1"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        reg.register_store("st1", St())
        assert reg.get_store("st1") is not None
        assert reg.get_store("nope") is None

    def test_register_and_get_filter(self):
        reg = CollectionRegistry()
        f = LengthFilter(min_length=1)
        reg.register_filter("lf", f)
        assert reg.get_filter("lf") is f
        assert reg.get_filter("nope") is None

    def test_create_pipeline(self):
        reg = CollectionRegistry()
        class S:
            name = "s"
            def read(self): return iter([])
        class St:
            name = "st"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        reg.register_source("s", S())
        reg.register_store("st", St())
        p = reg.create_pipeline("p1", "s", "st")
        assert p is not None
        assert p.name == "p1"

    def test_create_pipeline_missing_source(self):
        reg = CollectionRegistry()
        class St:
            name = "st"
            def write(self, r): pass
            def read_all(self): return iter([])
            def count(self): return 0
        reg.register_store("st", St())
        assert reg.create_pipeline("p1", "missing", "st") is None

    def test_list_methods(self):
        reg = CollectionRegistry()
        class S:
            name = "s"
            def read(self): return iter([])
        reg.register_source("s1", S())
        assert "s1" in reg.list_sources()
        assert reg.list_stores() == []
        assert reg.list_filters() == []
        assert reg.list_pipelines() == []

    def test_remove_pipeline(self):
        reg = CollectionRegistry()
        assert reg.remove_pipeline("nonexistent") is False

    def test_stats(self):
        reg = CollectionRegistry()
        stats = reg.stats()
        assert "sources" in stats
        assert "stores" in stats
        assert "filters" in stats
        assert "pipelines" in stats

    def test_get_registry_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
