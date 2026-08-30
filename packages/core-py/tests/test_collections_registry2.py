"""Tests for domains.collections.registry — CollectionRegistry, get_registry."""

from domains.collections.registry import CollectionRegistry, get_registry
from domains.collections.sources import GeneratorSource, Record
from domains.collections.stores import MemoryStore
from domains.collections.filters import LengthFilter, KeywordFilter


def _source(name="src1"):
    return GeneratorSource(lambda: [Record(content="hello world"), Record(content="foo bar")], name=name)


def _store(name="st1"):
    return MemoryStore(name=name)


class TestCollectionRegistry:
    def test_init_empty(self):
        reg = CollectionRegistry()
        assert reg.list_sources() == []
        assert reg.list_stores() == []
        assert reg.list_filters() == []
        assert reg.list_pipelines() == []

    def test_register_and_get_source(self):
        reg = CollectionRegistry()
        s = _source("s1")
        reg.register_source("s1", s)
        assert reg.get_source("s1") is s

    def test_get_missing_source(self):
        reg = CollectionRegistry()
        assert reg.get_source("nonexistent") is None

    def test_register_and_get_store(self):
        reg = CollectionRegistry()
        st = _store("st1")
        reg.register_store("st1", st)
        assert reg.get_store("st1") is st

    def test_get_missing_store(self):
        reg = CollectionRegistry()
        assert reg.get_store("nonexistent") is None

    def test_register_and_get_filter(self):
        reg = CollectionRegistry()
        f = LengthFilter(min_length=5)
        reg.register_filter("lf", f)
        assert reg.get_filter("lf") is f

    def test_get_missing_filter(self):
        reg = CollectionRegistry()
        assert reg.get_filter("nope") is None

    def test_register_source_overwrites(self):
        reg = CollectionRegistry()
        s1 = _source("s1")
        s2 = _source("s2")
        reg.register_source("key", s1)
        reg.register_source("key", s2)
        assert reg.get_source("key") is s2

    def test_list_sources(self):
        reg = CollectionRegistry()
        reg.register_source("a", _source("a"))
        reg.register_source("b", _source("b"))
        names = reg.list_sources()
        assert "a" in names
        assert "b" in names

    def test_list_stores(self):
        reg = CollectionRegistry()
        reg.register_store("x", _store("x"))
        reg.register_store("y", _store("y"))
        names = reg.list_stores()
        assert "x" in names
        assert "y" in names

    def test_list_filters(self):
        reg = CollectionRegistry()
        reg.register_filter("f1", LengthFilter())
        reg.register_filter("f2", KeywordFilter(keywords=["py"]))
        names = reg.list_filters()
        assert "f1" in names
        assert "f2" in names


class TestCreatePipeline:
    def test_create_pipeline_success(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        p = reg.create_pipeline("p1", "src", "sto")
        assert p is not None
        assert p.name == "p1"
        assert reg.get_pipeline("p1") is p

    def test_create_pipeline_missing_source(self):
        reg = CollectionRegistry()
        reg.register_store("sto", _store("sto"))
        p = reg.create_pipeline("p1", "missing_src", "sto")
        assert p is None

    def test_create_pipeline_missing_store(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        p = reg.create_pipeline("p1", "src", "missing_sto")
        assert p is None

    def test_create_pipeline_both_missing(self):
        reg = CollectionRegistry()
        p = reg.create_pipeline("p1", "no", "no")
        assert p is None

    def test_create_pipeline_with_filters(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.register_filter("lf", LengthFilter(min_length=1))
        reg.register_filter("kf", KeywordFilter(keywords=["hello"]))
        p = reg.create_pipeline("p1", "src", "sto", filter_names=["lf", "kf"])
        assert p is not None
        assert len(p.filters) == 2

    def test_create_pipeline_partial_filter_names(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.register_filter("lf", LengthFilter())
        p = reg.create_pipeline("p1", "src", "sto", filter_names=["lf", "nonexistent"])
        assert p is not None
        assert len(p.filters) == 1

    def test_create_pipeline_no_filter_names(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        p = reg.create_pipeline("p1", "src", "sto", filter_names=None)
        assert p is not None
        assert p.filters == []

    def test_create_pipeline_empty_filter_names(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        p = reg.create_pipeline("p1", "src", "sto", filter_names=[])
        assert p is not None
        assert p.filters == []


class TestRemovePipeline:
    def test_remove_existing(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.create_pipeline("p1", "src", "sto")
        assert reg.remove_pipeline("p1") is True
        assert reg.get_pipeline("p1") is None

    def test_remove_nonexistent(self):
        reg = CollectionRegistry()
        assert reg.remove_pipeline("ghost") is False


class TestCollect:
    def test_collect_missing_pipeline(self):
        reg = CollectionRegistry()
        assert reg.collect("nope") == 0

    def test_collect_existing_pipeline(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.create_pipeline("p1", "src", "sto")
        count = reg.collect("p1")
        assert count == 2

    def test_collect_with_filter_reduces_count(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.register_filter("lf", LengthFilter(min_length=100))
        reg.create_pipeline("p1", "src", "sto", filter_names=["lf"])
        count = reg.collect("p1")
        assert count == 0

    def test_collect_twice_accumulates(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.create_pipeline("p1", "src", "sto")
        reg.collect("p1")
        reg.collect("p1")
        assert reg.get_store("sto").count() == 4


class TestListPipelines:
    def test_list_pipelines(self):
        reg = CollectionRegistry()
        reg.register_source("src", _source("src"))
        reg.register_store("sto", _store("sto"))
        reg.create_pipeline("p1", "src", "sto")
        reg.create_pipeline("p2", "src", "sto")
        names = reg.list_pipelines()
        assert "p1" in names
        assert "p2" in names

    def test_list_pipelines_empty(self):
        reg = CollectionRegistry()
        assert reg.list_pipelines() == []


class TestStats:
    def test_stats_empty(self):
        reg = CollectionRegistry()
        s = reg.stats()
        assert s["sources"] == []
        assert s["stores"] == []
        assert s["filters"] == []
        assert s["pipelines"] == {}

    def test_stats_with_entries(self):
        reg = CollectionRegistry()
        reg.register_source("s", _source("s"))
        reg.register_store("t", _store("t"))
        reg.register_filter("f", LengthFilter())
        reg.create_pipeline("p", "s", "t")
        s = reg.stats()
        assert "s" in s["sources"]
        assert "t" in s["stores"]
        assert "f" in s["filters"]
        assert "p" in s["pipelines"]
        assert isinstance(s["pipelines"]["p"], dict)


class TestGetRegistrySingleton:
    def test_returns_same_instance(self):
        a = get_registry()
        b = get_registry()
        assert a is b

    def test_is_collection_registry(self):
        r = get_registry()
        assert isinstance(r, CollectionRegistry)

    def test_singleton_persists_state(self):
        r = get_registry()
        before = len(r.list_sources())
        r.register_source("singleton_test", _source("st"))
        r2 = get_registry()
        assert len(r2.list_sources()) > before

    def test_singleton_resets_on_new_registry(self):
        import domains.collections.registry as mod
        old = mod._default_registry
        mod._default_registry = None
        r = get_registry()
        assert isinstance(r, CollectionRegistry)
        assert r is not old or len(r.list_sources()) == 0
        mod._default_registry = old
