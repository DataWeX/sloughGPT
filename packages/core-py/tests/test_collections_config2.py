"""Tests for domains.collections.config — SourceConfig, StoreConfig, FilterConfig, PipelineConfig."""

from dataclasses import fields, asdict

from domains.collections.config import (
    SourceConfig,
    StoreConfig,
    FilterConfig,
    PipelineConfig,
)


class TestSourceConfig:
    def test_defaults(self):
        c = SourceConfig()
        assert c.type == "file"
        assert c.path == ""
        assert c.url == ""
        assert c.name == ""
        assert c.timeout == 30
        assert c.poll_interval == 60.0
        assert c.headers == {}

    def test_custom_values(self):
        c = SourceConfig(type="url", path="/data", url="http://example.com", name="web",
                         timeout=60, poll_interval=120.5, headers={"Auth": "tok"})
        assert c.type == "url"
        assert c.path == "/data"
        assert c.url == "http://example.com"
        assert c.name == "web"
        assert c.timeout == 60
        assert c.poll_interval == 120.5
        assert c.headers == {"Auth": "tok"}

    def test_dataclass_fields_exist(self):
        names = {f.name for f in fields(SourceConfig)}
        assert names == {"type", "path", "url", "name", "timeout", "poll_interval", "headers"}

    def test_equality(self):
        a = SourceConfig(type="file", path="/a")
        b = SourceConfig(type="file", path="/a")
        assert a == b

    def test_inequality(self):
        a = SourceConfig(timeout=10)
        b = SourceConfig(timeout=20)
        assert a != b

    def test_headers_mutable_default_is_independent(self):
        a = SourceConfig()
        b = SourceConfig()
        a.headers["key"] = "val"
        assert b.headers == {}

    def test_to_dict_roundtrip(self):
        c = SourceConfig(type="api", name="test")
        d = asdict(c)
        assert d["type"] == "api"
        assert d["name"] == "test"
        assert "headers" in d

    def test_repr(self):
        c = SourceConfig(name="x")
        r = repr(c)
        assert "SourceConfig" in r
        assert "name='x'" in r


class TestStoreConfig:
    def test_defaults(self):
        c = StoreConfig()
        assert c.type == "file"
        assert c.path == ""
        assert c.name == ""
        assert c.max_size == 10000

    def test_custom_values(self):
        c = StoreConfig(type="memory", path="/tmp/store", name="mem", max_size=500)
        assert c.type == "memory"
        assert c.path == "/tmp/store"
        assert c.name == "mem"
        assert c.max_size == 500

    def test_dataclass_fields_exist(self):
        names = {f.name for f in fields(StoreConfig)}
        assert names == {"type", "path", "name", "max_size"}

    def test_equality(self):
        a = StoreConfig(max_size=100)
        b = StoreConfig(max_size=100)
        assert a == b

    def test_to_dict(self):
        c = StoreConfig(type="redis", max_size=999)
        d = asdict(c)
        assert d["type"] == "redis"
        assert d["max_size"] == 999


class TestFilterConfig:
    def test_defaults(self):
        c = FilterConfig()
        assert c.type == "length"
        assert c.min_length == 10
        assert c.max_length == 100000
        assert c.keywords == []
        assert c.mode == "include"
        assert c.pattern == ""
        assert c.allowed_chars_ratio == 0.8

    def test_custom_values(self):
        c = FilterConfig(type="keyword", keywords=["py", "thon"], mode="exclude",
                         min_length=5, max_length=500, pattern=r"\d+",
                         allowed_chars_ratio=0.9)
        assert c.type == "keyword"
        assert c.keywords == ["py", "thon"]
        assert c.mode == "exclude"
        assert c.min_length == 5
        assert c.max_length == 500
        assert c.pattern == r"\d+"
        assert c.allowed_chars_ratio == 0.9

    def test_dataclass_fields_exist(self):
        names = {f.name for f in fields(FilterConfig)}
        assert names == {"type", "min_length", "max_length", "keywords", "mode", "pattern", "allowed_chars_ratio"}

    def test_keywords_mutable_default_independence(self):
        a = FilterConfig()
        b = FilterConfig()
        a.keywords.append("test")
        assert b.keywords == []

    def test_equality(self):
        a = FilterConfig(type="dedup")
        b = FilterConfig(type="dedup")
        assert a == b


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.name == ""
        assert isinstance(c.source, SourceConfig)
        assert isinstance(c.store, StoreConfig)
        assert c.filters == []
        assert c.collect_interval == 0.0
        assert c.max_rounds is None

    def test_custom_source_and_store(self):
        src = SourceConfig(type="rss", url="http://feed.example.com")
        sto = StoreConfig(type="memory", max_size=500)
        c = PipelineConfig(name="rss-pipe", source=src, store=sto,
                           collect_interval=30.0, max_rounds=10)
        assert c.name == "rss-pipe"
        assert c.source.type == "rss"
        assert c.source.url == "http://feed.example.com"
        assert c.store.type == "memory"
        assert c.store.max_size == 500
        assert c.collect_interval == 30.0
        assert c.max_rounds == 10

    def test_nested_factory_independence(self):
        a = PipelineConfig()
        b = PipelineConfig()
        a.source.path = "/a"
        assert b.source.path == ""

    def test_filters_list_independence(self):
        a = PipelineConfig()
        b = PipelineConfig()
        a.filters.append(FilterConfig(type="dedup"))
        assert b.filters == []

    def test_max_rounds_none(self):
        c = PipelineConfig()
        assert c.max_rounds is None

    def test_to_dict_roundtrip(self):
        c = PipelineConfig(name="test", collect_interval=5.0, max_rounds=3)
        d = asdict(c)
        assert d["name"] == "test"
        assert d["collect_interval"] == 5.0
        assert d["max_rounds"] == 3
        assert "source" in d
        assert "store" in d

    def test_repr(self):
        c = PipelineConfig(name="p1")
        r = repr(c)
        assert "PipelineConfig" in r
        assert "name='p1'" in r
