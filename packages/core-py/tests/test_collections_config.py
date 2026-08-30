"""Tests for domains.collections.config — pipeline configuration dataclasses.

Covers: SourceConfig, StoreConfig, FilterConfig, PipelineConfig defaults and custom values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

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

    def test_custom(self):
        c = SourceConfig(type="url", url="https://example.com", timeout=60)
        assert c.type == "url"
        assert c.url == "https://example.com"
        assert c.timeout == 60

    def test_name_field(self):
        c = SourceConfig(name="my_source")
        assert c.name == "my_source"

    def test_headers_default_empty(self):
        c = SourceConfig()
        assert c.headers == {}
        assert isinstance(c.headers, dict)

    def test_headers_custom(self):
        c = SourceConfig(headers={"Authorization": "Bearer tok"})
        assert c.headers["Authorization"] == "Bearer tok"

    def test_poll_interval_float(self):
        c = SourceConfig(poll_interval=1.5)
        assert c.poll_interval == 1.5

    def test_equality(self):
        a = SourceConfig(type="url", timeout=60)
        b = SourceConfig(type="url", timeout=60)
        assert a == b

    def test_inequality(self):
        a = SourceConfig(type="url")
        b = SourceConfig(type="file")
        assert a != b

    def test_repr(self):
        c = SourceConfig(type="url")
        assert "url" in repr(c)

    def test_path_custom(self):
        c = SourceConfig(path="/data/input.json")
        assert c.path == "/data/input.json"

    def test_timeout_zero(self):
        c = SourceConfig(timeout=0)
        assert c.timeout == 0

    def test_timeout_negative(self):
        c = SourceConfig(timeout=-1)
        assert c.timeout == -1

    def test_poll_interval_zero(self):
        c = SourceConfig(poll_interval=0.0)
        assert c.poll_interval == 0.0

    def test_headers_mutable(self):
        c = SourceConfig()
        c.headers["X-Custom"] = "val"
        assert c.headers["X-Custom"] == "val"

    def test_multiple_instances_independent(self):
        a = SourceConfig()
        b = SourceConfig()
        a.headers["k"] = "v"
        assert "k" not in b.headers

    def test_type_variants(self):
        for t in ("file", "url", "s3", "kafka", "sqlite"):
            c = SourceConfig(type=t)
            assert c.type == t

    def test_fields_all_present(self):
        c = SourceConfig()
        fields = {f.name for f in c.__dataclass_fields__.values()}
        assert fields == {"type", "path", "url", "name", "timeout", "poll_interval", "headers"}


class TestStoreConfig:
    def test_defaults(self):
        c = StoreConfig()
        assert c.type == "file"
        assert c.path == ""
        assert c.name == ""
        assert c.max_size == 10000

    def test_custom(self):
        c = StoreConfig(type="sqlite", path="/tmp/db.sqlite", max_size=50000)
        assert c.type == "sqlite"
        assert c.max_size == 50000

    def test_name_field(self):
        c = StoreConfig(name="my_store")
        assert c.name == "my_store"

    def test_equality(self):
        a = StoreConfig(type="sqlite", max_size=500)
        b = StoreConfig(type="sqlite", max_size=500)
        assert a == b

    def test_inequality(self):
        a = StoreConfig(type="file")
        b = StoreConfig(type="sqlite")
        assert a != b

    def test_repr(self):
        c = StoreConfig(type="memory")
        assert "memory" in repr(c)

    def test_max_size_zero(self):
        c = StoreConfig(max_size=0)
        assert c.max_size == 0

    def test_max_size_large(self):
        c = StoreConfig(max_size=10**9)
        assert c.max_size == 10**9

    def test_path_custom(self):
        c = StoreConfig(path="/var/data/store.json")
        assert c.path == "/var/data/store.json"

    def test_type_variants(self):
        for t in ("file", "sqlite", "memory", "redis", "s3"):
            c = StoreConfig(type=t)
            assert c.type == t

    def test_fields_all_present(self):
        c = StoreConfig()
        fields = {f.name for f in c.__dataclass_fields__.values()}
        assert fields == {"type", "path", "name", "max_size"}

    def test_multiple_instances_independent(self):
        a = StoreConfig()
        b = StoreConfig()
        a.max_size = 999
        assert b.max_size == 10000

    def test_path_empty_string(self):
        c = StoreConfig()
        assert c.path == ""

    def test_name_default_empty(self):
        c = StoreConfig()
        assert c.name == ""


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

    def test_custom(self):
        c = FilterConfig(type="keyword", keywords=["python", "test"], mode="exclude")
        assert c.keywords == ["python", "test"]
        assert c.mode == "exclude"

    def test_pattern_field(self):
        c = FilterConfig(pattern=r"^\d{4}-\d{2}-\d{2}$")
        assert c.pattern == r"^\d{4}-\d{2}-\d{2}$"

    def test_allowed_chars_ratio_custom(self):
        c = FilterConfig(allowed_chars_ratio=0.95)
        assert c.allowed_chars_ratio == 0.95

    def test_equality(self):
        a = FilterConfig(type="length", min_length=5)
        b = FilterConfig(type="length", min_length=5)
        assert a == b

    def test_inequality(self):
        a = FilterConfig(type="length")
        b = FilterConfig(type="keyword")
        assert a != b

    def test_repr(self):
        c = FilterConfig(type="regex")
        assert "regex" in repr(c)

    def test_keywords_mutable(self):
        c = FilterConfig()
        c.keywords.append("new")
        assert "new" in c.keywords

    def test_keywords_default_empty_list(self):
        c = FilterConfig()
        assert isinstance(c.keywords, list)
        assert len(c.keywords) == 0

    def test_min_length_zero(self):
        c = FilterConfig(min_length=0)
        assert c.min_length == 0

    def test_max_length_zero(self):
        c = FilterConfig(max_length=0)
        assert c.max_length == 0

    def test_min_greater_than_max(self):
        c = FilterConfig(min_length=100, max_length=10)
        assert c.min_length > c.max_length

    def test_mode_variants(self):
        for m in ("include", "exclude", "regex", "exact"):
            c = FilterConfig(mode=m)
            assert c.mode == m

    def test_type_variants(self):
        for t in ("length", "keyword", "regex", "semantic", "dedup"):
            c = FilterConfig(type=t)
            assert c.type == t

    def test_multiple_keywords(self):
        kw = ["alpha", "beta", "gamma", "delta"]
        c = FilterConfig(keywords=kw)
        assert len(c.keywords) == 4

    def test_allowed_chars_ratio_zero(self):
        c = FilterConfig(allowed_chars_ratio=0.0)
        assert c.allowed_chars_ratio == 0.0

    def test_allowed_chars_ratio_one(self):
        c = FilterConfig(allowed_chars_ratio=1.0)
        assert c.allowed_chars_ratio == 1.0

    def test_fields_all_present(self):
        c = FilterConfig()
        fields = {f.name for f in c.__dataclass_fields__.values()}
        assert fields == {"type", "min_length", "max_length", "keywords", "mode", "pattern", "allowed_chars_ratio"}

    def test_multiple_instances_independent(self):
        a = FilterConfig()
        b = FilterConfig()
        a.keywords.append("test")
        assert len(b.keywords) == 0


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.name == ""
        assert c.collect_interval == 0.0
        assert c.max_rounds is None

    def test_nested(self):
        c = PipelineConfig(
            name="test_pipeline",
            source=SourceConfig(type="url"),
            store=StoreConfig(type="memory"),
            filters=[FilterConfig(type="length")],
        )
        assert c.source.type == "url"
        assert c.store.type == "memory"
        assert len(c.filters) == 1

    def test_source_default(self):
        c = PipelineConfig()
        assert isinstance(c.source, SourceConfig)
        assert c.source.type == "file"

    def test_store_default(self):
        c = PipelineConfig()
        assert isinstance(c.store, StoreConfig)
        assert c.store.type == "file"

    def test_filters_default_empty(self):
        c = PipelineConfig()
        assert isinstance(c.filters, list)
        assert len(c.filters) == 0

    def test_collect_interval_custom(self):
        c = PipelineConfig(collect_interval=30.5)
        assert c.collect_interval == 30.5

    def test_max_rounds_custom(self):
        c = PipelineConfig(max_rounds=10)
        assert c.max_rounds == 10

    def test_max_rounds_none(self):
        c = PipelineConfig(max_rounds=None)
        assert c.max_rounds is None

    def test_equality(self):
        a = PipelineConfig(name="p1", collect_interval=1.0)
        b = PipelineConfig(name="p1", collect_interval=1.0)
        assert a == b

    def test_inequality(self):
        a = PipelineConfig(name="p1")
        b = PipelineConfig(name="p2")
        assert a != b

    def test_repr(self):
        c = PipelineConfig(name="test")
        assert "test" in repr(c)

    def test_multiple_filters(self):
        filters = [FilterConfig(type="length"), FilterConfig(type="keyword")]
        c = PipelineConfig(filters=filters)
        assert len(c.filters) == 2

    def test_filters_independent_instances(self):
        c1 = PipelineConfig()
        c2 = PipelineConfig()
        c1.filters.append(FilterConfig())
        assert len(c2.filters) == 0

    def test_name_custom(self):
        c = PipelineConfig(name="production_pipeline")
        assert c.name == "production_pipeline"

    def test_deeply_nested(self):
        c = PipelineConfig(
            name="deep",
            source=SourceConfig(type="s3", path="/bucket/data"),
            store=StoreConfig(type="redis", max_size=100000),
            filters=[
                FilterConfig(type="length", min_length=5),
                FilterConfig(type="keyword", keywords=["important"]),
                FilterConfig(type="regex", pattern=r"^error"),
            ],
            collect_interval=5.0,
            max_rounds=100,
        )
        assert c.source.type == "s3"
        assert c.store.type == "redis"
        assert len(c.filters) == 3
        assert c.collect_interval == 5.0
        assert c.max_rounds == 100

    def test_fields_all_present(self):
        c = PipelineConfig()
        fields = {f.name for f in c.__dataclass_fields__.values()}
        assert fields == {"name", "source", "store", "filters", "collect_interval", "max_rounds"}

    def test_collect_interval_zero(self):
        c = PipelineConfig(collect_interval=0.0)
        assert c.collect_interval == 0.0

    def test_max_rounds_large(self):
        c = PipelineConfig(max_rounds=10**6)
        assert c.max_rounds == 10**6

    def test_max_rounds_negative(self):
        c = PipelineConfig(max_rounds=-1)
        assert c.max_rounds == -1

    def test_collect_interval_negative(self):
        c = PipelineConfig(collect_interval=-5.0)
        assert c.collect_interval == -5.0

    def test_name_empty_string(self):
        c = PipelineConfig(name="")
        assert c.name == ""

    def test_source_replacement(self):
        c = PipelineConfig()
        new_source = SourceConfig(type="kafka")
        c.source = new_source
        assert c.source.type == "kafka"

    def test_store_replacement(self):
        c = PipelineConfig()
        new_store = StoreConfig(type="memory")
        c.store = new_store
        assert c.store.type == "memory"

    def test_filter_append(self):
        c = PipelineConfig()
        c.filters.append(FilterConfig(type="semantic"))
        assert len(c.filters) == 1
        assert c.filters[0].type == "semantic"
