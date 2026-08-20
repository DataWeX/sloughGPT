"""Tests for domains.collections.config — SourceConfig, StoreConfig, FilterConfig, PipelineConfig."""

from domains.collections.config import SourceConfig, StoreConfig, FilterConfig, PipelineConfig


class TestSourceConfig:
    def test_defaults(self):
        sc = SourceConfig()
        assert sc.type == "file"
        assert sc.path == ""
        assert sc.timeout == 30
        assert sc.poll_interval == 60.0

    def test_custom(self):
        sc = SourceConfig(type="url", url="http://x", timeout=60)
        assert sc.type == "url"
        assert sc.url == "http://x"


class TestStoreConfig:
    def test_defaults(self):
        sc = StoreConfig()
        assert sc.type == "file"
        assert sc.max_size == 10000

    def test_custom(self):
        sc = StoreConfig(type="memory", max_size=100)
        assert sc.type == "memory"
        assert sc.max_size == 100


class TestFilterConfig:
    def test_defaults(self):
        fc = FilterConfig()
        assert fc.type == "length"
        assert fc.min_length == 10
        assert fc.keywords == []

    def test_custom(self):
        fc = FilterConfig(type="keyword", keywords=["python"], mode="include")
        assert fc.type == "keyword"
        assert fc.keywords == ["python"]


class TestPipelineConfig:
    def test_defaults(self):
        pc = PipelineConfig()
        assert pc.name == ""
        assert pc.filters == []
        assert pc.max_rounds is None

    def test_nested(self):
        pc = PipelineConfig(
            name="test",
            source=SourceConfig(type="url"),
            store=StoreConfig(type="memory"),
            filters=[FilterConfig(type="length")],
        )
        assert pc.source.type == "url"
        assert pc.store.type == "memory"
        assert len(pc.filters) == 1
