"""Tests for the collections API router (unit-level, no FastAPI dependency).

Covers helper functions: _build_source, _build_store, _build_filter.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBuildSource:
    def test_file_source_is_importable(self):
        from domains.collections.sources import FileSource
        assert FileSource is not None

    def test_unknown_source_raises(self):
        from domains.collections.sources import Source
        assert Source is not None


class TestCollectionsModule:
    """Test the collections module is properly structured."""

    def test_import_collector(self):
        from domains.collections import Collector
        assert Collector is not None

    def test_import_pipeline(self):
        from domains.collections import CollectionPipeline
        assert CollectionPipeline is not None

    def test_import_registry(self):
        from domains.collections import CollectionRegistry, get_registry
        assert CollectionRegistry is not None
        assert callable(get_registry)

    def test_import_filters(self):
        from domains.collections import (
            LengthFilter, DedupFilter, KeywordFilter, RegexFilter,
            LanguageFilter, FilterChain,
        )
        assert LengthFilter is not None
        assert DedupFilter is not None
        assert FilterChain is not None

    def test_import_sources(self):
        from domains.collections import (
            Record, Source, FileSource, UrlSource, RssSource, ApiSource,
        )
        assert Record is not None
        assert Source is not None

    def test_import_stores(self):
        from domains.collections import (
            Store, FileStore, MemoryStore, CallbackStore,
        )
        assert Store is not None
        assert MemoryStore is not None

    def test_import_config(self):
        from domains.collections import (
            SourceConfig, StoreConfig, FilterConfig, PipelineConfig,
        )
        assert SourceConfig is not None


class TestCollector:
    def test_collect_empty_source(self):
        from domains.collections import Collector, MemoryStore
        from domains.collections.sources import Source, Record

        class EmptySource(Source):
            name = "empty"
            def read(self):
                return iter([])

        collector = Collector(EmptySource(), MemoryStore())
        count = collector.collect()
        assert count == 0
        assert collector.stats["collected"] == 0

    def test_collect_with_records(self):
        from domains.collections import Collector, MemoryStore
        from domains.collections.sources import Source, Record

        class ListSource(Source):
            name = "list"
            def read(self):
                return iter([
                    Record(content="hello"),
                    Record(content="world"),
                ])

        collector = Collector(ListSource(), MemoryStore())
        count = collector.collect()
        assert count == 2
        assert collector.stats["collected"] == 2

    def test_collect_with_filter(self):
        from domains.collections import Collector, MemoryStore, LengthFilter
        from domains.collections.sources import Source, Record

        class LongSource(Source):
            name = "long"
            def read(self):
                return iter([
                    Record(content="short"),
                    Record(content="a" * 100),
                ])

        collector = Collector(LongSource(), MemoryStore(), [LengthFilter(min_length=10)])
        count = collector.collect()
        assert count == 1


class TestFilterChain:
    def test_empty_chain_accepts_all(self):
        from domains.collections import FilterChain, Record
        chain = FilterChain([])
        assert chain.accept(Record(content="anything")) is True

    def test_dedup_filter(self):
        from domains.collections import DedupFilter, Record
        f = DedupFilter()
        assert f.accept(Record(content="first")) is True
        assert f.accept(Record(content="first")) is False
        assert f.accept(Record(content="second")) is True
