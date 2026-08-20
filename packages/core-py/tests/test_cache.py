"""Tests for domains.infrastructure.cache — CacheEntry and CacheManager init."""

import time
import pytest
from domains.infrastructure.cache import CacheEntry, CacheManager


class TestCacheEntry:
    def test_fields(self):
        now = time.time()
        entry = CacheEntry(
            key="k", value="v", ttl=60,
            created_at=now, accessed_count=1, last_accessed=now,
        )
        assert entry.key == "k"
        assert entry.value == "v"
        assert entry.ttl == 60
        assert entry.accessed_count == 1

    def test_none_ttl(self):
        entry = CacheEntry(
            key="k", value=42, ttl=None,
            created_at=0.0, accessed_count=0, last_accessed=0.0,
        )
        assert entry.ttl is None

    def test_value_types(self):
        entry = CacheEntry(
            key="k", value={"a": [1, 2]}, ttl=10,
            created_at=0.0, accessed_count=0, last_accessed=0.0,
        )
        assert entry.value["a"] == [1, 2]


class TestCacheManager:
    def setup_method(self):
        self.mgr = CacheManager()

    def test_defaults(self):
        assert self.mgr.max_size == 10000
        assert self.mgr.default_ttl == 3600
        assert self.mgr.cache == {}

    def test_stats_init(self):
        assert self.mgr.stats["hits"] == 0
        assert self.mgr.stats["misses"] == 0
        assert self.mgr.stats["sets"] == 0
        assert self.mgr.stats["evictions"] == 0

    def test_is_not_initialized(self):
        assert self.mgr.is_initialized is False

    def test_component_name(self):
        assert self.mgr.component_name == "cache_manager"
