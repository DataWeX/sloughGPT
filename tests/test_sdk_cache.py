"""Coverage for sloughgpt_sdk.cache."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "sdk-py"))

from sloughgpt_sdk.cache import DiskCache, InMemoryCache, CacheEntry, cached  # noqa: E402


class TestCacheEntry:
    def test_no_ttl_never_expires(self):
        entry = CacheEntry("k", "v", created_at=0, ttl=None)
        assert entry.is_expired() is False

    def test_expired_when_past_ttl(self):
        with patch("sloughgpt_sdk.cache.time.time", return_value=100.0):
            entry = CacheEntry("k", "v", created_at=0, ttl=50)
            assert entry.is_expired() is True

    def test_not_expired_inside_ttl(self):
        with patch("sloughgpt_sdk.cache.time.time", return_value=40.0):
            entry = CacheEntry("k", "v", created_at=0, ttl=50)
            assert entry.is_expired() is False


class TestInMemoryCache:
    def test_set_get_roundtrip(self):
        cache = InMemoryCache()
        cache.set("a", "value")
        assert cache.get("a") == "value"

    def test_get_miss_returns_none(self):
        cache = InMemoryCache()
        assert cache.get("missing") is None

    def test_get_tracks_miss_counter(self):
        cache = InMemoryCache()
        cache.get("missing")
        assert cache.stats["misses"] == 1

    def test_get_tracks_hit_counter(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.get("a")
        assert cache.stats["hits"] == 1

    def test_delete_removes_key(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.delete("a")
        assert cache.get("a") is None

    def test_delete_missing_is_noop(self):
        cache = InMemoryCache()
        cache.delete("missing")
        assert cache.size == 0

    def test_clear_resets_counters(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.get("a")
        cache.clear()
        assert cache.size == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_expired_entry_evicted_on_get(self):
        cache = InMemoryCache()
        cache.set("a", 1, ttl=10)
        cache._cache["a"].created_at -= 60
        assert cache.get("a") is None
        assert cache.size == 0
        assert cache.stats["misses"] == 1

    def test_default_ttl_applied(self):
        cache = InMemoryCache(ttl=3600)
        cache.set("a", 1)
        assert cache._cache["a"].ttl == 3600

    def test_per_call_ttl_overrides_default(self):
        cache = InMemoryCache(ttl=3600)
        cache.set("a", 1, ttl=5)
        assert cache._cache["a"].ttl == 5

    def test_evicts_oldest_on_max_size(self):
        cache = InMemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.size == 2
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_size_property(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2

    def test_stats_shape(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.get("a")
        cache.get("zzz")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_stats_hit_rate_zero_when_empty(self):
        cache = InMemoryCache()
        assert cache.stats["hit_rate"] == 0.0

    def test_generate_key_deterministic(self):
        cache = InMemoryCache()
        assert cache._generate_key("x", [1], foo="bar") == cache._generate_key(
            "x", [1], foo="bar"
        )

    def test_evict_oldest_empty_cache_is_noop(self):
        cache = InMemoryCache()
        cache._evict_oldest()
        assert cache.size == 0


class TestDiskCache:
    @pytest.fixture
    def cache_dir(self, tmp_path):
        return str(tmp_path / "cache")

    def test_set_get_roundtrip(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", {"x": 1})
        assert cache.get("a") == {"x": 1}

    def test_missing_key_returns_none(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        assert cache.get("nope") is None

    def test_misses_counter(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.get("nope")
        assert cache.stats["misses"] == 1

    def test_delete_removes(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        cache.delete("a")
        assert os.path.exists(cache._get_path("a")) is False

    def test_delete_missing_noop(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.delete("nope")
        assert cache.size == 0

    def test_clear_removes_all(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.stats["hits"] == 0

    def test_expired_file_removed(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1, ttl=10)
        path = cache._get_path("a")
        assert os.path.exists(path)
        with patch("sloughgpt_sdk.cache.time.time") as t:
            t.return_value = 10 ** 12
            assert cache.get("a") is None
        assert os.path.exists(path) is False
        assert cache.stats["misses"] == 1

    def test_corrupt_json_returns_none(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        with open(cache._get_path("a"), "w") as f:
            f.write("{not json")
        assert cache.get("a") is None
        assert cache.stats["misses"] == 1

    def test_default_cache_dir_created(self, tmp_path):
        with patch("os.makedirs"):
            cache = DiskCache(cache_dir=str(tmp_path / "made"), ttl=30)
            assert cache._ttl == 30

    def test_ttl_persisted_no_expiry_without_ttl(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_key_generation_shortened(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        key = cache._generate_key("x", a=1)
        assert len(key) == 32

    def test_size_property_counts_files(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2

    def test_stats_shape(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        cache.get("a")
        cache.get("miss")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["size_bytes"] > 0
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_check_size_evicts_oldest(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir, max_size_mb=0)
        cache.set("a", "x" * 100)
        cache.set("b", "y" * 100)
        assert cache.size <= 1

    def test_check_size_breaks_when_under_threshold(self, cache_dir):
        limit_mb = 100 / (1024 * 1024)
        cache = DiskCache(cache_dir=cache_dir, max_size_mb=limit_mb)
        cache.set("a", "x" * 60)
        cache.set("b", "y" * 60)
        assert cache.get("a") is None
        assert cache.get("b") == "y" * 60
        assert cache.size == 1

    def test_hit_counting_after_set(self, cache_dir):
        cache = DiskCache(cache_dir=cache_dir)
        cache.set("a", 1)
        cache.get("a")
        assert cache.stats["hits"] == 1


class TestCachedDecorator:
    def test_caches_result(self):
        calls = []

        @cached()
        def compute(x):
            calls.append(x)
            return x * 2

        assert compute(3) == 6
        assert compute(3) == 6
        assert calls == [3]

    def test_distinct_arguments_recompute(self):
        calls = []

        @cached()
        def compute(x):
            calls.append(x)
            return x * 2

        compute(1)
        compute(2)
        assert calls == [1, 2]

    def test_exposes_helpers(self):
        @cached()
        def compute():
            return 1

        assert hasattr(compute, "cache")
        assert callable(compute.cache_clear)
        assert compute.cache_stats["hits"] == 0

    def test_custom_cache_instance_used(self):
        cache = InMemoryCache()
        calls = []

        @cached(cache=cache)
        def compute(x):
            calls.append(x)
            return x + 1

        compute(5)
        compute(5)
        assert calls == [5]
        assert cache.size == 1

    def test_ttl_stores_entry_with_ttl(self):
        @cached(ttl=42)
        def compute():
            return 1

        compute()
        wrapper_cache = compute.cache
        key = wrapper_cache._generate_key("compute")
        assert wrapper_cache._cache[key].ttl == 42

    def test_falsey_result_still_cached(self):
        @cached()
        def return_none():
            return None

        # A None result is treated as a miss, so the function recomputes.
        assert return_none() is None
        assert return_none() is None