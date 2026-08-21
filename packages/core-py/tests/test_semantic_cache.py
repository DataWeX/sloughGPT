"""Tests for domains/inference/semantic_cache.py."""

import time

import pytest

from domains.inference.semantic_cache import (
    CacheEntry,
    CachedSoulEngine,
    SemanticCache,
)


def make_cache(**kwargs):
    kwargs.setdefault("dim", 128)
    return SemanticCache(**kwargs)


class TestCacheEntry:
    def test_defaults(self):
        entry = CacheEntry(
            id="1", query="q", response="r", hypervector=[], metadata={}, timestamp=1.0
        )
        assert entry.hit_count == 0
        assert entry.last_accessed == 0

    def test_fields(self):
        entry = CacheEntry(
            id="abc",
            query="hello",
            response="hi there",
            hypervector=[1.0, -1.0],
            metadata={"model": "x"},
            timestamp=42.0,
            hit_count=3,
            last_accessed=43.0,
        )
        assert entry.id == "abc"
        assert entry.query == "hello"
        assert entry.response == "hi there"
        assert entry.hypervector == [1.0, -1.0]
        assert entry.metadata == {"model": "x"}
        assert entry.timestamp == 42.0
        assert entry.hit_count == 3
        assert entry.last_accessed == 43.0


class TestSemanticCacheBasics:
    def test_put_returns_id(self):
        cache = make_cache()
        entry_id = cache.put("what is the capital of France", "Paris")
        assert isinstance(entry_id, str)
        assert len(entry_id) == 16

    def test_put_is_deterministic(self):
        cache = make_cache()
        a = cache.put("hello world", "hi")
        b = cache.put("hello world", "hi")
        assert a == b

    def test_put_stores_entry(self):
        cache = make_cache()
        entry_id = cache.put("question", "answer", metadata={"k": "v"})
        entry = cache.entries[entry_id]
        assert entry.query == "question"
        assert entry.response == "answer"
        assert entry.metadata == {"k": "v"}
        assert entry.timestamp > 0

    def test_put_same_query_overwrites(self):
        cache = make_cache()
        first = cache.put("question", "answer one")
        second = cache.put("question", "answer two")
        assert first == second
        assert cache.get("question") == "answer two"

    def test_get_empty_cache_returns_none(self):
        cache = make_cache()
        assert cache.get("anything") is None

    def test_exact_query_hit(self):
        cache = make_cache()
        cache.put("what is the capital of France", "Paris")
        assert cache.get("what is the capital of France") == "Paris"

    def test_content_word_match_hit(self):
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        assert cache.get("capital of France is Paris") == "Paris"

    def test_paraphrase_miss(self):
        cache = make_cache()
        cache.put("what is the capital of France", "Paris")
        assert cache.get("how many moons does Jupiter have") is None

    def test_miss_increments_stats(self):
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        cache.get("capital of France is Paris")
        cache.get("unrelated topic entirely")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_hit_increments_hit_count(self):
        cache = make_cache()
        entry_id = cache.put("the capital of France is Paris", "Paris")
        cache.get("capital of France is Paris")
        assert cache.entries[entry_id].hit_count == 1

    def test_encode_query_dimension(self):
        cache = make_cache(dim=256)
        vector = cache.encode_query("hello world")
        assert len(vector) == 256

    def test_stats_snapshot(self):
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["entries"] == 1
        assert stats["max_entries"] == 1000
        assert stats["dimension"] == 128
        assert stats["similarity_threshold"] == 0.30
        assert stats["ttl_seconds"] == 3600
        assert stats["hit_rate"] == "0.0%"

    def test_reset_stats(self):
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        cache.get("capital of France is Paris")
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestSemanticCacheEviction:
    def test_ttl_expiration_on_get(self, monkeypatch):
        cache = make_cache(ttl_seconds=10)
        entry_id = cache.put("the capital of France is Paris", "Paris")
        monkeypatch.setattr(cache.entries[entry_id], "timestamp", time.time() - 100)
        assert cache.get("capital of France is Paris") is None

    def test_put_evicts_expired(self, monkeypatch):
        cache = make_cache(ttl_seconds=10)
        cache.put("the capital of France is Paris", "Paris")
        for entry in cache.entries.values():
            entry.timestamp = time.time() - 100
        cache.put("second question is stored", "answer")
        assert cache.get_stats()["expirations"] == 1
        assert cache.get_stats()["entries"] == 1

    def test_lru_eviction_when_full(self):
        cache = make_cache(max_entries=2)
        cache.put("first question to store", "a")
        cache.put("second question to store", "b")
        cache.put("third question to store", "c")
        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["evictions"] == 1

    def test_evict_lru_prefers_last_accessed(self):
        cache = make_cache(max_entries=1)
        first = cache.put("first question to store", "a")
        cache.entries[first].last_accessed = 100.0
        cache.put("second question to store", "b")
        assert first not in cache.entries


class TestSemanticCacheInvalidate:
    def test_invalidate_exact_match(self):
        cache = make_cache()
        entry_id = cache.put("the capital of France is Paris", "Paris")
        assert cache.invalidate("the capital of France is Paris") is True
        assert entry_id not in cache.entries

    def test_invalidate_miss(self):
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        assert cache.invalidate("completely unrelated thing") is False
        assert cache.get_stats()["entries"] == 1

    def test_invalidate_empty(self):
        assert make_cache().invalidate("nothing here") is False


class TestSemanticCacheClear:
    def test_clear_returns_count(self):
        cache = make_cache()
        cache.put("first question to store", "a")
        cache.put("second question to store", "b")
        assert cache.clear() == 2
        assert cache.get_stats()["entries"] == 0

    def test_clear_empty(self):
        assert make_cache().clear() == 0


class TestCachedSoulEngine:
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return f"response:{prompt}"

    def test_miss_generates_and_caches(self):
        engine = self.FakeEngine()
        cached = CachedSoulEngine(engine, cache=make_cache())
        assert cached.generate("the capital of France is Paris") == "response:the capital of France is Paris"
        assert len(engine.calls) == 1
        assert cached.generate("the capital of France is Paris") == "response:the capital of France is Paris"
        assert len(engine.calls) == 1

    def test_hit_skips_engine(self):
        engine = self.FakeEngine()
        cached = CachedSoulEngine(engine, cache=make_cache())
        cached.generate("the capital of France is Paris", temp=0.7)
        cached.generate("the capital of France is Paris")
        assert len(engine.calls) == 1
        assert engine.calls[0][1] == {"temp": 0.7}

    def test_cache_responses_disabled(self):
        engine = self.FakeEngine()
        cached = CachedSoulEngine(engine, cache=make_cache(), cache_responses=False)
        cached.generate("the capital of France is Paris")
        cached.generate("the capital of France is Paris")
        assert len(engine.calls) == 2

    def test_cache_stats(self):
        cached = CachedSoulEngine(self.FakeEngine(), cache=make_cache())
        stats = cached.get_cache_stats()
        assert stats["entries"] == 0
        assert stats["enabled"] is True

    def test_clear_cache(self):
        cached = CachedSoulEngine(self.FakeEngine(), cache=make_cache())
        cached.generate("the capital of France is Paris")
        assert cached.clear_cache() == 1
        assert cached.get_cache_stats()["entries"] == 0

    def test_invalidate(self):
        cached = CachedSoulEngine(self.FakeEngine(), cache=make_cache())
        cached.generate("the capital of France is Paris")
        assert cached.invalidate("the capital of France is Paris") is True


class FakeHyperdim:
    """Deterministic HD stand-in — encode is fixed, similarity is scripted."""

    def __init__(self, similarity=0.0):
        self._similarity = similarity

    def encode_text(self, text):
        return [1.0, 0.0]

    def similarity(self, a, b):
        return self._similarity


class TestSemanticCacheScoringBands:
    ENTRY = "alpha beta gamma delta"

    def _seed(self, cache):
        cache.put(self.ENTRY, "cached response")

    def test_medium_overlap_hd_validates_hit(self):
        cache = make_cache()
        cache._hyperdim = FakeHyperdim(similarity=0.9)
        self._seed(cache)
        assert cache.get("alpha beta gamma delta omega tau") == "cached response"

    def test_medium_overlap_low_hd_miss(self):
        cache = make_cache()
        cache._hyperdim = FakeHyperdim(similarity=0.4)
        self._seed(cache)
        assert cache.get("alpha beta gamma delta omega tau") is None

    def test_low_overlap_hd_validates_hit(self):
        cache = make_cache()
        cache._hyperdim = FakeHyperdim(similarity=0.9)
        self._seed(cache)
        assert cache.get("alpha beta gamma delta p1 p2 p3 p4 p5 p6") == "cached response"

    def test_low_overlap_low_hd_miss(self):
        cache = make_cache()
        cache._hyperdim = FakeHyperdim(similarity=0.5)
        self._seed(cache)
        assert cache.get("alpha beta gamma delta p1 p2 p3 p4 p5 p6") is None

    def test_evict_lru_empty_returns_false(self):
        assert make_cache()._evict_lru() is False


class TestSemanticCacheStatsDetail:
    def test_get_stats_hit_rate_with_hits(self):
        """hit_rate should be non-zero after a hit."""
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        cache.get("the capital of France is Paris")  # hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] != "0.0%"

    def test_get_stats_hit_rate_zero_total(self):
        """hit_rate should be 0.0% when no requests have been made."""
        cache = make_cache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == "0.0%"

    def test_reset_stats_then_recount(self):
        """After reset, hits/misses should be 0 but new requests should count."""
        cache = make_cache()
        cache.put("the capital of France is Paris", "Paris")
        cache.get("the capital of France is Paris")
        cache.reset_stats()
        cache.get("unrelated topic entirely")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_concurrent_put_get(self):
        """Multiple concurrent puts and gets should not crash."""
        import asyncio

        cache = make_cache()
        cache._hyperdim = FakeHyperdim(similarity=0.0)

        async def scenario():
            async def putter(i):
                cache.put(f"question number {i} about topic", f"answer {i}")

            async def getter(i):
                cache.get(f"question number {i} about topic")

            await asyncio.gather(*[putter(i) for i in range(5)])
            await asyncio.gather(*[getter(i) for i in range(5)])

        asyncio.run(scenario())
        assert len(cache.entries) == 5
