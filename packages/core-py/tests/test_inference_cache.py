"""Tests for domains.inference.semantic_cache — CacheEntry, SemanticCache, CachedSoulEngine;
domains.inference.forward_pass — ForwardPassResult, timed_forward, ForwardPassable."""

import time
import numpy as np
import pytest

from domains.inference.semantic_cache import CacheEntry, SemanticCache, CachedSoulEngine
from domains.inference.forward_pass import ForwardPassResult, timed_forward


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_fields(self):
        ce = CacheEntry(id="c1", query="hello", response="world", hypervector=[0.1, 0.2], metadata={}, timestamp=1.0)
        assert ce.id == "c1"
        assert ce.query == "hello"
        assert ce.response == "world"
        assert ce.hit_count == 0

    def test_defaults(self):
        ce = CacheEntry(id="c1", query="q", response="r", hypervector=[], metadata={}, timestamp=0.0)
        assert ce.last_accessed == 0

    def test_metadata_is_mutable(self):
        m = {"key": "val"}
        ce = CacheEntry(id="x", query="q", response="r", hypervector=[], metadata=m, timestamp=0.0)
        ce.metadata["new_key"] = 42
        assert ce.metadata["new_key"] == 42

    def test_hypervector_stores_floats(self):
        hv = [float(i) for i in range(10)]
        ce = CacheEntry(id="x", query="q", response="r", hypervector=hv, metadata={}, timestamp=0.0)
        assert len(ce.hypervector) == 10
        assert ce.hypervector[5] == 5.0

    def test_hit_count_starts_zero(self):
        ce = CacheEntry(id="x", query="q", response="r", hypervector=[], metadata={}, timestamp=1.0)
        assert ce.hit_count == 0
        assert ce.last_accessed == 0

    def test_timestamp_precision(self):
        ts = time.time()
        ce = CacheEntry(id="x", query="q", response="r", hypervector=[], metadata={}, timestamp=ts)
        assert abs(ce.timestamp - ts) < 1e-6

    def test_empty_string_fields(self):
        ce = CacheEntry(id="", query="", response="", hypervector=[], metadata={}, timestamp=0.0)
        assert ce.id == ""
        assert ce.query == ""
        assert ce.response == ""

    def test_large_hypervector(self):
        hv = [0.001] * 10000
        ce = CacheEntry(id="big", query="q", response="r", hypervector=hv, metadata={}, timestamp=0.0)
        assert len(ce.hypervector) == 10000

    def test_multiple_cache_entries_independent(self):
        ce1 = CacheEntry(id="1", query="q1", response="r1", hypervector=[1.0], metadata={}, timestamp=1.0)
        ce2 = CacheEntry(id="2", query="q2", response="r2", hypervector=[2.0], metadata={}, timestamp=2.0)
        ce1.hit_count = 5
        assert ce2.hit_count == 0

    def test_cache_entry_equality(self):
        ce1 = CacheEntry(id="a", query="q", response="r", hypervector=[], metadata={}, timestamp=1.0)
        ce2 = CacheEntry(id="a", query="q", response="r", hypervector=[], metadata={}, timestamp=1.0)
        assert ce1.id == ce2.id
        assert ce1.query == ce2.query


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class TestSemanticCache:
    def test_init(self):
        sc = SemanticCache(dim=100, max_entries=50)
        assert sc.dim == 100
        assert sc.max_entries == 50

    def test_init_defaults(self):
        sc = SemanticCache()
        assert sc.dim == 10000
        assert sc.max_entries == 1000
        assert sc.similarity_threshold == 0.30
        assert sc.ttl_seconds == 3600

    def test_set_and_get(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("hello world", "response1")
        result = sc.get("hello world")
        assert result is not None
        assert result == "response1"

    def test_miss(self):
        sc = SemanticCache(dim=100, max_entries=10)
        result = sc.get("nonexistent query xyz")
        assert result is None

    def test_eviction(self):
        sc = SemanticCache(dim=100, max_entries=3)
        for i in range(5):
            sc.put(f"query {i}", f"response {i}")
        assert len(sc.entries) <= 3

    def test_stats(self):
        sc = SemanticCache(dim=100, max_entries=10)
        stats = sc.get_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_stats_initial(self):
        sc = SemanticCache(dim=100, max_entries=10)
        stats = sc.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0
        assert stats["entries"] == 0
        assert stats["enabled"] is True

    def test_clear(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("q1", "r1")
        sc.put("q2", "r2")
        count = sc.clear()
        assert count == 2
        assert len(sc.entries) == 0

    def test_clear_empty(self):
        sc = SemanticCache(dim=100, max_entries=10)
        count = sc.clear()
        assert count == 0

    def test_reset_stats(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.get("nope")
        sc.reset_stats()
        stats = sc.get_stats()
        assert stats["misses"] == 0
        assert stats["hits"] == 0

    def test_put_returns_id(self):
        sc = SemanticCache(dim=100, max_entries=10)
        entry_id = sc.put("query", "response")
        assert isinstance(entry_id, str)
        assert len(entry_id) == 16

    def test_put_same_query_overwrites(self):
        sc = SemanticCache(dim=100, max_entries=10)
        id1 = sc.put("hello", "resp1")
        id2 = sc.put("hello", "resp2")
        assert id1 == id2

    def test_multiple_entries(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("what is python", "a programming language")
        sc.put("what is java", "another language")
        assert len(sc.entries) == 2

    def test_get_increments_hit_count(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("unique query for hit count", "response")
        sc.get("unique query for hit count")
        stats = sc.get_stats()
        assert stats["hits"] == 1

    def test_get_increments_miss_count(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.get("totally missing query")
        stats = sc.get_stats()
        assert stats["misses"] == 1

    def test_stats_hit_rate_string(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("q", "r")
        sc.get("q")
        sc.get("nope")
        stats = sc.get_stats()
        assert "%" in stats["hit_rate"]

    def test_max_entries_zero_still_stores(self):
        sc = SemanticCache(dim=100, max_entries=0)
        sc.put("q", "r")
        assert len(sc.entries) == 1

    def test_ttl_expiration(self):
        sc = SemanticCache(dim=100, max_entries=10, ttl_seconds=0)
        sc.put("expiring query", "response")
        result = sc.get("expiring query")
        assert result is None

    def test_stats_max_entries(self):
        sc = SemanticCache(dim=500, max_entries=42)
        stats = sc.get_stats()
        assert stats["max_entries"] == 42
        assert stats["dimension"] == 500

    def test_similarity_threshold_stored(self):
        sc = SemanticCache(dim=100, max_entries=10, similarity_threshold=0.75)
        assert sc.similarity_threshold == 0.75

    def test_entries_dict_empty_initially(self):
        sc = SemanticCache(dim=100, max_entries=10)
        assert isinstance(sc.entries, dict)
        assert len(sc.entries) == 0

    def test_eviction_increments_counter(self):
        sc = SemanticCache(dim=100, max_entries=2)
        sc.put("q1", "r1")
        sc.put("q2", "r2")
        sc.put("q3", "r3")
        stats = sc.get_stats()
        assert stats["evictions"] >= 1

    def test_put_with_metadata(self):
        sc = SemanticCache(dim=100, max_entries=10)
        entry_id = sc.put("q", "r", metadata={"model": "test"})
        assert entry_id in sc.entries
        assert sc.entries[entry_id].metadata == {"model": "test"}

    def test_put_without_metadata_default(self):
        sc = SemanticCache(dim=100, max_entries=10)
        entry_id = sc.put("q", "r")
        assert sc.entries[entry_id].metadata == {}


# ---------------------------------------------------------------------------
# CachedSoulEngine
# ---------------------------------------------------------------------------

class TestCachedSoulEngine:
    def _make_engine(self, response="generated"):
        class FakeEngine:
            def generate(self, prompt, **kwargs):
                return response
        return FakeEngine()

    def test_init_default_cache(self):
        cse = CachedSoulEngine(self._make_engine())
        assert cse.cache is not None
        assert cse.cache_responses is True

    def test_init_custom_cache(self):
        custom = SemanticCache(dim=50, max_entries=5)
        cse = CachedSoulEngine(self._make_engine(), cache=custom)
        assert cse.cache is custom

    def test_generate_caches_response(self):
        cse = CachedSoulEngine(self._make_engine("hello"))
        result = cse.generate("test prompt")
        assert result == "hello"
        cached = cse.cache.get("test prompt")
        assert cached == "hello"

    def test_generate_returns_cached_on_second_call(self):
        cse = CachedSoulEngine(self._make_engine("new response"))
        cse.generate("cached query")
        cse._engine = self._make_engine("should not be called")
        result = cse.generate("cached query")
        assert result == "new response"

    def test_get_cache_stats(self):
        cse = CachedSoulEngine(self._make_engine())
        stats = cse.get_cache_stats()
        assert "entries" in stats

    def test_clear_cache(self):
        cse = CachedSoulEngine(self._make_engine("r"))
        cse.generate("q")
        cleared = cse.clear_cache()
        assert cleared == 1
        assert len(cse.cache.entries) == 0

    def test_invalidate(self):
        cse = CachedSoulEngine(self._make_engine("r"))
        cse.generate("q to invalidate")
        result = cse.invalidate("q to invalidate")
        assert result is True

    def test_cache_disabled(self):
        cse = CachedSoulEngine(self._make_engine("r"), cache_responses=False)
        cse.generate("no cache me")
        assert len(cse.cache.entries) == 0

    def test_generate_passes_kwargs(self):
        received = {}
        class CaptureEngine:
            def generate(self, prompt, **kwargs):
                received.update(kwargs)
                return "ok"
        cse = CachedSoulEngine(CaptureEngine())
        cse.generate("prompt", temperature=0.5, max_tokens=100)
        assert received["temperature"] == 0.5
        assert received["max_tokens"] == 100

    def test_empty_response_not_cached(self):
        class EmptyEngine:
            def generate(self, prompt, **kwargs):
                return ""
        cse = CachedSoulEngine(EmptyEngine(), cache_responses=True)
        cse.generate("q")
        assert len(cse.cache.entries) == 0

    def test_invalidate_nonexistent_returns_false(self):
        cse = CachedSoulEngine(self._make_engine())
        assert cse.invalidate("nonexistent") is False

    def test_engine_generate_called(self):
        class CountEngine:
            call_count = 0
            def generate(self, prompt, **kwargs):
                CountEngine.call_count += 1
                return "r"
        eng = CountEngine()
        cse = CachedSoulEngine(eng)
        cse.generate("q1")
        cse.generate("q1")  # cached
        assert eng.call_count == 1


# ---------------------------------------------------------------------------
# ForwardPassResult
# ---------------------------------------------------------------------------

class TestForwardPassResult:
    def test_fields(self):
        logits = np.random.randn(1, 10, 100).astype(np.float32)
        fpr = ForwardPassResult(logits=logits, forward_time_ms=1.5, model_name="gpt2", engine="numpy")
        assert fpr.model_name == "gpt2"
        assert fpr.engine == "numpy"
        assert fpr.forward_time_ms == 1.5

    def test_shape(self):
        logits = np.random.randn(2, 5, 50).astype(np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == [2, 5, 50]

    def test_defaults(self):
        logits = np.zeros((1, 1, 10), dtype=np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.forward_time_ms == 0.0
        assert fpr.cached_tokens == 0
        assert fpr.engine == "unknown"

    def test_shape_1d(self):
        logits = np.zeros((10,), dtype=np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == [10]

    def test_shape_4d(self):
        logits = np.zeros((1, 3, 4, 5), dtype=np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == [1, 3, 4, 5]

    def test_logits_reference(self):
        arr = np.ones((1, 2, 3), dtype=np.float32)
        fpr = ForwardPassResult(logits=arr)
        arr[0, 0, 0] = 999.0
        assert fpr.logits[0, 0, 0] == 999.0

    def test_model_name_default(self):
        fpr = ForwardPassResult(logits=np.zeros((1, 1, 1)))
        assert fpr.model_name == ""

    def test_cached_tokens_custom(self):
        fpr = ForwardPassResult(logits=np.zeros((1, 1, 1)), cached_tokens=512)
        assert fpr.cached_tokens == 512

    def test_forward_time_ms_negative(self):
        fpr = ForwardPassResult(logits=np.zeros((1, 1, 1)), forward_time_ms=-1.0)
        assert fpr.forward_time_ms == -1.0

    def test_engine_override(self):
        fpr = ForwardPassResult(logits=np.zeros((1, 1, 1)), engine="c")
        assert fpr.engine == "c"

    def test_large_batch(self):
        logits = np.zeros((64, 128, 50000), dtype=np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == [64, 128, 50000]

    def test_zero_shape(self):
        logits = np.zeros((), dtype=np.float32)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == []

    def test_complex_dtype_still_works(self):
        logits = np.zeros((1, 1, 1), dtype=np.float64)
        fpr = ForwardPassResult(logits=logits)
        assert fpr.shape == [1, 1, 1]

    def test_all_fields_settable(self):
        logits = np.zeros((1, 1, 1), dtype=np.float32)
        fpr = ForwardPassResult(
            logits=logits,
            forward_time_ms=42.0,
            model_name="mymodel",
            cached_tokens=100,
            engine="c",
        )
        assert fpr.forward_time_ms == 42.0
        assert fpr.model_name == "mymodel"
        assert fpr.cached_tokens == 100
        assert fpr.engine == "c"


# ---------------------------------------------------------------------------
# timed_forward
# ---------------------------------------------------------------------------

class TestTimedForward:
    def test_timing(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 10)))
        result = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64), model_name="test")
        assert result.forward_time_ms >= 0.0
        assert result.model_name == "test"

    def test_sets_model_name(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 10)))
        result = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64), model_name="my_model")
        assert result.model_name == "my_model"

    def test_empty_model_name(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 10)))
        result = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64))
        assert result.model_name == ""

    def test_timing_is_nonnegative(self):
        class SlowModel:
            def forward_pass(self, input_ids):
                import time
                time.sleep(0.001)
                return ForwardPassResult(logits=np.zeros((1, 1, 5)))
        result = timed_forward(SlowModel(), np.zeros((1, 1), dtype=np.int64))
        assert result.forward_time_ms >= 0.0

    def test_preserves_logits(self):
        arr = np.ones((1, 2, 3), dtype=np.float32)
        class IdModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=arr)
        result = timed_forward(IdModel(), np.zeros((1, 1), dtype=np.int64))
        np.testing.assert_array_equal(result.logits, arr)

    def test_forward_pass_raises_propagates(self):
        class BadModel:
            def forward_pass(self, input_ids):
                raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            timed_forward(BadModel(), np.zeros((1, 1), dtype=np.int64))

    def test_overwrites_forward_time_ms(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 1)), forward_time_ms=999.0)
        result = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64))
        assert result.forward_time_ms != 999.0
        assert result.forward_time_ms >= 0.0

    def test_overwrites_model_name(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 1)), model_name="old")
        result = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64), model_name="new")
        assert result.model_name == "new"

    def test_multiple_calls_consistent(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, 1, 10)))
        r1 = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64))
        r2 = timed_forward(FakeModel(), np.zeros((1, 1), dtype=np.int64))
        assert r1.shape == r2.shape

    def test_large_input(self):
        class FakeModel:
            def forward_pass(self, input_ids):
                return ForwardPassResult(logits=np.zeros((1, input_ids.shape[1], 100)))
        ids = np.zeros((1, 512), dtype=np.int64)
        result = timed_forward(FakeModel(), ids, model_name="big")
        assert result.shape == [1, 512, 100]


# ---------------------------------------------------------------------------
# SemanticCache — advanced edge cases
# ---------------------------------------------------------------------------

class TestSemanticCacheEdgeCases:
    def test_get_empty_cache_misses(self):
        sc = SemanticCache(dim=100, max_entries=10)
        result = sc.get("anything")
        assert result is None
        stats = sc.get_stats()
        assert stats["misses"] == 1

    def test_put_many_then_get(self):
        sc = SemanticCache(dim=100, max_entries=100)
        for i in range(20):
            sc.put(f"unique query {i}", f"response {i}")
        assert len(sc.entries) == 20

    def test_get_after_ttl_expiry(self):
        sc = SemanticCache(dim=100, max_entries=10, ttl_seconds=0.001)
        sc.put("will expire", "response")
        time.sleep(0.01)
        result = sc.get("will expire")
        assert result is None

    def test_stats_after_mixed_operations(self):
        sc = SemanticCache(dim=100, max_entries=5)
        sc.put("q1", "r1")
        sc.put("q2", "r2")
        sc.get("q1")
        sc.get("missing")
        stats = sc.get_stats()
        assert stats["entries"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_zero_max_entries_still_stores(self):
        sc = SemanticCache(dim=100, max_entries=0)
        sc.put("q", "r")
        assert len(sc.entries) == 1

    def test_single_entry_get_hit(self):
        sc = SemanticCache(dim=100, max_entries=10)
        sc.put("exact match query test", "response")
        result = sc.get("exact match query test")
        assert result == "response"

    def test_ttl_configurable(self):
        sc = SemanticCache(dim=100, max_entries=10, ttl_seconds=7200)
        assert sc.ttl_seconds == 7200

    def test_similarity_threshold_configurable(self):
        sc = SemanticCache(dim=100, max_entries=10, similarity_threshold=0.9)
        assert sc.similarity_threshold == 0.9

    def test_clear_resets_entries(self):
        sc = SemanticCache(dim=100, max_entries=10)
        for i in range(5):
            sc.put(f"q{i}", f"r{i}")
        sc.clear()
        assert len(sc.entries) == 0
        result = sc.get("q0")
        assert result is None
