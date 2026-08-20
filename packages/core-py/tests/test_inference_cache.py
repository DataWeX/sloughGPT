"""Tests for domains.inference.semantic_cache — CacheEntry, SemanticCache; domains.inference.forward_pass — ForwardPassResult."""

import numpy as np
from domains.inference.semantic_cache import CacheEntry, SemanticCache
from domains.inference.forward_pass import ForwardPassResult


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


class TestSemanticCache:
    def test_init(self):
        sc = SemanticCache(dim=100, max_entries=50)
        assert sc.dim == 100
        assert sc.max_entries == 50

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
