"""Meaningful tests for QuantumState, QuantumCognitiveEngine, QuantumParallelProcessor, HyperdimensionalProcessor, HDMemoryStore."""

import random
import pytest
from domains.soul.quantum import (
    QuantumState, QuantumCognitiveEngine, QuantumParallelProcessor,
    HyperdimensionalProcessor,
)
from domains.soul.hd_memory import HDMemoryStore, HDMemoryItem


# ── QuantumState ───────────────────────────────────────────────────────

class TestQuantumState:
    def test_initial_state(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="test")
        assert qs.basis_state == "test"
        assert qs.probability == 1.0
        assert qs.phase == 0.0

    def test_normalize(self):
        qs = QuantumState(amplitude=3 + 4j)
        qs.normalize()
        assert abs(qs.amplitude) == pytest.approx(1.0)
        assert qs.probability == pytest.approx(1.0)

    def test_normalize_zero(self):
        qs = QuantumState(amplitude=0 + 0j)
        qs.normalize()  # Should not crash
        assert qs.amplitude == 0 + 0j

    def test_measure_high_probability(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="yes")
        # With probability 1.0, measure should always return basis_state
        results = [qs.measure() for _ in range(10)]
        assert all(r == "yes" for r in results)

    def test_measure_zero_probability(self):
        qs = QuantumState(amplitude=0 + 0j, basis_state="no")
        results = [qs.measure() for _ in range(10)]
        assert all(r == "" for r in results)


# ── QuantumCognitiveEngine ─────────────────────────────────────────────

class TestQuantumCognitiveEngine:
    def test_create_superposition(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["thought1", "thought2", "thought3"])
        assert len(qce.superposition) == 3
        # Amplitudes should have equal magnitude
        for state in qce.superposition:
            assert abs(abs(state.amplitude) - 1 / 3**0.5) < 0.01

    def test_create_superposition_empty(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition([])
        assert len(qce.superposition) == 0

    def test_interfere(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["cat", "dog", "cat"])
        results = qce.interfere()
        assert len(results) == 3
        # Probabilities should sum to ~1
        total = sum(p for _, p in results)
        assert total == pytest.approx(1.0)

    def test_measure_collapses(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["A", "B", "C"])
        result = qce.measure()
        assert result in ["A", "B", "C"]

    def test_measure_empty(self):
        qce = QuantumCognitiveEngine()
        assert qce.measure() == ""

    def test_entangle(self):
        qce = QuantumCognitiveEngine()
        qce.entangle("thought1", "thought2")
        assert len(qce.entangled_pairs) == 1
        assert qce.entangled_pairs[0] == ("thought1", "thought2")

    def test_tunnel_with_pairs(self):
        random.seed(42)
        qce = QuantumCognitiveEngine(coherence=1.0)
        qce.entangle("A", "B")
        # High coherence → high tunnel probability
        results = [qce.tunnel("barrier") for _ in range(100)]
        assert any(r is not None for r in results)

    def test_tunnel_without_pairs(self):
        qce = QuantumCognitiveEngine(coherence=1.0)
        assert qce.tunnel("barrier") is None

    def test_similarity(self):
        qce = QuantumCognitiveEngine()
        assert qce._similarity("hello world", "hello world") == 1.0
        assert qce._similarity("hello", "world") == 0.0
        assert qce._similarity("", "test") == 0.0

    def test_similarity_partial(self):
        qce = QuantumCognitiveEngine()
        sim = qce._similarity("hello world", "hello there")
        assert 0 < sim < 1


# ── QuantumParallelProcessor ───────────────────────────────────────────

class TestQuantumParallelProcessor:
    def test_parallel_process(self):
        qpp = QuantumParallelProcessor(num_streams=4)
        results = qpp.parallel_process([1, 2, 3, 4], lambda x: x * 2)
        assert results == [2, 4, 6, 8]

    def test_streams_distributed(self):
        qpp = QuantumParallelProcessor(num_streams=2)
        qpp.parallel_process([1, 2, 3, 4], lambda x: x)
        assert len(qpp.streams[0]) == 2
        assert len(qpp.streams[1]) == 2

    def test_results_tracked(self):
        qpp = QuantumParallelProcessor()
        qpp.parallel_process([1, 2], lambda x: x)
        assert len(qpp.results) == 1
        assert qpp.results[0]["inputs"] == 2

    def test_get_capacity(self):
        qpp = QuantumParallelProcessor(num_streams=8)
        assert qpp.get_parallel_capacity() == 8


# ── HyperdimensionalProcessor ──────────────────────────────────────────

class TestHyperdimensionalProcessor:
    def test_encode(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode("test")
        assert len(v) == 100
        assert all(x in [-1, 1] for x in v)

    def test_encode_deterministic(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("test")
        v2 = hdp.encode("test")
        assert v1 == v2

    def test_encode_different(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("hello")
        v2 = hdp.encode("world")
        assert v1 != v2

    def test_encode_text(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode_text("hello world")
        assert len(v) == 100

    def test_encode_text_empty(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode_text("")
        assert v == [0] * 100

    def test_bundle(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = [1.0] * 100
        v2 = [-1.0] * 100
        result = hdp.bundle([v1, v2])
        assert len(result) == 100
        assert all(x in [-1.0, 1.0] for x in result)

    def test_bundle_empty(self):
        hdp = HyperdimensionalProcessor(dim=100)
        assert hdp.bundle([]) == [0] * 100

    def test_bind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bound = hdp.bind(v1, v2)
        assert len(bound) == 100
        assert all(x in [-1.0, 1.0] for x in bound)

    def test_bind_different_length_raises(self):
        hdp = HyperdimensionalProcessor(dim=100)
        with pytest.raises(ValueError):
            hdp.bind([1, 2], [1, 2, 3])

    def test_similarity_same(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode("test")
        assert hdp.similarity(v, v) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        hdp = HyperdimensionalProcessor(dim=10000)
        v1 = hdp.encode("hello")
        v2 = hdp.encode("world")
        sim = hdp.similarity(v1, v2)
        # Random binary vectors should have similarity near 0
        assert abs(sim) < 0.1

    def test_unbind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bound = hdp.bind(v1, v2)
        unbound = hdp.unbind(bound, v2)
        # unbind(bound, v2) = bind(bound, v2) = element-wise multiply
        assert len(unbound) == 100

    def test_cleanup(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["zero"] = [0] * 10
        hdp.vectors["normal"] = [1] * 10
        removed = hdp.cleanup()
        assert removed == 1
        assert "zero" not in hdp.vectors
        assert "normal" in hdp.vectors


# ── HDMemoryStore ──────────────────────────────────────────────────────

class TestHDMemoryStore:
    def test_add(self):
        store = HDMemoryStore(dim=100, max_items=10)
        item_id = store.add("hello world", role="user")
        assert item_id.startswith("mem_")
        assert len(store.items) == 1

    def test_add_eviction(self):
        store = HDMemoryStore(dim=100, max_items=3)
        for i in range(5):
            store.add(f"message {i}")
        assert len(store.items) == 3

    def test_search(self):
        store = HDMemoryStore(dim=1000, max_items=10)
        store.add("Python programming language", role="user")
        store.add("Java programming language", role="assistant")
        results = store.search("Python", top_k=2)
        assert len(results) >= 1

    def test_search_empty(self):
        store = HDMemoryStore(dim=100)
        assert store.search("query") == []

    def test_search_role_filter(self):
        store = HDMemoryStore(dim=1000, max_items=10)
        store.add("hello", role="user")
        store.add("hi there", role="assistant")
        results = store.search("hello", role_filter="user")
        assert all(r[2] >= 0 for r in results)

    def test_get_context(self):
        store = HDMemoryStore(dim=1000, max_items=10)
        store.add("Python is great", role="user")
        store.add("Java is okay", role="assistant")
        ctx = store.get_context("Python", max_chars=500)
        assert isinstance(ctx, str)

    def test_get_context_empty(self):
        store = HDMemoryStore(dim=100)
        assert store.get_context("query") == ""
