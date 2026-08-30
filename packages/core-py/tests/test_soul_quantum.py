"""Tests for domains/soul/quantum.py — pure logic, no mocks."""

import math
import random

import pytest

from domains.soul.quantum import (
    HyperdimensionalProcessor,
    QuantumCognitiveEngine,
    QuantumParallelProcessor,
    QuantumState,
    TemporalReasoningEngine,
)


# ---------------------------------------------------------------------------
# QuantumState
# ---------------------------------------------------------------------------

class TestQuantumState:
    def test_default_amplitude_and_phase(self):
        qs = QuantumState()
        assert qs.amplitude == 1 + 0j
        assert qs.basis_state == ""
        assert qs.probability == pytest.approx(1.0)

    def test_custom_amplitude(self):
        qs = QuantumState(amplitude=1 + 1j, basis_state="a")
        assert qs.basis_state == "a"
        assert qs.probability == pytest.approx(2.0)

    def test_normalize_reduces_to_unit(self):
        qs = QuantumState(amplitude=3 + 4j)
        qs.normalize()
        assert abs(qs.amplitude) == pytest.approx(1.0)
        assert qs.probability == pytest.approx(1.0)

    def test_normalize_zero_amplitude_noop(self):
        qs = QuantumState(amplitude=0 + 0j)
        qs.normalize()
        assert qs.amplitude == 0 + 0j
        assert qs.probability == pytest.approx(0.0)

    def test_measure_returns_basis_state_when_probability_high(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="yes")
        # probability=1.0, so random.random() < 1.0 is always True
        assert qs.measure() == "yes"

    def test_measure_returns_empty_when_probability_zero(self):
        qs = QuantumState(amplitude=0 + 0j, basis_state="no")
        assert qs.measure() == ""


# ---------------------------------------------------------------------------
# QuantumCognitiveEngine
# ---------------------------------------------------------------------------

class TestQuantumCognitiveEngine:
    def test_superposition_count(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["a", "b", "c"])
        assert len(engine.superposition) == 3

    def test_superposition_empty(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition([])
        assert engine.superposition == []

    def test_superposition_amplitude_magnitude(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["x", "y"])
        expected_mag = 1.0 / math.sqrt(2)
        for qs in engine.superposition:
            assert abs(qs.amplitude) == pytest.approx(expected_mag)

    def test_interfere_returns_normalized_probs(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["cat", "dog", "bird"])
        results = engine.interfere()
        assert len(results) == 3
        total = sum(p for _, p in results)
        assert total == pytest.approx(1.0)

    def test_interfere_empty_superposition(self):
        engine = QuantumCognitiveEngine()
        assert engine.interfere() == []

    def test_measure_collapses_to_one(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["a", "b"])
        result = engine.measure()
        assert result in ("a", "b")

    def test_measure_empty_returns_empty(self):
        engine = QuantumCognitiveEngine()
        assert engine.measure() == ""

    def test_entangle_stores_pair(self):
        engine = QuantumCognitiveEngine()
        engine.entangle("x", "y")
        assert engine.entangled_pairs == [("x", "y")]

    def test_tunnel_no_pairs_returns_none(self):
        engine = QuantumCognitiveEngine()
        assert engine.tunnel("barrier") is None

    def test_tunnel_probability_respects_coherence(self):
        engine = QuantumCognitiveEngine(coherence=1.0)
        engine.entangle("a", "b")
        # With coherence=1.0, tunnel_prob=0.1. Run many times, expect some None.
        results = [engine.tunnel("wall") for _ in range(200)]
        assert None in results
        assert any(r is not None for r in results)

    def test_similarity_empty_strings(self):
        engine = QuantumCognitiveEngine()
        assert engine._similarity("", "") == 0.0

    def test_similarity_identical(self):
        engine = QuantumCognitiveEngine()
        assert engine._similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_similarity_disjoint(self):
        engine = QuantumCognitiveEngine()
        assert engine._similarity("a b", "c d") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# QuantumParallelProcessor
# ---------------------------------------------------------------------------

class TestQuantumParallelProcessor:
    def test_default_capacity(self):
        qpp = QuantumParallelProcessor()
        assert qpp.get_parallel_capacity() == 8

    def test_custom_capacity(self):
        qpp = QuantumParallelProcessor(num_streams=4)
        assert qpp.get_parallel_capacity() == 4

    def test_parallel_process_distributes(self):
        qpp = QuantumParallelProcessor(num_streams=3)
        inputs = ["a", "b", "c", "d"]
        results = qpp.parallel_process(inputs, lambda x: x.upper())
        assert results == ["A", "B", "C", "D"]
        assert len(qpp.streams[0]) == 2  # a, d
        assert len(qpp.streams[1]) == 1  # b
        assert len(qpp.streams[2]) == 1  # c

    def test_parallel_process_records_metadata(self):
        qpp = QuantumParallelProcessor(num_streams=4)
        qpp.parallel_process(["x", "y", "z"], lambda x: x)
        assert len(qpp.results) == 1
        assert qpp.results[0]["inputs"] == 3
        assert qpp.results[0]["streams_used"] == 3


# ---------------------------------------------------------------------------
# HyperdimensionalProcessor
# ---------------------------------------------------------------------------

class TestHyperdimensionalProcessor:
    def test_encode_caches(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("test")
        v2 = hdp.encode("test")
        assert v1 is v2
        assert len(v1) == 100

    def test_encode_binary_values(self):
        hdp = HyperdimensionalProcessor(dim=200)
        v = hdp.encode("sym")
        assert all(x in (-1, 1) for x in v)

    def test_encode_text_empty(self):
        hdp = HyperdimensionalProcessor(dim=50)
        assert hdp.encode_text("") == [0] * 50

    def test_encode_text_single_word(self):
        hdp = HyperdimensionalProcessor(dim=50)
        v = hdp.encode_text("hello")
        assert len(v) == 50
        assert all(x in (-1, 1) for x in v)

    def test_encode_text_multi_word(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode_text("hello world")
        assert len(v) == 100

    def test_bundle_empty(self):
        hdp = HyperdimensionalProcessor(dim=10)
        assert hdp.bundle([]) == [0] * 10

    def test_bundle_single(self):
        hdp = HyperdimensionalProcessor(dim=10)
        v = [1.0] * 10
        result = hdp.bundle([v])
        assert all(x in (-1.0, 1.0) for x in result)

    def test_bundle_multiple(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        result = hdp.bundle([v1, v2])
        assert len(result) == 100
        assert all(x in (-1.0, 1.0) for x in result)

    def test_bind_same_dim(self):
        hdp = HyperdimensionalProcessor(dim=50)
        v1 = hdp.encode("x")
        v2 = hdp.encode("y")
        result = hdp.bind(v1, v2)
        assert len(result) == 50

    def test_bind_different_dim_raises(self):
        hdp = HyperdimensionalProcessor(dim=50)
        v1 = [1.0] * 50
        v2 = [1.0] * 30
        with pytest.raises(ValueError):
            hdp.bind(v1, v2)

    def test_bind_inverse_property(self):
        """bind(v, v) == identity for binary vectors."""
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode("self")
        result = hdp.bind(v, v)
        # Element-wise: 1*1=1 or (-1)*(-1)=1, so all 1 → binarized to all 1.0
        assert all(x == 1.0 for x in result)

    def test_similarity_identical(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode("same")
        assert hdp.similarity(v, v) == pytest.approx(1.0)

    def test_similarity_random_low(self):
        hdp = HyperdimensionalProcessor(dim=1000)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        sim = hdp.similarity(v1, v2)
        assert -1.0 <= sim <= 1.0
        assert abs(sim) < 0.3  # random vectors should have low similarity

    def test_similarity_different_dim_raises(self):
        hdp = HyperdimensionalProcessor(dim=50)
        with pytest.raises(ValueError):
            hdp.similarity([1.0] * 50, [1.0] * 30)

    def test_unbind_is_bind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("u")
        v2 = hdp.encode("v")
        bound = hdp.bind(v1, v2)
        unbound = hdp.unbind(bound, v2)
        assert unbound == hdp.bind(bound, v2)

    def test_cleanup_removes_all_zero(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["zero"] = [0] * 10
        hdp.vectors["real"] = [1] * 10
        removed = hdp.cleanup()
        assert removed == 1
        assert "zero" not in hdp.vectors
        assert "real" in hdp.vectors


# ---------------------------------------------------------------------------
# TemporalReasoningEngine
# ---------------------------------------------------------------------------

class TestTemporalReasoningEngine:
    def test_default_depth(self):
        tre = TemporalReasoningEngine()
        assert tre.timeline_depth == 5
        assert len(tre.timelines) == 5

    def test_add_event_to_current_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"action": "speak"})
        assert len(tre.timelines[0]) == 1
        assert tre.timelines[0][0]["action"] == "speak"

    def test_add_event_to_specific_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"action": "move"}, timeline=2)
        assert len(tre.timelines[2]) == 1
        assert tre.timelines[2][0]["action"] == "move"

    def test_add_event_out_of_range_ignored(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"action": "lost"}, timeline=99)
        assert all(len(t) == 0 for t in tre.timelines)

    def test_branch_creates_new_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"action": "start"})
        new = tre.branch("if_a")
        assert new != 0
        assert tre.timelines[new] == tre.timelines[0]

    def test_branch_wraps_around(self):
        tre = TemporalReasoningEngine(timeline_depth=3)
        tre.switch_timeline(2)
        new = tre.branch("cond")
        assert new == 0  # (2+1)%3 == 0

    def test_switch_timeline_valid(self):
        tre = TemporalReasoningEngine()
        assert tre.switch_timeline(3) is True
        assert tre.current_timeline == 3

    def test_switch_timeline_invalid(self):
        tre = TemporalReasoningEngine()
        assert tre.switch_timeline(99) is False
        assert tre.current_timeline == 0

    def test_get_current_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"step": 1})
        tre.add_event({"step": 2})
        events = tre.get_current_events()
        assert len(events) == 2
        assert events[0]["step"] == 1

    def test_get_current_events_limit(self):
        tre = TemporalReasoningEngine()
        for i in range(5):
            tre.add_event({"step": i})
        events = tre.get_current_events(n=2)
        assert len(events) == 2

    def test_merge_timelines(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"from": "t0"})
        tre.branch("split")
        tre.switch_timeline(1)
        tre.add_event({"from": "t1"})
        merged = tre.merge_timelines(0, 1)
        # branch() copies timeline 0 into timeline 1, so timeline 1 has 2 events
        # total is 3 (t0 once + t1 once + copied t0)
        assert len(merged) == 3
        # Sorted by timestamp
        assert merged[0]["timestamp"] <= merged[-1]["timestamp"]
