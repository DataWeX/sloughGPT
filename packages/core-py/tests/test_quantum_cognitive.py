"""Tests for the quantum cognitive module (superposition, HD computing, temporal reasoning)."""

from __future__ import annotations

import math
import random
import pytest

from domains.soul.quantum import (
    QuantumState,
    QuantumCognitiveEngine,
    QuantumParallelProcessor,
    HyperdimensionalProcessor,
    TemporalReasoningEngine,
)


# ── QuantumState ─────────────────────────────────────────────────────────────


class TestQuantumState:
    def test_default_amplitude_and_basis(self):
        qs = QuantumState()
        assert qs.amplitude == 1 + 0j
        assert qs.basis_state == ""

    def test_normalize_sets_amplitude_to_unit_length(self):
        qs = QuantumState(amplitude=3 + 4j, basis_state="test")
        qs.normalize()
        assert abs(qs.amplitude) == pytest.approx(1.0)

    def test_normalize_preserves_direction(self):
        qs = QuantumState(amplitude=2 + 0j, basis_state="x")
        qs.normalize()
        assert qs.amplitude == pytest.approx(1 + 0j)
        qs2 = QuantumState(amplitude=-6 + 0j, basis_state="y")
        qs2.normalize()
        assert qs2.amplitude == pytest.approx(-1 + 0j)

    def test_normalize_zero_amplitude_no_crash(self):
        qs = QuantumState(amplitude=0j, basis_state="z")
        qs.normalize()
        assert qs.amplitude == 0j
        assert qs.probability == pytest.approx(0.0)

    def test_normalize_updates_probability(self):
        qs = QuantumState(amplitude=3 + 4j, basis_state="p")
        qs.normalize()
        assert qs.probability == pytest.approx(1.0)

    def test_measure_returns_basis_state_when_probability_one(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="yes")
        qs.normalize()
        results = {qs.measure() for _ in range(50)}
        assert results == {"yes"}

    def test_measure_returns_empty_when_probability_zero(self):
        qs = QuantumState(amplitude=0j, basis_state="no")
        qs.normalize()
        results = {qs.measure() for _ in range(50)}
        assert results == {""}

    def test_measure_returns_basis_state_or_empty_for_partial(self):
        qs = QuantumState(amplitude=0.5 + 0j, basis_state="maybe")
        qs.normalize()
        results = {qs.measure() for _ in range(200)}
        assert results.issubset({"maybe", ""})

    def test_probability_equals_amplitude_squared(self):
        qs = QuantumState(amplitude=2 + 0j)
        assert qs.probability == pytest.approx(4.0)

    def test_phase_is_correct(self):
        qs = QuantumState(amplitude=0 + 1j)
        assert qs.phase == pytest.approx(math.pi / 2)


# ── QuantumCognitiveEngine ───────────────────────────────────────────────────


class TestQuantumCognitiveEngine:
    def test_create_superposition_count(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["a", "b", "c"])
        assert len(engine.superposition) == 3

    def test_create_superposition_empty(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition([])
        assert len(engine.superposition) == 0

    def test_create_superposition_amplitude_magnitude(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["x", "y", "z"])
        for state in engine.superposition:
            assert abs(state.amplitude) == pytest.approx(1.0 / math.sqrt(3), abs=0.01)

    def test_create_superposition_basis_states(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["alpha", "beta"])
        bases = [s.basis_state for s in engine.superposition]
        assert "alpha" in bases
        assert "beta" in bases

    def test_interfere_returns_probabilities_summing_to_one(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["a", "b", "c"])
        results = engine.interfere()
        total = sum(prob for _, prob in results)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_interfere_returns_tuple_list(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["x", "y"])
        results = engine.interfere()
        assert len(results) == 2
        for name, prob in results:
            assert isinstance(name, str)
            assert isinstance(prob, float)
            assert prob >= 0

    def test_interfere_empty_superposition(self):
        engine = QuantumCognitiveEngine()
        results = engine.interfere()
        assert results == []

    def test_measure_returns_string(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["thought1", "thought2", "thought3"])
        result = engine.measure()
        assert isinstance(result, str)
        assert result in ["thought1", "thought2", "thought3"]

    def test_measure_empty_superposition_returns_empty(self):
        engine = QuantumCognitiveEngine()
        assert engine.measure() == ""

    def test_entangle_stores_pair(self):
        engine = QuantumCognitiveEngine()
        engine.entangle("A", "B")
        assert ("A", "B") in engine.entangled_pairs

    def test_entangle_multiple_pairs(self):
        engine = QuantumCognitiveEngine()
        engine.entangle("a", "b")
        engine.entangle("c", "d")
        assert len(engine.entangled_pairs) == 2

    def test_tunnel_returns_insight_or_none(self):
        engine = QuantumCognitiveEngine(coherence=1.0)
        engine.entangle("x", "y")
        results = {engine.tunnel("barrier") for _ in range(200)}
        for r in results:
            assert r is None or isinstance(r, str)

    def test_tunnel_with_no_entanglement_returns_none(self):
        engine = QuantumCognitiveEngine(coherence=0.0)
        engine.create_superposition(["a"])
        results = {engine.tunnel("barrier") for _ in range(50)}
        assert results == {None}

    def test_tunnel_with_high_coherence_may_succeed(self):
        engine = QuantumCognitiveEngine(coherence=1.0)
        engine.entangle("alpha", "beta")
        results = {engine.tunnel("wall") for _ in range(500)}
        assert None in results
        succeeded = [r for r in results if r is not None]
        assert len(succeeded) > 0
        for insight in succeeded:
            assert "Insight:" in insight

    def test_interfere_similar_thoughts_interfere(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["hello world", "hello there", "goodbye world"])
        results = engine.interfere()
        probs = [p for _, p in results]
        assert all(p >= 0 for p in probs)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)


# ── QuantumParallelProcessor ────────────────────────────────────────────────


class TestQuantumParallelProcessor:
    def test_parallel_process_returns_results(self):
        proc = QuantumParallelProcessor(num_streams=4)
        inputs = ["a", "b", "c", "d"]
        results = proc.parallel_process(inputs, lambda x: x.upper())
        assert results == ["A", "B", "C", "D"]

    def test_parallel_process_empty_inputs(self):
        proc = QuantumParallelProcessor(num_streams=4)
        results = proc.parallel_process([], lambda x: x)
        assert results == []

    def test_parallel_process_applies_processor(self):
        proc = QuantumParallelProcessor(num_streams=2)
        results = proc.parallel_process([1, 2, 3, 4], lambda x: x * 2)
        assert results == [2, 4, 6, 8]

    def test_parallel_process_records_metadata(self):
        proc = QuantumParallelProcessor(num_streams=4)
        proc.parallel_process(["a", "b"], lambda x: x)
        assert len(proc.results) == 1
        assert proc.results[0]["inputs"] == 2
        assert proc.results[0]["streams_used"] == 2

    def test_parallel_process_streams_usage(self):
        proc = QuantumParallelProcessor(num_streams=2)
        proc.parallel_process(["a", "b", "c"], lambda x: x)
        assert proc.results[0]["streams_used"] == 2

    def test_parallel_process_populates_streams(self):
        proc = QuantumParallelProcessor(num_streams=2)
        proc.parallel_process(["a", "b", "c"], lambda x: x)
        assert len(proc.streams[0]) == 2
        assert len(proc.streams[1]) == 1

    def test_get_parallel_capacity(self):
        proc = QuantumParallelProcessor(num_streams=8)
        assert proc.get_parallel_capacity() == 8

    def test_get_parallel_capacity_various(self):
        for n in [1, 4, 16, 32]:
            proc = QuantumParallelProcessor(num_streams=n)
            assert proc.get_parallel_capacity() == n

    def test_parallel_process_multiple_calls_accumulate(self):
        proc = QuantumParallelProcessor(num_streams=2)
        proc.parallel_process(["a"], lambda x: x)
        proc.parallel_process(["b", "c"], lambda x: x)
        assert len(proc.results) == 2
        assert proc.results[0]["inputs"] == 1
        assert proc.results[1]["inputs"] == 2


# ── HyperdimensionalProcessor ────────────────────────────────────────────────


class TestHyperdimensionalProcessor:
    def test_encode_returns_consistent_vector(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("symbol")
        v2 = hdp.encode("symbol")
        assert v1 is v2
        assert v1 == v2

    def test_encode_different_symbols_get_different_vectors(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("alpha")
        v2 = hdp.encode("beta")
        assert v1 != v2

    def test_encode_returns_correct_dimension(self):
        hdp = HyperdimensionalProcessor(dim=500)
        v = hdp.encode("test")
        assert len(v) == 500

    def test_encode_returns_binary_values(self):
        hdp = HyperdimensionalProcessor(dim=200)
        v = hdp.encode("val")
        for val in v:
            assert val in (-1, 1)

    def test_encode_text_returns_list_of_dim(self):
        hdp = HyperdimensionalProcessor(dim=100)
        result = hdp.encode_text("hello world")
        assert isinstance(result, list)
        assert len(result) == 100

    def test_encode_text_empty_returns_zeros(self):
        hdp = HyperdimensionalProcessor(dim=50)
        result = hdp.encode_text("")
        assert result == [0] * 50

    def test_encode_text_single_word(self):
        hdp = HyperdimensionalProcessor(dim=100)
        result = hdp.encode_text("hello")
        assert len(result) == 100
        assert any(v != 0 for v in result)

    def test_bundle_produces_vector(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bundled = hdp.bundle([v1, v2])
        assert isinstance(bundled, list)
        assert len(bundled) == 100

    def test_bundle_empty_returns_zeros(self):
        hdp = HyperdimensionalProcessor(dim=50)
        bundled = hdp.bundle([])
        assert bundled == [0] * 50

    def test_bundle_result_is_binary(self):
        hdp = HyperdimensionalProcessor(dim=200)
        v1 = hdp.encode("x")
        v2 = hdp.encode("y")
        bundled = hdp.bundle([v1, v2])
        for val in bundled:
            assert val in (-1.0, 1.0)

    def test_bind_produces_vector(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bound = hdp.bind(v1, v2)
        assert isinstance(bound, list)
        assert len(bound) == 100

    def test_bind_result_is_binary(self):
        hdp = HyperdimensionalProcessor(dim=200)
        v1 = hdp.encode("x")
        v2 = hdp.encode("y")
        bound = hdp.bind(v1, v2)
        for val in bound:
            assert val in (-1.0, 1.0)

    def test_bind_different_vectors_different_results(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        v3 = hdp.encode("c")
        assert hdp.bind(v1, v2) != hdp.bind(v1, v3)

    def test_similarity_range_negative_one_to_one(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        sim = hdp.similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_similarity_self_is_one(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v = hdp.encode("x")
        sim = hdp.similarity(v, v)
        assert sim == pytest.approx(1.0)

    def test_similarity_opposites(self):
        hdp = HyperdimensionalProcessor(dim=200)
        v = hdp.encode("x")
        opposite = [-x for x in v]
        sim = hdp.similarity(v, opposite)
        assert sim == pytest.approx(-1.0)

    def test_unbind_reverses_bind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bound = hdp.bind(v1, v2)
        unbound = hdp.unbind(bound, v2)
        assert unbound == hdp.bind(bound, v2)
        sim = hdp.similarity(unbound, v1)
        assert sim > 0.0

    def test_cleanup_removes_zero_vectors(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["keep"] = [1] * 10
        hdp.vectors["remove"] = [0] * 10
        removed = hdp.cleanup()
        assert removed == 1
        assert "remove" not in hdp.vectors
        assert "keep" in hdp.vectors

    def test_cleanup_no_removable_returns_zero(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["a"] = [1] * 10
        hdp.vectors["b"] = [-1] * 10
        removed = hdp.cleanup()
        assert removed == 0
        assert len(hdp.vectors) == 2

    def test_cleanup_empty_store(self):
        hdp = HyperdimensionalProcessor(dim=10)
        removed = hdp.cleanup()
        assert removed == 0

    def test_bind_dimension_mismatch_raises(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = [1] * 100
        v2 = [1] * 50
        with pytest.raises(ValueError, match="same dimension"):
            hdp.bind(v1, v2)

    def test_similarity_dimension_mismatch_raises(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = [1] * 100
        v2 = [1] * 50
        with pytest.raises(ValueError, match="same dimension"):
            hdp.similarity(v1, v2)


# ── TemporalReasoningEngine ──────────────────────────────────────────────────


class TestTemporalReasoningEngine:
    def test_add_event_stores_event(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "action", "value": 1})
        events = engine.get_current_events()
        assert len(events) == 1
        assert events[0]["type"] == "action"
        assert events[0]["value"] == 1

    def test_add_event_multiple(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "a"})
        engine.add_event({"type": "b"})
        engine.add_event({"type": "c"})
        events = engine.get_current_events()
        assert len(events) == 3
        assert events[2]["type"] == "c"

    def test_add_event_to_specific_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "main"}, timeline=0)
        engine.add_event({"type": "branch"}, timeline=2)
        assert len(engine.timelines[0]) == 1
        assert len(engine.timelines[2]) == 1
        assert engine.timelines[0][0]["type"] == "main"
        assert engine.timelines[2][0]["type"] == "branch"

    def test_add_event_out_of_range_ignored(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "bad"}, timeline=10)
        assert all(len(tl) == 0 for tl in engine.timelines)

    def test_add_event_includes_timeline_and_timestamp(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "x"})
        event = engine.get_current_events()[0]
        assert "timeline" in event
        assert "timestamp" in event

    def test_branch_creates_new_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "before"})
        new_tl = engine.branch("diverge")
        assert new_tl != 0
        assert isinstance(new_tl, int)

    def test_branch_copies_current_state(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "shared"})
        new_tl = engine.branch("split")
        assert len(engine.timelines[new_tl]) == 1
        assert engine.timelines[new_tl][0]["type"] == "shared"

    def test_branch_records_branch_point(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        new_tl = engine.branch("condition")
        assert len(engine.branch_points) == 1
        bp = engine.branch_points[0]
        assert bp["from_timeline"] == 0
        assert bp["to_timeline"] == new_tl
        assert bp["condition"] == "condition"

    def test_switch_timeline_returns_true_for_valid(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        assert engine.switch_timeline(3) is True
        assert engine.current_timeline == 3

    def test_switch_timeline_returns_false_for_invalid(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        assert engine.switch_timeline(10) is False
        assert engine.current_timeline == 0

    def test_switch_timeline_negative_returns_false(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        assert engine.switch_timeline(-1) is False

    def test_get_current_events_returns_list(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "a"})
        engine.add_event({"type": "b"})
        events = engine.get_current_events()
        assert isinstance(events, list)
        assert len(events) == 2

    def test_get_current_events_empty_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        events = engine.get_current_events()
        assert events == []

    def test_get_current_events_with_n(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        for i in range(10):
            engine.add_event({"i": i})
        events = engine.get_current_events(n=3)
        assert len(events) == 3
        assert events[0]["i"] == 7

    def test_merge_timelines_returns_merged_list(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "a"}, timeline=0)
        engine.add_event({"type": "b"}, timeline=1)
        merged = engine.merge_timelines(0, 1)
        assert isinstance(merged, list)
        assert len(merged) == 2

    def test_merge_timelines_sorted_by_timestamp(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "first"}, timeline=0)
        engine.add_event({"type": "second"}, timeline=1)
        merged = engine.merge_timelines(0, 1)
        timestamps = [e["timestamp"] for e in merged]
        assert timestamps == sorted(timestamps)

    def test_merge_timelines_with_empty(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "only"}, timeline=0)
        merged = engine.merge_timelines(0, 2)
        assert len(merged) == 1

    def test_switch_timeline_and_add_event(self):
        engine = TemporalReasoningEngine(timeline_depth=5)
        engine.add_event({"type": "main"})
        engine.branch("split")
        engine.switch_timeline(1)
        engine.add_event({"type": "branch_event"})
        assert len(engine.timelines[0]) == 1
        assert len(engine.timelines[1]) == 2

    def test_branch_wraps_around(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        new_tl = engine.branch("first")
        assert 0 <= new_tl < 3

    def test_merge_timelines_from_same(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"type": "a"}, timeline=0)
        merged = engine.merge_timelines(0, 0)
        assert len(merged) == 2
