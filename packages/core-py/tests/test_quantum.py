"""Tests for domains.soul.quantum — QuantumState, QuantumCognitiveEngine, QuantumParallelProcessor, HyperdimensionalProcessor, TemporalReasoningEngine."""

import math
import random
import pytest
from domains.soul.quantum import (
    QuantumState, QuantumCognitiveEngine, QuantumParallelProcessor,
    HyperdimensionalProcessor, TemporalReasoningEngine,
)


# ── QuantumState ───────────────────────────────────────────────────────

class TestQuantumState:
    def test_init(self):
        qs = QuantumState()
        assert qs.amplitude == 1 + 0j

    def test_normalize(self):
        qs = QuantumState(amplitude=3 + 4j)
        qs.normalize()
        assert abs(qs.amplitude) == 1.0

    def test_measure(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="hello")
        result = qs.measure()
        assert result == "hello"

    def test_init_with_custom_amplitude(self):
        qs = QuantumState(amplitude=2 + 3j)
        assert qs.amplitude == 2 + 3j
        assert abs(qs.probability - 13.0) < 1e-10

    def test_init_with_basis_state(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="test_state")
        assert qs.basis_state == "test_state"

    def test_init_default_basis_state_empty(self):
        qs = QuantumState()
        assert qs.basis_state == ""

    def test_phase_is_computed(self):
        qs = QuantumState(amplitude=1 + 1j)
        assert abs(qs.phase - math.pi / 4) < 1e-10

    def test_normalize_zero_amplitude(self):
        qs = QuantumState(amplitude=0 + 0j)
        qs.normalize()
        assert qs.amplitude == 0 + 0j

    def test_normalize_already_normalized(self):
        qs = QuantumState(amplitude=1 + 0j)
        qs.normalize()
        assert abs(qs.amplitude) == 1.0

    def test_normalize_updates_probability(self):
        qs = QuantumState(amplitude=3 + 4j)
        qs.normalize()
        assert abs(qs.probability - 1.0) < 1e-10

    def test_measure_zero_probability(self):
        qs = QuantumState(amplitude=0 + 0j, basis_state="zero")
        result = qs.measure()
        assert result == ""

    def test_measure_probability_one_always_returns(self):
        qs = QuantumState(amplitude=1 + 0j, basis_state="always")
        for _ in range(50):
            assert qs.measure() == "always"

    def test_measure_probability_zero_never_returns(self):
        qs = QuantumState(amplitude=0 + 0j, basis_state="never")
        for _ in range(50):
            assert qs.measure() != "never"

    def test_phase_with_negative_real(self):
        qs = QuantumState(amplitude=-1 + 0j)
        assert abs(qs.phase - math.pi) < 1e-10

    def test_phase_with_negative_imaginary(self):
        qs = QuantumState(amplitude=0 - 1j)
        assert abs(qs.phase + math.pi / 2) < 1e-10


# ── QuantumCognitiveEngine ─────────────────────────────────────────────

class TestQuantumCognitiveEngine:
    def test_init(self):
        qce = QuantumCognitiveEngine()
        assert qce.coherence == 0.9

    def test_create_superposition(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["thought1", "thought2", "thought3"])
        assert len(qce.superposition) == 3

    def test_interfere(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["alpha", "beta"])
        result = qce.interfere()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_measure(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["x", "y"])
        result = qce.measure()
        assert result in ["x", "y"]

    def test_entangle(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["a", "b", "c"])
        qce.entangle("a", "b")
        assert len(qce.entangled_pairs) > 0

    def test_tunnel(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["a", "b"])
        result = qce.tunnel("barrier")
        assert result is None or isinstance(result, str)

    def test_create_superposition_empty(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition([])
        assert len(qce.superposition) == 0

    def test_create_superposition_single(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["only_one"])
        assert len(qce.superposition) == 1
        assert qce.superposition[0].basis_state == "only_one"

    def test_superposition_amplitudes_are_normalized(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["a", "b", "c"])
        total_prob = sum(abs(s.amplitude) ** 2 for s in qce.superposition)
        assert abs(total_prob - 1.0) < 1e-10

    def test_interfere_probabilities_sum_to_one(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["x", "y", "z"])
        result = qce.interfere()
        total = sum(p for _, p in result)
        assert abs(total - 1.0) < 1e-10

    def test_interfere_different_states_gives_different_results(self):
        qce1 = QuantumCognitiveEngine()
        qce1.create_superposition(["alpha", "beta"])
        r1 = qce1.interfere()
        qce2 = QuantumCognitiveEngine()
        qce2.create_superposition(["hello", "world"])
        r2 = qce2.interfere()
        assert r1 != r2

    def test_measure_empty_superposition(self):
        qce = QuantumCognitiveEngine()
        result = qce.measure()
        assert result == ""

    def test_entangle_multiple_pairs(self):
        qce = QuantumCognitiveEngine()
        qce.entangle("a", "b")
        qce.entangle("c", "d")
        assert len(qce.entangled_pairs) == 2
        assert qce.entangled_pairs[0] == ("a", "b")
        assert qce.entangled_pairs[1] == ("c", "d")

    def test_tunnel_no_entangled_pairs(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["a", "b"])
        result = qce.tunnel("barrier")
        assert result is None

    def test_coherence_affects_tunnel_probability(self):
        qce_high = QuantumCognitiveEngine(coherence=0.99)
        qce_low = QuantumCognitiveEngine(coherence=0.01)
        qce_high.create_superposition(["a", "b"])
        qce_low.create_superposition(["a", "b"])
        qce_high.entangle("a", "b")
        qce_low.entangle("a", "b")
        high_insights = sum(1 for _ in range(1000) if qce_high.tunnel("barrier") is not None)
        low_insights = sum(1 for _ in range(1000) if qce_low.tunnel("barrier") is not None)
        assert high_insights >= low_insights

    def test_measure_returns_from_superposition(self):
        qce = QuantumCognitiveEngine()
        thoughts = ["a", "b", "c", "d"]
        qce.create_superposition(thoughts)
        for _ in range(100):
            result = qce.measure()
            assert result in thoughts

    def test_create_superposition_resets_previous(self):
        qce = QuantumCognitiveEngine()
        qce.create_superposition(["x", "y"])
        assert len(qce.superposition) == 2
        qce.create_superposition(["a"])
        assert len(qce.superposition) == 1

    def test_decoherence_rate_default(self):
        qce = QuantumCognitiveEngine()
        assert qce.decoherence_rate == 0.01

    def test_init_custom_coherence(self):
        qce = QuantumCognitiveEngine(coherence=0.5)
        assert qce.coherence == 0.5

    def test_tunnel_generates_insight_format(self):
        qce = QuantumCognitiveEngine(coherence=0.99)
        qce.entangle("thought_a", "thought_b")
        random.seed(42)
        result = qce.tunnel("barrier")
        if result is not None:
            assert "Insight:" in result
            assert "thought_a" in result
            assert "thought_b" in result

    def test_similarity_identical_strings(self):
        qce = QuantumCognitiveEngine()
        assert qce._similarity("hello", "hello") == 1.0

    def test_similarity_completely_different(self):
        qce = QuantumCognitiveEngine()
        assert qce._similarity("hello", "world") == 0.0

    def test_similarity_empty_strings(self):
        qce = QuantumCognitiveEngine()
        assert qce._similarity("", "") == 0.0
        assert qce._similarity("hello", "") == 0.0
        assert qce._similarity("", "hello") == 0.0

    def test_similarity_partial_overlap(self):
        qce = QuantumCognitiveEngine()
        sim = qce._similarity("a b c", "b c d")
        assert 0.0 < sim < 1.0


# ── QuantumParallelProcessor ───────────────────────────────────────────

class TestQuantumParallelProcessor:
    def test_init(self):
        qpp = QuantumParallelProcessor()
        assert qpp.num_streams == 8

    def test_parallel_process(self):
        qpp = QuantumParallelProcessor(num_streams=2)
        inputs = ["hello", "world", "test"]
        result = qpp.parallel_process(inputs, lambda x: x.upper())
        assert result == ["HELLO", "WORLD", "TEST"]

    def test_get_parallel_capacity(self):
        qpp = QuantumParallelProcessor(num_streams=4)
        assert qpp.get_parallel_capacity() == 4

    def test_init_custom_streams(self):
        qpp = QuantumParallelProcessor(num_streams=16)
        assert qpp.num_streams == 16
        assert len(qpp.streams) == 16

    def test_streams_populated(self):
        qpp = QuantumParallelProcessor(num_streams=3)
        qpp.parallel_process(["a", "b", "c", "d"], lambda x: x)
        assert len(qpp.streams[0]) == 2  # a, d
        assert len(qpp.streams[1]) == 1  # b
        assert len(qpp.streams[2]) == 1  # c

    def test_results_logged(self):
        qpp = QuantumParallelProcessor(num_streams=2)
        qpp.parallel_process(["x"], lambda x: x)
        assert len(qpp.results) == 1
        assert qpp.results[0]["inputs"] == 1
        assert qpp.results[0]["streams_used"] == 1

    def test_empty_inputs(self):
        qpp = QuantumParallelProcessor()
        result = qpp.parallel_process([], lambda x: x)
        assert result == []

    def test_single_input(self):
        qpp = QuantumParallelProcessor(num_streams=4)
        result = qpp.parallel_process(["only"], lambda x: x * 2)
        assert result == ["onlyonly"]

    def test_numeric_processor(self):
        qpp = QuantumParallelProcessor()
        result = qpp.parallel_process([1, 2, 3], lambda x: x ** 2)
        assert result == [1, 4, 9]

    def test_many_inputs_wraps_streams(self):
        qpp = QuantumParallelProcessor(num_streams=2)
        inputs = [str(i) for i in range(10)]
        result = qpp.parallel_process(inputs, lambda x: x)
        assert result == inputs
        assert len(qpp.streams[0]) == 5
        assert len(qpp.streams[1]) == 5

    def test_timestamp_recorded(self):
        qpp = QuantumParallelProcessor()
        qpp.parallel_process(["a"], lambda x: x)
        assert "timestamp" in qpp.results[0]

    def test_results_accumulate(self):
        qpp = QuantumParallelProcessor()
        qpp.parallel_process(["a"], lambda x: x)
        qpp.parallel_process(["b", "c"], lambda x: x)
        assert len(qpp.results) == 2
        assert qpp.results[0]["inputs"] == 1
        assert qpp.results[1]["inputs"] == 2

    def test_identity_processor(self):
        qpp = QuantumParallelProcessor()
        inputs = ["a", "b", "c"]
        result = qpp.parallel_process(inputs, lambda x: x)
        assert result == inputs


# ── HyperdimensionalProcessor ──────────────────────────────────────────

class TestHyperdimensionalProcessor:
    def test_init(self):
        hdp = HyperdimensionalProcessor(dim=100)
        assert hdp.dim == 100

    def test_encode(self):
        hdp = HyperdimensionalProcessor(dim=100)
        vec = hdp.encode("hello")
        assert len(vec) == 100
        assert all(v in [-1, 1] for v in vec)

    def test_encode_deterministic(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("hello")
        v2 = hdp.encode("hello")
        assert v1 == v2

    def test_encode_different_symbols(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("hello")
        v2 = hdp.encode("world")
        assert v1 != v2

    def test_bundle(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        result = hdp.bundle([v1, v2])
        assert len(result) == 100
        assert all(v in [-1.0, 1.0] for v in result)

    def test_bundle_empty(self):
        hdp = HyperdimensionalProcessor(dim=100)
        result = hdp.bundle([])
        assert result == [0] * 100

    def test_bind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        result = hdp.bind(v1, v2)
        assert len(result) == 100
        assert all(v in [-1.0, 1.0] for v in result)

    def test_bind_dimension_mismatch(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        with pytest.raises(ValueError):
            hdp.bind(v1, [1, 2, 3])

    def test_similarity_identical(self):
        hdp = HyperdimensionalProcessor(dim=1000)
        v = hdp.encode("hello")
        sim = hdp.similarity(v, v)
        assert sim == 1.0

    def test_similarity_independent_vectors(self):
        hdp = HyperdimensionalProcessor(dim=10000)
        random.seed(42)
        v1 = [random.choice([-1, 1]) for _ in range(10000)]
        v2 = [random.choice([-1, 1]) for _ in range(10000)]
        sim = hdp.similarity(v1, v2)
        assert -0.1 < sim < 0.1

    def test_similarity_dimension_mismatch(self):
        hdp = HyperdimensionalProcessor(dim=100)
        with pytest.raises(ValueError):
            hdp.similarity([1, 2], [1, 2, 3])

    def test_unbind(self):
        hdp = HyperdimensionalProcessor(dim=100)
        v1 = hdp.encode("a")
        v2 = hdp.encode("b")
        bound = hdp.bind(v1, v2)
        unbound = hdp.unbind(bound, v2)
        sim = hdp.similarity(unbound, v1)
        assert sim > 0.5

    def test_encode_text(self):
        hdp = HyperdimensionalProcessor(dim=100)
        vec = hdp.encode_text("hello world test")
        assert len(vec) == 100
        assert all(v in [-1.0, 1.0] for v in vec)

    def test_encode_text_empty(self):
        hdp = HyperdimensionalProcessor(dim=100)
        vec = hdp.encode_text("")
        assert vec == [0] * 100

    def test_encode_text_single_word(self):
        hdp = HyperdimensionalProcessor(dim=100)
        vec1 = hdp.encode_text("hello")
        vec2 = hdp.encode("hello")
        assert vec1 == vec2

    def test_cleanup_removes_zero_sum_vectors(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["a"] = [0] * 10
        hdp.vectors["b"] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        removed = hdp.cleanup()
        assert removed == 2
        assert "a" not in hdp.vectors
        assert "b" not in hdp.vectors

    def test_cleanup_keeps_nonzero_sum_vectors(self):
        hdp = HyperdimensionalProcessor(dim=10)
        hdp.vectors["a"] = [1] * 10
        hdp.vectors["b"] = [-1] * 10
        removed = hdp.cleanup()
        assert removed == 0
        assert "a" in hdp.vectors
        assert "b" in hdp.vectors

    def test_bind_self_inverse(self):
        hdp = HyperdimensionalProcessor(dim=1000)
        v = hdp.encode("test")
        bound = hdp.bind(v, v)
        unbound = hdp.unbind(bound, v)
        sim = hdp.similarity(unbound, v)
        assert sim > 0.8

    def test_default_dim(self):
        hdp = HyperdimensionalProcessor()
        assert hdp.dim == 10000

    def test_vectors_dict_populated(self):
        hdp = HyperdimensionalProcessor(dim=50)
        hdp.encode("x")
        assert "x" in hdp.vectors
        assert len(hdp.vectors["x"]) == 50


# ── TemporalReasoningEngine ────────────────────────────────────────────

class TestTemporalReasoningEngine:
    def test_init(self):
        tre = TemporalReasoningEngine()
        assert tre.timeline_depth == 5
        assert len(tre.timelines) == 5

    def test_add_event(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "observation", "value": 42})
        events = tre.get_current_events()
        assert len(events) == 1
        assert events[0]["type"] == "observation"

    def test_add_event_specific_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"}, timeline=2)
        tre.switch_timeline(2)
        assert len(tre.get_current_events()) == 1

    def test_branch(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "start"})
        new_tl = tre.branch("condition_a")
        assert new_tl != tre.current_timeline
        assert len(tre.branch_points) == 1

    def test_switch_timeline(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(2)
        assert result is True
        assert tre.current_timeline == 2

    def test_switch_timeline_invalid(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(10)
        assert result is False
        assert tre.current_timeline == 0

    def test_switch_timeline_negative(self):
        tre = TemporalReasoningEngine()
        result = tre.switch_timeline(-1)
        assert result is False

    def test_get_current_events_empty(self):
        tre = TemporalReasoningEngine()
        events = tre.get_current_events()
        assert events == []

    def test_get_current_events_limit(self):
        tre = TemporalReasoningEngine()
        for i in range(10):
            tre.add_event({"i": i})
        events = tre.get_current_events(n=3)
        assert len(events) == 3

    def test_merge_timelines(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "a"}, timeline=0)
        tre.add_event({"type": "b"}, timeline=1)
        merged = tre.merge_timelines(0, 1)
        assert len(merged) == 2

    def test_branch_copies_state(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "start"})
        new_tl = tre.branch("split")
        tre.switch_timeline(new_tl)
        events = tre.get_current_events()
        assert len(events) == 1
        assert events[0]["type"] == "start"

    def test_branch_wraps_around(self):
        tre = TemporalReasoningEngine(timeline_depth=3)
        tre.current_timeline = 2
        new_tl = tre.branch("wrap")
        assert new_tl == 0

    def test_add_event_to_boundary_timeline(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "edge"}, timeline=0)
        tre.add_event({"type": "edge"}, timeline=4)
        assert len(tre.timelines[0]) == 1
        assert len(tre.timelines[4]) == 1

    def test_add_event_out_of_range_ignored(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"type": "bad"}, timeline=100)
        tre.add_event({"type": "bad"}, timeline=-1)
        for tl in tre.timelines:
            assert len(tl) == 0

    def test_branch_records_timestamp(self):
        tre = TemporalReasoningEngine()
        tre.branch("cond")
        assert "timestamp" in tre.branch_points[0]

    def test_multiple_branches(self):
        tre = TemporalReasoningEngine()
        tre.branch("c1")
        tre.branch("c2")
        tre.branch("c3")
        assert len(tre.branch_points) == 3

    def test_merge_preserves_all_events(self):
        tre = TemporalReasoningEngine()
        tre.add_event({"tick": 1}, timeline=0)
        tre.add_event({"tick": 3}, timeline=1)
        tre.add_event({"tick": 2}, timeline=0)
        merged = tre.merge_timelines(0, 1)
        ticks = [e["tick"] for e in merged]
        assert sorted(ticks) == [1, 2, 3]

    def test_default_timeline_depth(self):
        tre = TemporalReasoningEngine()
        assert tre.timeline_depth == 5
        assert tre.current_timeline == 0

    def test_custom_timeline_depth(self):
        tre = TemporalReasoningEngine(timeline_depth=10)
        assert tre.timeline_depth == 10
        assert len(tre.timelines) == 10
