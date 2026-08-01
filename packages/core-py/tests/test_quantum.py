"""Tests for domains.soul.quantum: quantum cognition + hyperdimensional computing."""

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

DIM = 64


class TestQuantumState:
    def test_probability_from_amplitude(self):
        state = QuantumState(amplitude=0.5 + 0.5j, basis_state="x")
        assert state.probability == pytest.approx(0.5, abs=1e-9)
        assert state.phase == pytest.approx(math.pi / 4, abs=1e-9)

    def test_default_amplitude(self):
        state = QuantumState()
        assert state.probability == 1.0

    def test_normalize(self):
        state = QuantumState(amplitude=2 + 0j, basis_state="x")
        state.normalize()
        assert state.probability == pytest.approx(1.0, abs=1e-9)

    def test_measure_collapses_when_probable(self):
        state = QuantumState(amplitude=1 + 0j, basis_state="collapsed")
        assert state.measure() == "collapsed"

    def test_measure_empty_when_impossible(self):
        state = QuantumState(amplitude=0 + 0j, basis_state="never")
        assert state.measure() == ""


class TestQuantumCognitiveEngine:
    def test_superposition_creation(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["alpha", "beta", "gamma"])
        assert len(engine.superposition) == 3
        probs = [s.probability for s in engine.superposition]
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_empty_superposition(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition([])
        assert engine.superposition == []

    def test_interfere_probabilities_normalized(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["alpha", "beta", "gamma"])
        results = engine.interfere()
        assert sum(p for _, p in results) == pytest.approx(1.0, abs=1e-6)
        assert {s for s, _ in results} == {"alpha", "beta", "gamma"}

    def test_measure_returns_a_thought(self):
        engine = QuantumCognitiveEngine()
        engine.create_superposition(["alpha", "beta", "gamma"])
        assert engine.measure() in {"alpha", "beta", "gamma"}

    def test_measure_empty(self):
        assert QuantumCognitiveEngine().measure() == ""

    def test_entangle_and_tunnel(self, monkeypatch):
        engine = QuantumCognitiveEngine(coherence=1.0)
        engine.entangle("thought_a", "thought_b")
        monkeypatch.setattr(random, "random", lambda: 0.0)
        monkeypatch.setattr(random, "choice", lambda seq: seq[0])
        insight = engine.tunnel("barrier")
        assert insight == "Insight: thought_a \u27f7 thought_b"

    def test_tunnel_no_entanglement(self, monkeypatch):
        engine = QuantumCognitiveEngine(coherence=1.0)
        monkeypatch.setattr(random, "random", lambda: 0.0)
        assert engine.tunnel("barrier") is None

    def test_tunnel_fails_often(self, monkeypatch):
        engine = QuantumCognitiveEngine(coherence=1.0)
        engine.entangle("a", "b")
        monkeypatch.setattr(random, "random", lambda: 0.99)
        assert engine.tunnel("barrier") is None

    def test_similarity_jaccard(self):
        engine = QuantumCognitiveEngine()
        assert engine._similarity("the cat sat", "the cat sat") == 1.0
        assert engine._similarity("the cat sat", "the dog ran") == pytest.approx(1 / 5)
        assert engine._similarity("", "anything") == 0.0


class TestQuantumParallelProcessor:
    def test_processes_each_input(self):
        proc = QuantumParallelProcessor(num_streams=4)
        results = proc.parallel_process(["a", "b", "c"], lambda s: s.upper())
        assert results == ["A", "B", "C"]

    def test_streams_recorded(self):
        proc = QuantumParallelProcessor(num_streams=2)
        proc.parallel_process(["a", "b", "c"], lambda s: s)
        assert proc.streams[0] == ["a", "c"]
        assert proc.streams[1] == ["b"]

    def test_metadata(self):
        proc = QuantumParallelProcessor(num_streams=3)
        proc.parallel_process(["a", "b", "c", "d"], lambda s: s)
        meta = proc.results[-1]
        assert meta["inputs"] == 4
        assert meta["streams_used"] == 3
        assert "timestamp" in meta

    def test_capacity(self):
        assert QuantumParallelProcessor(num_streams=8).get_parallel_capacity() == 8


class TestHyperdimensionalProcessor:
    def setup_method(self):
        random.seed(1)

    def test_encode_caches(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        v1 = hd.encode("symbol")
        v2 = hd.encode("symbol")
        assert v1 is v2
        assert len(v1) == DIM

    def test_encode_binary_values(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert set(hd.encode("symbol")) <= {-1, 1}

    def test_encode_text_dimension(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert len(hd.encode_text("hello world")) == DIM

    def test_encode_text_empty_returns_zeros(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert hd.encode_text("") == [0] * DIM

    def test_encode_text_deterministic(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert hd.encode_text("same words here") == hd.encode_text("same words here")

    def test_bundle_empty(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert hd.bundle([]) == [0] * DIM

    def test_bundle_binary_output(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        vector = hd.bundle([hd.encode("a"), hd.encode("b")])
        assert set(vector) <= {-1.0, 1.0}

    def test_bind_requires_same_dim(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        with pytest.raises(ValueError):
            hd.bind([1.0] * DIM, [1.0] * (DIM + 1))

    def test_bind_binary(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        result = hd.bind([1.0] * DIM, [1.0] * DIM)
        assert result == [1.0] * DIM

    def test_similarity_identical_is_one(self):
        hd = HyperdimensionalProcessor(dim=2)
        assert hd.similarity([1, -1], [1, -1]) == 1.0

    def test_similarity_opposite_is_minus_one(self):
        hd = HyperdimensionalProcessor(dim=2)
        assert hd.similarity([1, -1], [-1, 1]) == -1.0

    def test_similarity_dim_mismatch(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        with pytest.raises(ValueError):
            hd.similarity([1.0] * DIM, [1.0] * (DIM + 1))

    def test_unbind_inverts_bind(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        v = hd.encode("content")
        k = hd.encode("key")
        assert hd.unbind(hd.bind(v, k), k) == v

    def test_cleanup_removes_zero_vectors(self):
        hd = HyperdimensionalProcessor(dim=4)
        hd.vectors["zero_sum"] = [1.0, -1.0, 1.0, -1.0]
        hd.vectors["keep"] = [1.0, 1.0, 1.0, 1.0]
        assert hd.cleanup() == 1
        assert "keep" in hd.vectors

    def test_cleanup_empty(self):
        hd = HyperdimensionalProcessor(dim=DIM)
        assert hd.cleanup() == 0


class TestTemporalReasoningEngine:
    def test_add_event_to_current_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"text": "started"})
        assert engine.timelines[0][0]["text"] == "started"
        assert "timestamp" in engine.timelines[0][0]

    def test_add_event_explicit_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"text": "x"}, timeline=2)
        assert engine.timelines[2][0]["text"] == "x"
        assert engine.timelines[0] == []

    def test_add_event_out_of_range_ignored(self):
        engine = TemporalReasoningEngine(timeline_depth=2)
        engine.add_event({"text": "x"}, timeline=5)
        assert engine.timelines[1] == []
        assert all(t == [] for t in engine.timelines)

    def test_branch_copies_state(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        engine.add_event({"text": "shared"})
        new = engine.branch("if condition")
        assert new == 1
        assert engine.timelines[1][0]["text"] == "shared"
        assert engine.branch_points[-1]["condition"] == "if condition"

    def test_switch_timeline(self):
        engine = TemporalReasoningEngine(timeline_depth=3)
        assert engine.switch_timeline(2) is True
        assert engine.current_timeline == 2
        assert engine.switch_timeline(9) is False
        assert engine.current_timeline == 2

    def test_get_current_events_last_n(self):
        engine = TemporalReasoningEngine(timeline_depth=2)
        for i in range(5):
            engine.add_event({"text": f"e{i}"})
        events = engine.get_current_events(n=2)
        assert [e["text"] for e in events] == ["e3", "e4"]

    def test_merge_timelines(self):
        engine = TemporalReasoningEngine(timeline_depth=2)
        engine.add_event({"text": "a"}, timeline=0)
        engine.add_event({"text": "b"}, timeline=1)
        merged = engine.merge_timelines(0, 1)
        assert len(merged) == 2
