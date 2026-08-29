"""Meaningful tests for SessionMemory, EpisodicMemoryStore, CognitiveArchitecture, NeuralPlasticityEngine, MetaLearningEngine."""

import time
from domains.soul.cognitive import (
    SessionMemory, EpisodicMemoryStore, CognitiveArchitecture,
    NeuralPlasticityEngine, MetaLearningEngine,
)


# ── SessionMemory ──────────────────────────────────────────────────────

class TestSessionMemory:
    def test_add_message(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert msg["turn"] == 0

    def test_add_multiple(self):
        sm = SessionMemory()
        sm.add("user", "hi")
        sm.add("assistant", "hello")
        sm.add("user", "how are you")
        assert len(sm.conversation) == 3

    def test_max_turns_eviction(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg{i}")
        assert len(sm.conversation) == 3
        # Oldest messages evicted
        assert sm.conversation[0]["content"] == "msg2"

    def test_get_context(self):
        sm = SessionMemory()
        for i in range(10):
            sm.add("user", f"msg{i}")
        ctx = sm.get_context(n=3)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg7"

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("user", "b")
        full = sm.get_full_session()
        assert len(full) == 2
        # Should be a copy
        full.clear()
        assert len(sm.conversation) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "msg")
        old_id = sm.session_id
        sm.clear()
        assert len(sm.conversation) == 0
        assert sm.session_id != old_id

    def test_summary(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        summary = sm.get_summary()
        assert summary["turns"] == 2
        assert summary["session_id"] is not None

    def test_session_id_unique(self):
        s1 = SessionMemory()
        s2 = SessionMemory()
        assert s1.session_id != s2.session_id


# ── EpisodicMemoryStore ────────────────────────────────────────────────

class TestEpisodicMemoryStore:
    def test_save_episode(self):
        em = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        ep_id = em.save_episode("session_1", conv)
        assert ep_id.startswith("conv_")
        assert em.get_episode(ep_id) is not None

    def test_calculate_importance_empty(self):
        em = EpisodicMemoryStore()
        assert em._calculate_importance([]) == 0.0

    def test_calculate_importance_long(self):
        em = EpisodicMemoryStore()
        conv = [{"content": f"message {i}"} for i in range(15)]
        score = em._calculate_importance(conv)
        assert score > 0.5  # Bonus for length

    def test_calculate_importance_keywords(self):
        em = EpisodicMemoryStore()
        conv = [{"content": "This is important and critical information"}]
        score = em._calculate_importance(conv)
        assert score > 0.5  # Bonus for keywords

    def test_eviction(self):
        em = EpisodicMemoryStore(max_episodes=2)
        em.save_episode("s1", [{"content": "a"}])
        em.save_episode("s2", [{"content": "b"}])
        em.save_episode("s3", [{"content": "c"}])
        assert len(em.episodes) == 2

    def test_search_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"content": "Python programming"}])
        em.save_episode("s2", [{"content": "Java programming"}])
        em.save_episode("s3", [{"content": "Cooking recipes"}])
        results = em.search_episodes("Python")
        assert len(results) == 1
        assert results[0]["turns"] == 1

    def test_search_episodes_limit(self):
        em = EpisodicMemoryStore()
        for i in range(10):
            em.save_episode(f"s{i}", [{"content": "Python is great"}])
        results = em.search_episodes("Python", limit=3)
        assert len(results) == 3

    def test_get_recent_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"content": "a"}])
        em.save_episode("s2", [{"content": "b"}])
        recent = em.get_recent_episodes(1)
        assert len(recent) == 1


# ── CognitiveArchitecture ──────────────────────────────────────────────

class TestCognitiveArchitecture:
    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("visual input")
        assert result is True
        assert len(ca.sensory_buffer) == 1

    def test_sensory_buffer_cap(self):
        ca = CognitiveArchitecture()
        for i in range(120):
            ca.process_sensory(f"input {i}")
        # Cap at 100, then keep last 50, then 20 more = 70
        assert len(ca.sensory_buffer) <= 100

    def test_to_working(self):
        ca = CognitiveArchitecture()
        ca.to_working("item1")
        assert "item1" in ca.working_memory

    def test_to_working_eviction(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(5):
            ca.to_working(f"item{i}")
        assert len(ca.working_memory) == 3
        assert ca.working_memory[0] == "item2"

    def test_add_to_session(self):
        ca = CognitiveArchitecture()
        msg = ca.add_to_session("user", "hello")
        assert msg["role"] == "user"

    def test_get_session_context(self):
        ca = CognitiveArchitecture()
        for i in range(5):
            ca.add_to_session("user", f"msg{i}")
        ctx = ca.get_session_context(n=3)
        assert len(ctx) == 3

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "Paris is the capital of France")
        assert ca.retrieve_semantic("fact1") == "Paris is the capital of France"

    def test_semantic_strengthen(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "value1")
        ca.to_semantic("fact1", "value2")  # Strengthen
        assert ca.semantic_memory["fact1"]["strength"] == 1.1

    def test_retrieve_semantic_missing(self):
        ca = CognitiveArchitecture()
        assert ca.retrieve_semantic("nonexistent") is None

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        ca.add_to_session("assistant", "hi")
        ep_id = ca.save_session_as_episode()
        assert ep_id is not None
        assert ca.recall_episodes("hello") is not None


# ── NeuralPlasticityEngine ─────────────────────────────────────────────

class TestNeuralPlasticityEngine:
    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("neuron1", 0.8)
        assert 0.8 in npe.activation_history["neuron1"]

    def test_activate_history_cap(self):
        npe = NeuralPlasticityEngine()
        for i in range(120):
            npe.activate("n1", float(i))
        # Cap at 100, then keep last 50, then 20 more = 70
        assert len(npe.activation_history["n1"]) <= 100

    def test_hebbian_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("pre", 1.0)
        npe.activate("post", 1.0)
        weight = npe.hebbian_learn("pre", "post")
        assert weight > 0

    def test_hebbian_learn_default_activation(self):
        npe = NeuralPlasticityEngine()
        weight = npe.hebbian_learn("pre", "post")
        assert weight > 0  # Uses default activation of 1.0

    def test_hebbian_learn_with_reward(self):
        npe = NeuralPlasticityEngine()
        w1 = npe.hebbian_learn("a", "b", reward=1.0)
        npe2 = NeuralPlasticityEngine()
        w2 = npe2.hebbian_learn("a", "b", reward=2.0)
        assert w2 > w1

    def test_get_connection_strength(self):
        npe = NeuralPlasticityEngine()
        npe.hebbian_learn("a", "b")
        assert npe.get_connection_strength("a", "b") > 0

    def test_prune_weak_connections(self):
        npe = NeuralPlasticityEngine()
        npe.connections["a"]["b"] = 0.001
        npe.connections["a"]["c"] = 0.5
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned == 1
        assert npe.get_connection_strength("a", "c") == 0.5

    def test_prune_no_weak(self):
        npe = NeuralPlasticityEngine()
        npe.connections["a"]["b"] = 0.5
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned == 0


# ── MetaLearningEngine ─────────────────────────────────────────────────

class TestMetaLearningEngine:
    def test_initial_best_strategy(self):
        mle = MetaLearningEngine()
        assert mle.get_strategy() == "spaced"

    def test_record_outcome(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", False)
        assert mle.strategies["rote"]["attempts"] == 3
        assert mle.strategies["rote"]["success"] == 2

    def test_record_outcome_unknown_strategy(self):
        mle = MetaLearningEngine()
        mle.record_outcome("nonexistent", True)
        # Should not crash

    def test_update_weights(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
        mle.update_weights()
        # Weight = 0.7 * 1.0 + 0.3 * 1.0 = 1.0 (stays at 1.0 with 100% success)
        assert mle.strategies["rote"]["weight"] == 1.0
        assert mle.best_strategy == "rote"

    def test_update_weights_low_success(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", False)
        mle.update_weights()
        # Weight = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert mle.strategies["rote"]["weight"] == 0.7

    def test_get_strategy_after_update(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("elaborative", True)
        for _ in range(10):
            mle.record_outcome("rote", False)
        mle.update_weights()
        # elaborative: 0.7*1.0 + 0.3*1.0 = 1.0, rote: 0.7*1.0 + 0.3*0.0 = 0.7
        # Other strategies stay at 1.0 (no attempts). Tie-breaking depends on dict order.
        assert mle.strategies["elaborative"]["weight"] > mle.strategies["rote"]["weight"]
