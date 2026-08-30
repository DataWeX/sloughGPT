"""Tests for domains.soul.cognitive — CognitiveArchitecture and related classes."""

import pytest
from dataclasses import dataclass
from domains.soul.cognitive import (
    CognitiveArchitecture,
    SessionMemory,
    EpisodicMemoryStore,
    NeuralPlasticityEngine,
    MetaLearningEngine,
    DreamProcessingEngine,
    SentimentAnalyzer,
    EmotionalResponseGenerator,
    RelationshipMemory,
)


@dataclass
class FakeExperience:
    id: str
    data: str
    importance: float
    timestamp: str
    context: dict


class TestSessionMemory:
    def test_init_default(self):
        sm = SessionMemory()
        assert sm.max_turns == 20
        assert sm.conversation == []
        assert sm.session_id.startswith("session_")

    def test_init_custom_max_turns(self):
        sm = SessionMemory(max_turns=5)
        assert sm.max_turns == 5

    def test_session_id_unique(self):
        sm1 = SessionMemory()
        sm2 = SessionMemory()
        assert sm1.session_id != sm2.session_id

    def test_add_message(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert msg["turn"] == 0
        assert "timestamp" in msg
        assert len(sm.conversation) == 1

    def test_add_multiple_messages(self):
        sm = SessionMemory()
        sm.add("user", "hi")
        sm.add("assistant", "hello")
        sm.add("user", "how are you?")
        assert len(sm.conversation) == 3
        assert sm.conversation[0]["role"] == "user"
        assert sm.conversation[1]["role"] == "assistant"
        assert sm.conversation[2]["turn"] == 2

    def test_max_turns_eviction(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg{i}")
        assert len(sm.conversation) == 3
        assert sm.conversation[0]["content"] == "msg2"
        assert sm.conversation[2]["content"] == "msg4"

    def test_get_context(self):
        sm = SessionMemory()
        for i in range(5):
            sm.add("user", f"msg{i}")
        ctx = sm.get_context(3)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg2"

    def test_get_context_larger_than_session(self):
        sm = SessionMemory()
        sm.add("user", "only one")
        ctx = sm.get_context(10)
        assert len(ctx) == 1

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        full = sm.get_full_session()
        assert len(full) == 2
        full.append({"extra": True})
        assert len(sm.conversation) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "msg")
        old_id = sm.session_id
        sm.clear()
        assert sm.conversation == []
        assert sm.session_id != old_id

    def test_get_summary(self):
        sm = SessionMemory()
        sm.add("user", "hi")
        sm.add("assistant", "hello")
        summary = sm.get_summary()
        assert summary["turns"] == 2
        assert summary["session_id"] == sm.session_id
        assert "start" in summary

    def test_session_start_recorded(self):
        sm = SessionMemory()
        assert sm.session_start is not None

    def test_add_preserves_order(self):
        sm = SessionMemory()
        sm.add("user", "first")
        sm.add("assistant", "second")
        sm.add("user", "third")
        assert [m["content"] for m in sm.conversation] == ["first", "second", "third"]

    def test_add_empty_content(self):
        sm = SessionMemory()
        msg = sm.add("user", "")
        assert msg["content"] == ""
        assert len(sm.conversation) == 1

    def test_add_special_characters(self):
        sm = SessionMemory()
        msg = sm.add("user", "!@#$%^&*()_+")
        assert msg["content"] == "!@#$%^&*()_+"

    def test_add_unicode(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello 你好 مرحبا")
        assert msg["content"] == "hello 你好 مرحبا"

    def test_get_context_zero_returns_all(self):
        sm = SessionMemory()
        sm.add("user", "msg")
        ctx = sm.get_context(0)
        # conversation[-0:] returns the full list (Python slicing behavior)
        assert len(ctx) == 1

    def test_get_context_negative(self):
        sm = SessionMemory()
        sm.add("user", "msg")
        ctx = sm.get_context(-1)
        assert ctx == []

    def test_eviction_preserves_latest(self):
        sm = SessionMemory(max_turns=2)
        sm.add("user", "old1")
        sm.add("user", "old2")
        sm.add("user", "new")
        assert sm.conversation[-1]["content"] == "new"

    def test_clear_resets_start(self):
        sm = SessionMemory()
        sm.add("user", "msg")
        old_start = sm.session_start
        sm.clear()
        assert sm.session_start >= old_start

    def test_summary_roles(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        sm.add("user", "c")
        summary = sm.get_summary()
        assert summary["roles"]["user"] == 1
        assert summary["roles"]["assistant"] == 1

    def test_multiple_clears(self):
        sm = SessionMemory()
        sm.add("user", "msg1")
        sm.clear()
        sm.add("user", "msg2")
        sm.clear()
        assert sm.conversation == []
        assert len(sm.session_id) > 0

    def test_turn_monotonically_increases(self):
        sm = SessionMemory()
        turns = []
        for i in range(5):
            msg = sm.add("user", f"msg{i}")
            turns.append(msg["turn"])
        assert turns == sorted(turns)
        assert len(set(turns)) == 5


class TestEpisodicMemoryStore:
    def test_init(self):
        store = EpisodicMemoryStore()
        assert store.max_episodes == 100
        assert store.episodes == {}

    def test_save_episode(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hi"}]
        ep_id = store.save_episode("session_1", conv)
        assert ep_id.startswith("conv_")
        assert ep_id in store.episodes
        assert store.episodes[ep_id] == conv

    def test_save_multiple_episodes(self):
        store = EpisodicMemoryStore()
        ep1 = store.save_episode("s1", [{"role": "user", "content": "a"}])
        ep2 = store.save_episode("s2", [{"role": "user", "content": "b"}])
        assert ep1 != ep2
        assert len(store.episodes) == 2

    def test_get_episode(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}]
        ep_id = store.save_episode("s1", conv)
        result = store.get_episode(ep_id)
        assert result == conv

    def test_get_episode_missing(self):
        store = EpisodicMemoryStore()
        assert store.get_episode("nonexistent") is None

    def test_search_episodes(self):
        store = EpisodicMemoryStore()
        store.save_episode("s1", [{"role": "user", "content": "hello world"}])
        store.save_episode("s2", [{"role": "user", "content": "goodbye world"}])
        results = store.search_episodes("hello")
        assert len(results) == 1
        assert results[0]["episode_id"] in store.episodes

    def test_search_episodes_no_match(self):
        store = EpisodicMemoryStore()
        store.save_episode("s1", [{"role": "user", "content": "hello"}])
        results = store.search_episodes("nonexistent")
        assert len(results) == 0

    def test_search_episodes_limit(self):
        store = EpisodicMemoryStore()
        for i in range(10):
            store.save_episode(f"s{i}", [{"role": "user", "content": "keyword match"}])
        results = store.search_episodes("keyword", limit=3)
        assert len(results) == 3

    def test_eviction(self):
        store = EpisodicMemoryStore(max_episodes=2)
        store.save_episode("s1", [{"role": "user", "content": "a"}])
        store.save_episode("s2", [{"role": "user", "content": "b"}])
        store.save_episode("s3", [{"role": "user", "content": "c"}])
        assert len(store.episodes) == 2

    def test_calculate_importance_empty(self):
        store = EpisodicMemoryStore()
        assert store._calculate_importance([]) == 0.0

    def test_calculate_importance_short(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "short"}]
        score = store._calculate_importance(conv)
        assert 0.0 <= score <= 1.0

    def test_calculate_importance_long(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": f"msg{i}"} for i in range(15)]
        score = store._calculate_importance(conv)
        assert score >= 0.7

    def test_calculate_importance_keywords(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important remember critical key learn"}]
        score = store._calculate_importance(conv)
        assert score >= 0.9

    def test_get_recent_episodes(self):
        store = EpisodicMemoryStore()
        ep1 = store.save_episode("s1", [{"role": "user", "content": "a"}])
        ep2 = store.save_episode("s2", [{"role": "user", "content": "b"}])
        recent = store.get_recent_episodes(5)
        assert ep2 in recent
        assert ep1 in recent
        assert recent.index(ep2) < recent.index(ep1)

    def test_episode_metadata(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}]
        ep_id = store.save_episode("s1", conv)
        meta = store.episode_metadata[ep_id]
        assert meta["session_id"] == "s1"
        assert meta["turns"] == 1
        assert "saved" in meta
        assert "importance" in meta

    def test_get_recent_episodes_empty(self):
        store = EpisodicMemoryStore()
        assert store.get_recent_episodes() == []

    def test_search_case_insensitive(self):
        store = EpisodicMemoryStore()
        store.save_episode("s1", [{"role": "user", "content": "Hello World"}])
        results = store.search_episodes("hello world")
        assert len(results) == 1

    def test_save_copies_conversation(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hi"}]
        ep_id = store.save_episode("s1", conv)
        conv.append({"extra": True})
        assert len(store.episodes[ep_id]) == 1

    def test_evict_least_important(self):
        store = EpisodicMemoryStore(max_episodes=2)
        store.save_episode("s1", [{"role": "user", "content": "important remember key learn"}])
        store.save_episode("s2", [{"role": "user", "content": "b"}])
        store.save_episode("s3", [{"role": "user", "content": "c"}])
        assert len(store.episodes) == 2
        for meta in store.episode_metadata.values():
            assert meta["importance"] > 0

    def test_get_recent_episodes_limit(self):
        store = EpisodicMemoryStore()
        for i in range(5):
            store.save_episode(f"s{i}", [{"role": "user", "content": f"msg{i}"}])
        recent = store.get_recent_episodes(2)
        assert len(recent) == 2

    def test_search_returns_relevance(self):
        store = EpisodicMemoryStore()
        store.save_episode("s1", [{"role": "user", "content": "test query"}])
        results = store.search_episodes("test")
        assert results[0]["relevance"] == 0.5

    def test_search_returns_turns(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        store.save_episode("s1", conv)
        results = store.search_episodes("a")
        assert results[0]["turns"] == 2

    def test_empty_conversation_importance(self):
        store = EpisodicMemoryStore()
        assert store._calculate_importance([]) == 0.0

    def test_importance_cap_at_one(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important remember critical key learn extra long conversation with many messages " + "x " * 20}]
        score = store._calculate_importance(conv)
        assert score <= 1.0

    def test_same_session_id_same_hash(self):
        store = EpisodicMemoryStore()
        ep1 = store.save_episode("same_id", [{"role": "user", "content": "a"}])
        ep2 = store.save_episode("same_id", [{"role": "user", "content": "b"}])
        assert ep1 == ep2


class TestCognitiveArchitecture:
    def test_init(self):
        ca = CognitiveArchitecture()
        assert ca.working_capacity == 7
        assert len(ca.sensory_buffer) == 0
        assert len(ca.working_memory) == 0
        assert ca.semantic_memory == {}

    def test_custom_capacity(self):
        ca = CognitiveArchitecture(working_capacity=5)
        assert ca.working_capacity == 5

    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("input")
        assert result is True
        assert len(ca.sensory_buffer) == 1
        assert ca.sensory_buffer[0]["data"] == "input"
        assert "timestamp" in ca.sensory_buffer[0]

    def test_process_sensory_multiple(self):
        ca = CognitiveArchitecture()
        ca.process_sensory("a")
        ca.process_sensory("b")
        ca.process_sensory("c")
        assert len(ca.sensory_buffer) == 3

    def test_sensory_buffer_overflow(self):
        ca = CognitiveArchitecture()
        for i in range(101):
            ca.process_sensory(f"input{i}")
        assert len(ca.sensory_buffer) == 50

    def test_to_working(self):
        ca = CognitiveArchitecture()
        result = ca.to_working("item1")
        assert result is True
        assert len(ca.working_memory) == 1
        assert ca.working_memory[0] == "item1"

    def test_to_working_multiple(self):
        ca = CognitiveArchitecture()
        for i in range(5):
            ca.to_working(f"item{i}")
        assert len(ca.working_memory) == 5

    def test_working_memory_eviction(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(5):
            ca.to_working(f"item{i}")
        assert len(ca.working_memory) == 3
        assert ca.working_memory[0] == "item2"

    def test_add_to_session(self):
        ca = CognitiveArchitecture()
        result = ca.add_to_session("user", "hello")
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_get_session_context(self):
        ca = CognitiveArchitecture()
        for i in range(5):
            ca.add_to_session("user", f"msg{i}")
        ctx = ca.get_session_context(3)
        assert len(ctx) == 3

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        ca.add_to_session("assistant", "hi")
        ep_id = ca.save_session_as_episode()
        assert ep_id.startswith("conv_")

    def test_recall_episodes(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "important topic")
        ca.save_session_as_episode()
        results = ca.recall_episodes("important")
        assert len(results) >= 1

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        result = ca.to_semantic("fact1", "the sky is blue")
        assert result is True
        assert "fact1" in ca.semantic_memory
        assert ca.semantic_memory["fact1"]["value"] == "the sky is blue"
        assert ca.semantic_memory["fact1"]["strength"] == 1.0

    def test_to_semantic_strengthen(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "the sky is blue")
        ca.to_semantic("fact1", "the sky is blue")
        assert ca.semantic_memory["fact1"]["strength"] == 1.1

    def test_retrieve_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "the sky is blue")
        result = ca.retrieve_semantic("fact1")
        assert result == "the sky is blue"
        assert "last_accessed" in ca.semantic_memory["fact1"]

    def test_retrieve_semantic_missing(self):
        ca = CognitiveArchitecture()
        result = ca.retrieve_semantic("nonexistent")
        assert result is None

    def test_has_session_memory(self):
        ca = CognitiveArchitecture()
        assert hasattr(ca, "session_memory")
        assert isinstance(ca.session_memory, SessionMemory)

    def test_has_episodic_store(self):
        ca = CognitiveArchitecture()
        assert hasattr(ca, "episodic_store")
        assert isinstance(ca.episodic_store, EpisodicMemoryStore)

    def test_consolidation_creates_episode(self):
        ca = CognitiveArchitecture(working_capacity=2)
        ca.to_working("a")
        ca.to_working("b")
        ca.to_working("c")
        assert len(ca.working_memory) == 2

    def test_process_sensory_returns_bool(self):
        ca = CognitiveArchitecture()
        assert isinstance(ca.process_sensory("x"), bool)

    def test_to_working_returns_bool(self):
        ca = CognitiveArchitecture()
        assert isinstance(ca.to_working("x"), bool)

    def test_to_semantic_returns_bool(self):
        ca = CognitiveArchitecture()
        assert isinstance(ca.to_semantic("k", "v"), bool)

    def test_sensory_buffer_exact_overflow(self):
        ca = CognitiveArchitecture()
        for i in range(100):
            ca.process_sensory(f"i{i}")
        assert len(ca.sensory_buffer) == 100
        ca.process_sensory("overflow")
        assert len(ca.sensory_buffer) == 50

    def test_working_memory_exact_capacity(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(3):
            ca.to_working(f"i{i}")
        assert len(ca.working_memory) == 3
        ca.to_working("overflow")
        assert len(ca.working_memory) == 3

    def test_semantic_multiple_keys(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("k1", "v1")
        ca.to_semantic("k2", "v2")
        ca.to_semantic("k3", "v3")
        assert len(ca.semantic_memory) == 3
        assert ca.retrieve_semantic("k1") == "v1"
        assert ca.retrieve_semantic("k2") == "v2"
        assert ca.retrieve_semantic("k3") == "v3"

    def test_semantic_overwrite_value(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("k", "old")
        ca.to_semantic("k", "new")
        # Source code strengthens existing key but does NOT update value
        assert ca.semantic_memory["k"]["value"] == "old"
        assert ca.semantic_memory["k"]["strength"] == 1.1

    def test_recall_episodes_no_match(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        ca.save_session_as_episode()
        results = ca.recall_episodes("xyz")
        assert len(results) == 0

    def test_session_and_episodic_independent(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "msg1")
        ca.save_session_as_episode()
        ca.add_to_session("user", "msg2")
        assert len(ca.session_memory.conversation) == 2
        assert len(ca.episodic_store.episodes) == 1


class TestNeuralPlasticityEngine:
    def test_init(self):
        engine = NeuralPlasticityEngine()
        assert engine.learning_rate == 0.01
        assert len(engine.connections) == 0
        assert len(engine.activation_history) == 0

    def test_init_custom_lr(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        assert engine.learning_rate == 0.1

    def test_activate(self):
        engine = NeuralPlasticityEngine()
        engine.activate("neuron1", 0.5)
        assert len(engine.activation_history["neuron1"]) == 1
        assert engine.activation_history["neuron1"][0] == 0.5

    def test_activate_default_strength(self):
        engine = NeuralPlasticityEngine()
        engine.activate("neuron1")
        assert engine.activation_history["neuron1"][0] == 1.0

    def test_activate_overflow(self):
        engine = NeuralPlasticityEngine()
        for i in range(101):
            engine.activate("neuron1", float(i))
        assert len(engine.activation_history["neuron1"]) == 50

    def test_hebbian_learn(self):
        engine = NeuralPlasticityEngine()
        engine.activate("pre", 1.0)
        engine.activate("post", 1.0)
        strength = engine.hebbian_learn("pre", "post")
        assert strength > 0

    def test_hebbian_learn_no_activation(self):
        engine = NeuralPlasticityEngine()
        strength = engine.hebbian_learn("pre", "post")
        assert strength > 0

    def test_hebbian_learn_with_reward(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("pre", 1.0)
        engine.activate("post", 1.0)
        s1 = engine.hebbian_learn("pre", "post", reward=1.0)
        s2 = engine.hebbian_learn("pre", "post", reward=2.0)
        assert s2 > s1

    def test_get_connection_strength(self):
        engine = NeuralPlasticityEngine()
        assert engine.get_connection_strength("a", "b") == 0.0

    def test_prune_weak_connections(self):
        engine = NeuralPlasticityEngine(learning_rate=0.001)
        engine.activate("a", 0.01)
        engine.activate("b", 0.01)
        engine.hebbian_learn("a", "b")
        pruned = engine.prune_weak_connections(threshold=0.1)
        assert pruned >= 0

    def test_hebbian_accumulates(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        s1 = engine.hebbian_learn("a", "b")
        s2 = engine.hebbian_learn("a", "b")
        assert s2 > s1

    def test_prune_removes_weak(self):
        engine = NeuralPlasticityEngine(learning_rate=1.0)
        engine.activate("a", 0.001)
        engine.activate("b", 0.001)
        engine.hebbian_learn("a", "b")
        pruned = engine.prune_weak_connections(threshold=1.0)
        assert pruned >= 1
        assert engine.get_connection_strength("a", "b") == 0.0

    def test_prune_preserves_strong(self):
        engine = NeuralPlasticityEngine(learning_rate=0.5)
        engine.activate("a", 10.0)
        engine.activate("b", 10.0)
        engine.hebbian_learn("a", "b")
        pruned = engine.prune_weak_connections(threshold=0.01)
        assert pruned == 0
        assert engine.get_connection_strength("a", "b") > 0

    def test_multiple_neurons(self):
        engine = NeuralPlasticityEngine()
        for i in range(5):
            engine.activate(f"n{i}", float(i + 1))
        engine.hebbian_learn("n0", "n1")
        engine.hebbian_learn("n2", "n3")
        assert engine.get_connection_strength("n0", "n1") > 0
        assert engine.get_connection_strength("n2", "n3") > 0
        assert engine.get_connection_strength("n0", "n2") == 0.0

    def test_activate_exact_overflow_boundary(self):
        engine = NeuralPlasticityEngine()
        for i in range(100):
            engine.activate("n", float(i))
        assert len(engine.activation_history["n"]) == 100
        engine.activate("n", 100.0)
        assert len(engine.activation_history["n"]) == 50

    def test_hebbian_zero_reward(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        s = engine.hebbian_learn("a", "b", reward=0.0)
        assert s == 0.0

    def test_hebbian_negative_reward(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        s = engine.hebbian_learn("a", "b", reward=-1.0)
        assert s < 0

    def test_prune_empty(self):
        engine = NeuralPlasticityEngine()
        pruned = engine.prune_weak_connections(threshold=0.01)
        assert pruned == 0

    def test_prune_threshold_zero(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        engine.hebbian_learn("a", "b")
        pruned = engine.prune_weak_connections(threshold=0.0)
        assert pruned == 0

    def test_prune_threshold_negative(self):
        engine = NeuralPlasticityEngine(learning_rate=0.1)
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        engine.hebbian_learn("a", "b")
        pruned = engine.prune_weak_connections(threshold=-1.0)
        assert pruned == 0

    def test_connection_strength_bidirectional(self):
        engine = NeuralPlasticityEngine()
        engine.activate("a", 1.0)
        engine.activate("b", 1.0)
        engine.hebbian_learn("a", "b")
        assert engine.get_connection_strength("a", "b") > 0
        assert engine.get_connection_strength("b", "a") == 0.0

    def test_many_prune_cycles(self):
        engine = NeuralPlasticityEngine(learning_rate=0.001)
        for i in range(10):
            engine.activate(f"n{i}", 0.01)
            engine.hebbian_learn(f"n{i}", f"n{(i+1) % 10}")
        pruned = engine.prune_weak_connections(threshold=0.1)
        assert pruned >= 0


class TestMetaLearningEngine:
    def test_init(self):
        engine = MetaLearningEngine()
        assert "rote" in engine.strategies
        assert "spaced" in engine.strategies
        assert "interleaved" in engine.strategies
        assert "elaborative" in engine.strategies
        assert engine.best_strategy == "spaced"

    def test_record_outcome_success(self):
        engine = MetaLearningEngine()
        engine.record_outcome("rote", True)
        assert engine.strategies["rote"]["attempts"] == 1
        assert engine.strategies["rote"]["success"] == 1

    def test_record_outcome_failure(self):
        engine = MetaLearningEngine()
        engine.record_outcome("rote", False)
        assert engine.strategies["rote"]["attempts"] == 1
        assert engine.strategies["rote"]["success"] == 0

    def test_record_outcome_invalid_strategy(self):
        engine = MetaLearningEngine()
        engine.record_outcome("invalid", True)
        assert "invalid" not in engine.strategies

    def test_update_weights(self):
        engine = MetaLearningEngine()
        for _ in range(10):
            engine.record_outcome("rote", True)
        engine.update_weights()
        assert engine.strategies["rote"]["weight"] >= 1.0

    def test_get_strategy(self):
        engine = MetaLearningEngine()
        assert engine.get_strategy() in engine.strategies

    def test_best_strategy_updates(self):
        engine = MetaLearningEngine()
        for _ in range(10):
            engine.record_outcome("rote", True)
            engine.record_outcome("spaced", False)
        engine.update_weights()
        assert engine.get_strategy() == "rote"

    def test_all_strategies_default_weight(self):
        engine = MetaLearningEngine()
        for name in engine.strategies:
            assert engine.strategies[name]["weight"] == 1.0
            assert engine.strategies[name]["attempts"] == 0
            assert engine.strategies[name]["success"] == 0

    def test_weight_blending(self):
        engine = MetaLearningEngine()
        engine.record_outcome("spaced", True)
        engine.update_weights()
        w = engine.strategies["spaced"]["weight"]
        assert 0.7 <= w <= 1.0

    def test_multiple_updates_converge(self):
        engine = MetaLearningEngine()
        for _ in range(20):
            engine.record_outcome("rote", True)
            engine.record_outcome("spaced", False)
        engine.update_weights()
        engine.update_weights()
        assert engine.get_strategy() == "rote"
        assert engine.strategies["spaced"]["weight"] < 1.0

    def test_record_outcome_does_not_add_new(self):
        engine = MetaLearningEngine()
        engine.record_outcome("unknown", True)
        assert len(engine.strategies) == 4

    def test_all_success_rate_one(self):
        engine = MetaLearningEngine()
        for _ in range(100):
            engine.record_outcome("rote", True)
        engine.update_weights()
        # Weight formula: 0.7 * old + 0.3 * 1.0 = 0.7 + 0.3 = 1.0
        assert engine.strategies["rote"]["weight"] == pytest.approx(1.0)

    def test_all_failure_rate_zero(self):
        engine = MetaLearningEngine()
        for _ in range(100):
            engine.record_outcome("spaced", False)
        engine.update_weights()
        assert engine.strategies["spaced"]["weight"] < 1.0

    def test_interleaved_strategy(self):
        engine = MetaLearningEngine()
        engine.record_outcome("interleaved", True)
        engine.record_outcome("interleaved", True)
        engine.update_weights()
        # Weight formula: 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert engine.strategies["interleaved"]["weight"] == pytest.approx(1.0)

    def test_elaborative_strategy(self):
        engine = MetaLearningEngine()
        engine.record_outcome("elaborative", True)
        engine.update_weights()
        assert engine.strategies["elaborative"]["weight"] > 0

    def test_mixed_success_failure(self):
        engine = MetaLearningEngine()
        for _ in range(5):
            engine.record_outcome("rote", True)
        for _ in range(5):
            engine.record_outcome("rote", False)
        engine.update_weights()
        w = engine.strategies["rote"]["weight"]
        assert 0.7 <= w <= 1.0


class TestDreamProcessingEngine:
    def test_init(self):
        engine = DreamProcessingEngine()
        assert engine.dream_cycles == 0
        assert engine.consolidated == 0

    def test_dream_increments_cycles(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        engine.dream([], plasticity)
        assert engine.dream_cycles == 1

    def test_dream_no_memories(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        insights = engine.dream([], plasticity)
        assert insights == []

    def test_dream_with_memories(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id="m1", data="a", importance=0.9, timestamp="t1", context={}),
            FakeExperience(id="m2", data="b", importance=0.8, timestamp="t2", context={}),
            FakeExperience(id="m3", data="c", importance=0.7, timestamp="t3", context={}),
        ]
        insights = engine.dream(memories, plasticity)
        assert len(insights) >= 1
        assert engine.consolidated == 3

    def test_dream_connects_memories(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id="m1", data="a", importance=0.9, timestamp="t1", context={}),
            FakeExperience(id="m2", data="b", importance=0.8, timestamp="t2", context={}),
            FakeExperience(id="m3", data="c", importance=0.7, timestamp="t3", context={}),
        ]
        engine.dream(memories, plasticity)
        assert plasticity.get_connection_strength("m1", "m2") > 0

    def test_dream_limits_important(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id=f"m{i}", data=f"d{i}", importance=float(i) / 20, timestamp=f"t{i}", context={})
            for i in range(15)
        ]
        engine.dream(memories, plasticity)
        assert engine.consolidated == 10

    def test_dream_multiple_cycles(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        engine.dream([], plasticity)
        engine.dream([], plasticity)
        assert engine.dream_cycles == 2

    def test_dream_two_memories_no_insight(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id="m1", data="a", importance=0.9, timestamp="t1", context={}),
            FakeExperience(id="m2", data="b", importance=0.8, timestamp="t2", context={}),
        ]
        insights = engine.dream(memories, plasticity)
        assert insights == []
        assert engine.consolidated == 2

    def test_dream_consolidated_count_accumulates(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        mem1 = [FakeExperience(id="m1", data="a", importance=0.9, timestamp="t1", context={})]
        mem2 = [FakeExperience(id="m2", data="b", importance=0.8, timestamp="t2", context={})]
        engine.dream(mem1, plasticity)
        engine.dream(mem2, plasticity)
        assert engine.consolidated == 2

    def test_dream_sorted_by_importance(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id="low", data="a", importance=0.1, timestamp="t1", context={}),
            FakeExperience(id="high", data="b", importance=0.9, timestamp="t2", context={}),
            FakeExperience(id="mid", data="c", importance=0.5, timestamp="t3", context={}),
        ]
        engine.dream(memories, plasticity)
        assert plasticity.get_connection_strength("high", "mid") > 0

    def test_dream_insight_text(self):
        engine = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [
            FakeExperience(id=f"m{i}", data=f"d{i}", importance=0.9 - i * 0.1, timestamp=f"t{i}", context={})
            for i in range(5)
        ]
        insights = engine.dream(memories, plasticity)
        assert any("Pattern detected" in ins for ins in insights)


class TestSentimentAnalyzer:
    def test_init(self):
        sa = SentimentAnalyzer()
        assert "happy" in sa.emotion_keywords
        assert "sad" in sa.emotion_keywords
        assert "positive" in sa.sentiment_words

    def test_analyze_sentiment_positive(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("I love this amazing wonderful day")
        assert score > 0

    def test_analyze_sentiment_negative(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("I hate this terrible awful thing")
        assert score < 0

    def test_analyze_sentiment_neutral(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("okay fine alright")
        assert score >= 0

    def test_analyze_sentiment_empty(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("")
        assert score == 0.0

    def test_analyze_sentiment_mixed(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good but also bad")
        assert isinstance(score, float)

    def test_emotion_keywords_complete(self):
        sa = SentimentAnalyzer()
        for emotion in ["happy", "sad", "angry", "fear", "surprise", "neutral"]:
            assert len(sa.emotion_keywords[emotion]) > 0

    def test_sentiment_words_complete(self):
        sa = SentimentAnalyzer()
        assert len(sa.sentiment_words["positive"]) > 0
        assert len(sa.sentiment_words["negative"]) > 0

    def test_detect_emotion_happy(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am so happy and joyful today") == "happy"

    def test_detect_emotion_sad(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I feel sad and depressed") == "sad"

    def test_detect_emotion_angry(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am angry and frustrated") == "angry"

    def test_detect_emotion_fear(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am scared and worried") == "fear"

    def test_detect_emotion_surprise(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am surprised and shocked") == "surprise"

    def test_detect_emotion_neutral(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("just okay normal regular") == "neutral"

    def test_detect_emotion_empty(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("") == "neutral"

    def test_detect_emotion_no_match(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("xyz123 random") == "neutral"

    def test_analyze_returns_dict(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am happy")
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result
        assert "is_positive" in result
        assert "is_negative" in result
        assert "is_neutral" in result

    def test_analyze_positive(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I love this great wonderful day")
        assert result["sentiment"] > 0
        assert result["is_positive"] is True
        assert result["is_negative"] is False

    def test_analyze_negative(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I hate this terrible awful thing")
        assert result["sentiment"] < 0
        assert result["is_negative"] is True
        assert result["is_positive"] is False

    def test_analyze_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("just normal regular")
        assert result["is_neutral"] is True

    def test_analyze_intensity(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I love this amazing wonderful day")
        assert result["intensity"] == abs(result["sentiment"])

    def test_analyze_sentiment_boundaries(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good")
        assert score == 1.0

    def test_analyze_sentiment_all_negative(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("bad terrible awful")
        assert score == -1.0

    def test_analyze_sentiment_equal_mixed(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good bad")
        assert score == 0.0

    def test_analyze_sentiment_single_positive(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("great")
        assert score == 1.0

    def test_analyze_sentiment_single_negative(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("terrible")
        assert score == -1.0

    def test_analyze_sentiment_multiple_words(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good good bad")
        assert score == pytest.approx(1 / 3)

    def test_detect_emotion_multiple_emotions(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am happy but also scared")
        assert emotion in ("happy", "fear")

    def test_analyze_neutral_boundary(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("just")
        assert result["is_neutral"] is True


class TestEmotionalResponseGenerator:
    def test_init(self):
        er = EmotionalResponseGenerator()
        assert "happy" in er.empathy_responses
        assert "sad" in er.empathy_responses
        assert "angry" in er.empathy_responses
        assert "fear" in er.empathy_responses
        assert "surprise" in er.empathy_responses
        assert "neutral" in er.empathy_responses

    def test_init_qualifiers(self):
        er = EmotionalResponseGenerator()
        assert "high" in er.qualifiers
        assert "medium" in er.qualifiers
        assert "low" in er.qualifiers
        assert len(er.qualifiers["high"]) > 0
        assert len(er.qualifiers["medium"]) > 0
        assert len(er.qualifiers["low"]) > 0

    def test_generate_empathetic_response_happy(self):
        er = EmotionalResponseGenerator()
        resp = er.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_generate_empathetic_response_sad(self):
        er = EmotionalResponseGenerator()
        resp = er.generate_empathetic_response("sad", -0.8)
        assert isinstance(resp, str)

    def test_generate_empathetic_response_neutral(self):
        er = EmotionalResponseGenerator()
        resp = er.generate_empathetic_response("neutral", 0.0)
        assert isinstance(resp, str)

    def test_generate_empathetic_response_unknown(self):
        er = EmotionalResponseGenerator()
        resp = er.generate_empathetic_response("unknown_emotion", 0.0)
        assert isinstance(resp, str)

    def test_adapt_response_positive(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hello", "happy", 0.8)
        assert resp.endswith("! 😊")

    def test_adapt_response_negative(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hello", "sad", -0.8)
        assert resp.endswith(" 😔")

    def test_adapt_response_neutral(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hello", "neutral", 0.0)
        assert resp == "Hello"

    def test_format_emotional_response_with_empathy(self):
        er = EmotionalResponseGenerator()
        resp = er.format_emotional_response("Hello", "happy", 0.8, include_empathy=True)
        assert len(resp) > len("Hello")

    def test_format_emotional_response_without_empathy(self):
        er = EmotionalResponseGenerator()
        resp = er.format_emotional_response("Hello", "neutral", 0.0, include_empathy=False)
        assert resp == "Hello"

    def test_format_emotional_response_neutral_no_empathy(self):
        er = EmotionalResponseGenerator()
        resp = er.format_emotional_response("Hello", "neutral", 0.0)
        assert resp == "Hello"

    def test_adapt_response_boundary_positive(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hi", "happy", 0.5)
        assert resp == "Hi"

    def test_adapt_response_boundary_negative(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hi", "sad", -0.5)
        assert resp == "Hi"

    def test_format_emotional_response_sad_with_empathy(self):
        er = EmotionalResponseGenerator()
        resp = er.format_emotional_response("Hello", "sad", -0.8, include_empathy=True)
        assert len(resp) > len("Hello")

    def test_empathy_responses_all_nonempty(self):
        er = EmotionalResponseGenerator()
        for emotion, responses in er.empathy_responses.items():
            assert len(responses) > 0

    def test_qualifiers_all_nonempty(self):
        er = EmotionalResponseGenerator()
        for level, words in er.qualifiers.items():
            assert len(words) > 0

    def test_adapt_response_unknown_emotion(self):
        er = EmotionalResponseGenerator()
        resp = er.adapt_response("Hello", "unknown", 0.0)
        assert resp == "Hello"


class TestRelationshipMemory:
    def test_init(self):
        rm = RelationshipMemory()
        assert rm.user_profiles == {}
        assert rm.interaction_history == {}

    def test_get_user_profile_creates(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("user1")
        assert profile["user_id"] == "user1"
        assert profile["total_interactions"] == 0
        assert profile["satisfaction_score"] == 0.5

    def test_get_user_profile_existing(self):
        rm = RelationshipMemory()
        p1 = rm.get_user_profile("user1")
        p2 = rm.get_user_profile("user1")
        assert p1 is p2

    def test_update_from_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            "user1", "hello", "hi", 0.5, "neutral", None
        )
        profile = rm.get_user_profile("user1")
        assert profile["total_interactions"] == 1
        assert profile["last_interaction"] is not None

    def test_update_from_interaction_good_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] == 0.6

    def test_update_from_interaction_bad_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "bad")
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] == 0.4

    def test_update_multiple_interactions(self):
        rm = RelationshipMemory()
        for i in range(3):
            rm.update_from_interaction("user1", f"msg{i}", "resp", 0.5, "neutral")
        profile = rm.get_user_profile("user1")
        assert profile["total_interactions"] == 3

    def test_update_mood_history_trimming(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction("user1", "msg", "resp", 0.5, "neutral")
        profile = rm.get_user_profile("user1")
        assert len(profile["mood_history"]) <= 50

    def test_update_interaction_history_trimming(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction("user1", "msg", "resp", 0.5, "neutral")
        assert len(rm.interaction_history["user1"]) <= 100

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "happy")
        summary = rm.get_user_summary("user1")
        assert summary["user_id"] == "user1"
        assert summary["total_interactions"] == 1
        assert summary["dominant_emotion"] == "happy"

    def test_get_user_summary_empty(self):
        rm = RelationshipMemory()
        summary = rm.get_user_summary("new_user")
        assert summary["total_interactions"] == 0

    def test_get_relationship_context_new_user(self):
        rm = RelationshipMemory()
        ctx = rm.get_relationship_context("user1", "happy")
        assert "Currently feeling happy" in ctx

    def test_get_relationship_context_frequent_user(self):
        rm = RelationshipMemory()
        for i in range(6):
            rm.update_from_interaction("user1", "hello world", "hi", 0.5, "happy")
        ctx = rm.get_relationship_context("user1", "neutral")
        assert len(ctx) > 0

    def test_get_relationship_context_dissatisfied(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "bad")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "bad")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "bad")
        ctx = rm.get_relationship_context("user1", "neutral")
        assert "dissatisfied" in ctx

    def test_get_relationship_context_satisfied(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        ctx = rm.get_relationship_context("user1", "neutral")
        assert "happy" in ctx

    def test_topics_extracted(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "programming javascript python", "response", 0.5, "neutral")
        profile = rm.get_user_profile("user1")
        assert len(profile["topics_of_interest"]) > 0

    def test_emotional_tendencies_tracked(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "happy")
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "happy")
        profile = rm.get_user_profile("user1")
        assert profile["emotional_tendencies"]["happy"] == 2

    def test_satisfaction_score_clamped_max(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] <= 1.0

    def test_satisfaction_score_clamped_min(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "bad")
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] >= 0.0

    def test_mood_history_records_sentiment(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.7, "happy")
        profile = rm.get_user_profile("user1")
        assert profile["mood_history"][0]["sentiment"] == 0.7

    def test_interaction_history_records_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "hello", "hi", 0.5, "neutral", "good")
        history = rm.interaction_history["user1"]
        assert history[0]["feedback"] == "good"

    def test_multiple_users(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy")
        rm.update_from_interaction("u2", "hello", "hi", 0.5, "sad")
        assert rm.get_user_profile("u1")["total_interactions"] == 1
        assert rm.get_user_profile("u2")["total_interactions"] == 1

    def test_get_summary_multiple_topics(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "programming python data", "r", 0.5, "neutral")
        summary = rm.get_user_summary("u1")
        assert len(summary["top_topics"]) <= 5

    def test_get_relationship_context_no_tendencies(self):
        rm = RelationshipMemory()
        ctx = rm.get_relationship_context("new_user", "neutral")
        assert ctx == ""

    def test_update_with_none_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral", None)
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] == 0.5
