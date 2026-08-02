"""Tests for domains/soul/cognitive.py (cognitive SLO: memory & learning)."""

import hashlib

import pytest

from domains.soul.cognitive import (
    CognitiveArchitecture,
    DreamProcessingEngine,
    EmotionalResponseGenerator,
    EpisodicMemoryStore,
    MetaLearningEngine,
    NeuralPlasticityEngine,
    RelationshipMemory,
    SentimentAnalyzer,
    SessionMemory,
)


class FakeMemory:
    def __init__(self, mem_id, importance):
        self.id = mem_id
        self.importance = importance


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        assert self.analyzer.analyze_sentiment("this is great and wonderful") > 0

    def test_negative_sentiment(self):
        assert self.analyzer.analyze_sentiment("this is bad and terrible") < 0

    def test_mixed_sentiment_returns_ratio(self):
        val = self.analyzer.analyze_sentiment("good good bad")
        assert abs(val - (2 - 1) / 3) < 1e-9

    def test_empty_text_is_neutral(self):
        assert self.analyzer.analyze_sentiment("") == 0.0

    def test_no_sentiment_words_is_neutral(self):
        assert self.analyzer.analyze_sentiment("abcdef") == 0.0

    def test_detect_emotion_happy(self):
        assert self.analyzer.detect_emotion("I am so happy and excited") == "happy"

    def test_detect_emotion_sad(self):
        assert self.analyzer.detect_emotion("I feel so sad and depressed") == "sad"

    def test_detect_emotion_angry(self):
        assert self.analyzer.detect_emotion("this makes me angry and furious") == "angry"

    def test_detect_emotion_fear(self):
        assert self.analyzer.detect_emotion("I am scared and worried") == "fear"

    def test_detect_emotion_surprise(self):
        assert self.analyzer.detect_emotion("I am shocked and surprised by this") == "surprise"

    def test_detect_emotion_neutral_fallback(self):
        assert self.analyzer.detect_emotion("") == "neutral"
        assert self.analyzer.detect_emotion("xyzzy") == "neutral"

    def test_detect_emotion_is_case_insensitive(self):
        assert self.analyzer.detect_emotion("I AM HAPPY") == "happy"

    def test_analyze_structure(self):
        result = self.analyzer.analyze("I am happy today")
        assert result["emotion"] == "happy"
        assert result["sentiment"] > 0
        assert result["intensity"] > 0
        assert result["is_positive"] is True
        assert result["is_negative"] is False
        assert result["is_neutral"] is False

    def test_analyze_neutral(self):
        result = self.analyzer.analyze("nothing here")
        assert result["is_neutral"] is True
        assert result["sentiment"] == 0.0
        assert result["intensity"] == 0.0


# ---------------------------------------------------------------------------
# EmotionalResponseGenerator
# ---------------------------------------------------------------------------

class TestEmotionalResponseGenerator:
    def setup_method(self):
        self.gen = EmotionalResponseGenerator()

    def test_known_emotion_returns_response(self):
        resp = self.gen.generate_empathetic_response("sad", -0.5)
        assert resp in self.gen.empathy_responses["sad"]

    def test_unknown_emotion_falls_back_to_neutral(self):
        resp = self.gen.generate_empathetic_response("unknown", 0.0)
        assert resp in self.gen.empathy_responses["neutral"]

    def test_adapt_positive(self):
        out = self.gen.adapt_response("hello", "happy", 0.9)
        assert out == "hello! 😊"

    def test_adapt_negative(self):
        out = self.gen.adapt_response("hello", "sad", -0.9)
        assert out == "hello 😔"

    def test_adapt_neutral(self):
        assert self.gen.adapt_response("hello", "neutral", 0.0) == "hello"

    def test_format_with_empathy(self):
        out = self.gen.format_emotional_response("base", "sad", -0.9)
        assert any(out.startswith(r) for r in self.gen.empathy_responses["sad"])
        assert "base" in out

    def test_format_without_empathy(self):
        out = self.gen.format_emotional_response("base", "sad", -0.9, include_empathy=False)
        assert out == "base 😔"

    def test_format_neutral_emotion_no_empathy(self):
        out = self.gen.format_emotional_response("base", "neutral", 0.0)
        assert out == "base"


# ---------------------------------------------------------------------------
# RelationshipMemory
# ---------------------------------------------------------------------------

class TestRelationshipMemory:
    def setup_method(self):
        self.mem = RelationshipMemory()

    def test_get_user_profile_creates(self):
        p = self.mem.get_user_profile("u1")
        assert p["user_id"] == "u1"
        assert p["total_interactions"] == 0
        assert p["satisfaction_score"] == 0.5
        assert p["last_interaction"] is None

    def test_get_user_profile_idempotent(self):
        self.mem.get_user_profile("u1")
        assert self.mem.get_user_profile("u1") is self.mem.user_profiles["u1"]

    def test_update_from_interaction_basic(self):
        self.mem.update_from_interaction("u1", "I love this wonderful feature", "thanks", 0.8, "happy")
        p = self.mem.user_profiles["u1"]
        assert p["total_interactions"] == 1
        assert p["emotional_tendencies"]["happy"] == 1
        assert p["mood_history"][-1]["emotion"] == "happy"
        assert len(self.mem.interaction_history["u1"]) == 1

    def test_update_feedback_good(self):
        self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="good")
        assert self.mem.user_profiles["u1"]["satisfaction_score"] == 0.6

    def test_update_feedback_bad(self):
        self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="bad")
        assert self.mem.user_profiles["u1"]["satisfaction_score"] == 0.4

    def test_satisfaction_score_clamped_high(self):
        for _ in range(10):
            self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="good")
        assert self.mem.user_profiles["u1"]["satisfaction_score"] == 1.0

    def test_satisfaction_score_clamped_low(self):
        for _ in range(10):
            self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="bad")
        assert self.mem.user_profiles["u1"]["satisfaction_score"] == 0.0

    def test_mood_history_capped_at_50(self):
        for i in range(55):
            self.mem.update_from_interaction("u1", f"message {i}", "reply", 0.0, "neutral")
        assert len(self.mem.user_profiles["u1"]["mood_history"]) == 50

    def test_interaction_history_capped_at_100(self):
        for i in range(105):
            self.mem.update_from_interaction("u1", f"message {i}", "reply", 0.0, "neutral")
        assert len(self.mem.interaction_history["u1"]) == 100

    def test_topics_of_interest(self):
        self.mem.update_from_interaction("u1", "talking about photography and mountains", "ok", 0.0, "neutral")
        p = self.mem.user_profiles["u1"]
        assert p["topics_of_interest"]["photography"] == 1
        assert p["topics_of_interest"]["mountains"] == 1

    def test_get_user_summary_empty(self):
        s = self.mem.get_user_summary("u1")
        assert s["dominant_emotion"] == "neutral"
        assert s["total_interactions"] == 0
        assert s["top_topics"] == []

    def test_get_user_summary_populated(self):
        self.mem.update_from_interaction("u1", "I am happy about work", "nice", 0.9, "happy")
        self.mem.update_from_interaction("u1", "still happy", "nice", 0.9, "happy")
        s = self.mem.get_user_summary("u1")
        assert s["dominant_emotion"] == "happy"
        assert s["total_interactions"] == 2

    def test_get_relationship_context_few_interactions(self):
        ctx = self.mem.get_relationship_context("u1", "neutral")
        assert "lately" not in ctx

    def test_get_relationship_context_many_interactions(self):
        for _ in range(6):
            self.mem.update_from_interaction("u1", "message", "reply", 0.0, "happy")
        ctx = self.mem.get_relationship_context("u1", "neutral")
        assert "lately" in ctx

    def test_get_relationship_context_current_emotion(self):
        ctx = self.mem.get_relationship_context("u1", "sad")
        assert "Currently feeling sad." in ctx

    def test_get_relationship_context_dissatisfied(self):
        for _ in range(5):
            self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="bad")
        ctx = self.mem.get_relationship_context("u1", "neutral")
        assert "dissatisfied" in ctx

    def test_get_relationship_context_satisfied(self):
        for _ in range(5):
            self.mem.update_from_interaction("u1", "hi", "hey", 0.0, "neutral", feedback="good")
        ctx = self.mem.get_relationship_context("u1", "neutral")
        assert "happy" in ctx


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------

class TestSessionMemory:
    def test_init(self):
        s = SessionMemory(max_turns=5)
        assert s.max_turns == 5
        assert s.conversation == []
        assert s.session_id.startswith("session_")
        assert s.session_start

    def test_add_message(self):
        s = SessionMemory()
        m = s.add("user", "hello")
        assert m["role"] == "user"
        assert m["content"] == "hello"
        assert m["turn"] == 0
        assert len(s.conversation) == 1

    def test_max_turns_trimming(self):
        s = SessionMemory(max_turns=3)
        for i in range(5):
            s.add("user", f"msg{i}")
        assert len(s.conversation) == 3
        assert s.conversation[0]["content"] == "msg2"

    def test_get_context(self):
        s = SessionMemory()
        for i in range(5):
            s.add("user", f"msg{i}")
        ctx = s.get_context(2)
        assert [c["content"] for c in ctx] == ["msg3", "msg4"]

    def test_get_full_session_copy(self):
        s = SessionMemory()
        s.add("user", "hi")
        full = s.get_full_session()
        assert len(full) == 1
        full.append({"role": "x", "content": "y", "timestamp": "", "turn": 1})
        assert len(s.conversation) == 1

    def test_clear(self):
        s = SessionMemory()
        s.add("user", "hi")
        old_id = s.session_id
        s.clear()
        assert s.conversation == []
        assert s.session_id != old_id
        assert s.session_start

    def test_get_summary(self):
        s = SessionMemory()
        s.add("user", "hi")
        s.add("assistant", "hello")
        summary = s.get_summary()
        assert summary["session_id"] == s.session_id
        assert summary["turns"] == 2
        assert summary["roles"]["user"] == 1
        assert summary["roles"]["assistant"] == 1


# ---------------------------------------------------------------------------
# EpisodicMemoryStore
# ---------------------------------------------------------------------------

class TestEpisodicMemoryStore:
    def test_save_episode_returns_deterministic_id(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hi"}]
        eid1 = store.save_episode("sess1", conv)
        eid2 = store.save_episode("sess1", conv)
        assert eid1 == eid2
        assert eid1 == f"conv_{hashlib.md5(b'sess1').hexdigest()[:12]}"
        assert store.episodes[eid1] == conv
        assert store.episode_metadata[eid1]["turns"] == 1

    def test_importance_empty(self):
        store = EpisodicMemoryStore()
        assert store._calculate_importance([]) == 0.0

    def test_importance_baseline(self):
        store = EpisodicMemoryStore()
        assert store._calculate_importance([{"content": "x"}]) == 0.5

    def test_importance_long_conversation(self):
        store = EpisodicMemoryStore()
        conv = [{"content": "x"} for _ in range(12)]
        assert store._calculate_importance(conv) >= 0.7

    def test_importance_keywords(self):
        store = EpisodicMemoryStore()
        conv = [{"content": "remember this important critical key learn now"}]
        assert store._calculate_importance(conv) == 1.0

    def test_evict_least_important_empty(self):
        store = EpisodicMemoryStore()
        store._evict_least_important()
        assert store.episodes == {}
        assert store.episode_metadata == {}

    def test_eviction(self):
        store = EpisodicMemoryStore(max_episodes=2)
        store.save_episode("a", [{"content": "x"}])
        store.save_episode("b", [{"content": "x"}])
        store.save_episode("c", [{"content": "x"}])
        assert len(store.episodes) == 2
        assert "a" not in store.episodes

    def test_evict_removes_least_important(self):
        store = EpisodicMemoryStore(max_episodes=2)
        low_id = store.save_episode("low", [{"content": "x"}])
        high_id = store.save_episode("high", [{"content": "critical important remember key learn " * 3}])
        store.save_episode("new", [{"content": "x"}])
        assert low_id not in store.episodes
        assert high_id in store.episodes

    def test_get_episode(self):
        store = EpisodicMemoryStore()
        eid = store.save_episode("s", [{"content": "hello world"}])
        assert store.get_episode(eid)[0]["content"] == "hello world"
        assert store.get_episode("missing") is None

    def test_search_episodes(self):
        store = EpisodicMemoryStore()
        eid = store.save_episode("s", [{"content": "discussing the weather today"}])
        results = store.search_episodes("weather")
        assert len(results) == 1
        assert results[0]["episode_id"] == eid
        assert results[0]["relevance"] == 0.5
        assert results[0]["turns"] == 1

    def test_search_episodes_no_match(self):
        store = EpisodicMemoryStore()
        store.save_episode("s", [{"content": "nothing here"}])
        assert store.search_episodes("zzz") == []

    def test_search_episodes_limit(self):
        store = EpisodicMemoryStore()
        for i in range(3):
            store.save_episode(f"s{i}", [{"content": "keyword"}])
        assert len(store.search_episodes("keyword", limit=2)) == 2

    def test_get_recent_episodes_order(self):
        store = EpisodicMemoryStore()
        ids = [store.save_episode(f"s{i}", [{"content": "x"}]) for i in range(3)]
        recent = store.get_recent_episodes(2)
        assert len(recent) == 2
        assert recent[0] == ids[-1]


# ---------------------------------------------------------------------------
# CognitiveArchitecture
# ---------------------------------------------------------------------------

class TestCognitiveArchitecture:
    def test_init(self):
        arch = CognitiveArchitecture(working_capacity=5)
        assert arch.working_capacity == 5
        assert arch.sensory_buffer == []
        assert arch.working_memory == []
        assert arch.semantic_memory == {}

    def test_process_sensory(self):
        arch = CognitiveArchitecture()
        assert arch.process_sensory("sensor data") is True
        assert arch.sensory_buffer[0]["data"] == "sensor data"

    def test_process_sensory_cap(self):
        arch = CognitiveArchitecture()
        for i in range(101):
            arch.process_sensory(i)
        assert len(arch.sensory_buffer) == 50
        assert arch.sensory_buffer[0]["data"] == 51

    def test_to_working(self):
        arch = CognitiveArchitecture()
        assert arch.to_working("item") is True
        assert arch.working_memory == ["item"]

    def test_to_working_evicts_fifo(self):
        arch = CognitiveArchitecture(working_capacity=2)
        arch.to_working("a")
        arch.to_working("b")
        arch.to_working("c")
        assert arch.working_memory == ["b", "c"]

    def test_consolidate_to_episodic(self):
        arch = CognitiveArchitecture()
        assert arch._consolidate_to_episodic("item") is True

    def test_add_to_session(self):
        arch = CognitiveArchitecture()
        m = arch.add_to_session("user", "hello")
        assert m["content"] == "hello"
        assert arch.session_memory.conversation[0]["content"] == "hello"

    def test_get_session_context(self):
        arch = CognitiveArchitecture()
        arch.add_to_session("user", "a")
        arch.add_to_session("user", "b")
        ctx = arch.get_session_context(1)
        assert ctx[0]["content"] == "b"

    def test_save_session_as_episode(self):
        arch = CognitiveArchitecture()
        arch.add_to_session("user", "hello")
        eid = arch.save_session_as_episode()
        assert eid in arch.episodic_store.episodes

    def test_recall_episodes(self):
        arch = CognitiveArchitecture()
        arch.add_to_session("user", "I remember the rain")
        arch.save_session_as_episode()
        results = arch.recall_episodes("rain")
        assert len(results) == 1

    def test_to_semantic_new(self):
        arch = CognitiveArchitecture()
        assert arch.to_semantic("math", "2+2=4") is True
        assert arch.semantic_memory["math"]["value"] == "2+2=4"
        assert arch.semantic_memory["math"]["strength"] == 1.0

    def test_to_semantic_strengthen(self):
        arch = CognitiveArchitecture()
        arch.to_semantic("math", "x")
        arch.to_semantic("math", "x")
        assert arch.semantic_memory["math"]["strength"] == pytest.approx(1.1)

    def test_retrieve_semantic(self):
        arch = CognitiveArchitecture()
        arch.to_semantic("math", "2+2=4")
        assert arch.retrieve_semantic("math") == "2+2=4"
        assert "last_accessed" in arch.semantic_memory["math"]

    def test_retrieve_semantic_missing(self):
        arch = CognitiveArchitecture()
        assert arch.retrieve_semantic("nope") is None


# ---------------------------------------------------------------------------
# NeuralPlasticityEngine
# ---------------------------------------------------------------------------

class TestNeuralPlasticityEngine:
    def test_init(self):
        eng = NeuralPlasticityEngine(learning_rate=0.02)
        assert eng.learning_rate == 0.02
        assert eng.connections == {}
        assert eng.activation_history == {}

    def test_activate(self):
        eng = NeuralPlasticityEngine()
        eng.activate("n1", 0.5)
        assert eng.activation_history["n1"] == [0.5]

    def test_activation_history_cap(self):
        eng = NeuralPlasticityEngine()
        for i in range(101):
            eng.activate("n1", 1.0)
        assert len(eng.activation_history["n1"]) == 50

    def test_activation_history_bounded(self):
        eng = NeuralPlasticityEngine()
        for i in range(120):
            eng.activate("n1", 1.0)
        assert len(eng.activation_history["n1"]) <= 100

    def test_hebbian_learn_default_strength(self):
        eng = NeuralPlasticityEngine(learning_rate=0.1)
        w = eng.hebbian_learn("pre", "post", reward=2.0)
        assert w == pytest.approx(0.1 * 1.0 * 1.0 * 2.0)

    def test_hebbian_learn_uses_history(self):
        eng = NeuralPlasticityEngine(learning_rate=0.1)
        eng.activate("pre", 2.0)
        eng.activate("post", 3.0)
        w = eng.hebbian_learn("pre", "post")
        assert w == pytest.approx(0.1 * 2.0 * 3.0)

    def test_hebbian_accumulates(self):
        eng = NeuralPlasticityEngine(learning_rate=0.1)
        eng.hebbian_learn("a", "b")
        w2 = eng.hebbian_learn("a", "b")
        assert w2 == pytest.approx(0.2)

    def test_get_connection_strength_default_zero(self):
        eng = NeuralPlasticityEngine()
        assert eng.get_connection_strength("a", "b") == 0.0

    def test_prune_weak_connections(self):
        eng = NeuralPlasticityEngine(learning_rate=0.1)
        eng.hebbian_learn("a", "b")  # delta 0.1, above threshold
        eng.connections["x"]["y"] = 0.001
        eng.connections["x"]["z"] = 0.5
        pruned = eng.prune_weak_connections(threshold=0.05)
        assert pruned == 1
        assert "y" not in eng.connections["x"]
        assert "z" in eng.connections["x"]


# ---------------------------------------------------------------------------
# MetaLearningEngine
# ---------------------------------------------------------------------------

class TestMetaLearningEngine:
    def test_init(self):
        eng = MetaLearningEngine()
        assert set(eng.strategies.keys()) == {"rote", "spaced", "interleaved", "elaborative"}
        assert eng.best_strategy == "spaced"

    def test_record_outcome_known(self):
        eng = MetaLearningEngine()
        eng.record_outcome("spaced", True)
        assert eng.strategies["spaced"]["attempts"] == 1
        assert eng.strategies["spaced"]["success"] == 1

    def test_record_outcome_failure(self):
        eng = MetaLearningEngine()
        eng.record_outcome("spaced", False)
        assert eng.strategies["spaced"]["attempts"] == 1
        assert eng.strategies["spaced"]["success"] == 0

    def test_record_outcome_unknown_ignored(self):
        eng = MetaLearningEngine()
        eng.record_outcome("bogus", True)
        assert "bogus" not in eng.strategies

    def test_update_weights(self):
        eng = MetaLearningEngine()
        eng.record_outcome("spaced", True)
        eng.update_weights()
        assert eng.strategies["spaced"]["weight"] == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)

    def test_update_weights_no_attempts_unchanged(self):
        eng = MetaLearningEngine()
        eng.update_weights()
        assert eng.strategies["rote"]["weight"] == 1.0

    def test_best_strategy_updates(self):
        eng = MetaLearningEngine()
        for _ in range(10):
            eng.record_outcome("elaborative", True)
        for name in ("rote", "spaced", "interleaved"):
            eng.record_outcome(name, False)
        eng.update_weights()
        assert eng.get_strategy() == "elaborative"


# ---------------------------------------------------------------------------
# DreamProcessingEngine
# ---------------------------------------------------------------------------

class TestDreamProcessingEngine:
    def test_init(self):
        eng = DreamProcessingEngine()
        assert eng.dream_cycles == 0
        assert eng.consolidated == 0

    def test_dream_empty_memories(self):
        eng = DreamProcessingEngine()
        insights = eng.dream([], NeuralPlasticityEngine())
        assert insights == []
        assert eng.dream_cycles == 1
        assert eng.consolidated == 0

    def test_dream_single_memory_no_insight(self):
        eng = DreamProcessingEngine()
        insights = eng.dream([FakeMemory("m1", 0.9)], NeuralPlasticityEngine())
        assert insights == []
        assert eng.consolidated == 1

    def test_dream_three_memories_generates_insight(self):
        eng = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [FakeMemory(f"m{i}", 1.0) for i in range(3)]
        insights = eng.dream(memories, plasticity)
        assert "Pattern detected across 3 memories" in insights
        assert eng.consolidated == 3

    def test_dream_strengthens_connections(self):
        eng = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [FakeMemory(f"m{i}", 1.0) for i in range(3)]
        eng.dream(memories, plasticity)
        assert plasticity.get_connection_strength("m0", "m1") > 0

    def test_dream_sorts_by_importance(self):
        eng = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        memories = [FakeMemory(f"m{i}", float(i)) for i in range(5)]
        eng.dream(memories, plasticity)
        assert eng.consolidated == 5
        assert plasticity.get_connection_strength("m4", "m3") > 0
