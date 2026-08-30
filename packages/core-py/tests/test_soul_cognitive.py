"""Tests for domains.soul.cognitive — SentimentAnalyzer, EmotionalResponseGenerator,
RelationshipMemory, CognitiveArchitecture, NeuralPlasticityEngine,
MetaLearningEngine, DreamProcessingEngine."""

import pytest
from dataclasses import dataclass
from domains.soul.cognitive import (
    SentimentAnalyzer,
    EmotionalResponseGenerator,
    RelationshipMemory,
    SessionMemory,
    EpisodicMemoryStore,
    CognitiveArchitecture,
    NeuralPlasticityEngine,
    MetaLearningEngine,
    DreamProcessingEngine,
)


# ---------------------------------------------------------------------------
# Lightweight stand-in for Experience (foundation module may not exist)
# ---------------------------------------------------------------------------

@dataclass
class _Experience:
    id: str
    content: str = ""
    importance: float = 0.5


# ===================================================================
# SentimentAnalyzer
# ===================================================================

class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        score = self.analyzer.analyze_sentiment("I love this amazing thing")
        assert score > 0

    def test_negative_sentiment(self):
        score = self.analyzer.analyze_sentiment("I hate this terrible thing")
        assert score < 0

    def test_neutral_sentiment(self):
        score = self.analyzer.analyze_sentiment("the table is round")
        assert score == 0.0

    def test_sentiment_range(self):
        for text in ["happy", "sad", "neutral", "amazing terrible", ""]:
            score = self.analyzer.analyze_sentiment(text)
            assert -1.0 <= score <= 1.0

    def test_detect_happy(self):
        assert self.analyzer.detect_emotion("I am so happy today") == "happy"

    def test_detect_sad(self):
        assert self.analyzer.detect_emotion("I feel so sad and depressed") == "sad"

    def test_detect_angry(self):
        assert self.analyzer.detect_emotion("I am furious and angry") == "angry"

    def test_detect_fear(self):
        assert self.analyzer.detect_emotion("I am scared and anxious") == "fear"

    def test_detect_surprise(self):
        assert self.analyzer.detect_emotion("Wow that is unbelievable") == "surprise"

    def test_detect_neutral(self):
        assert self.analyzer.detect_emotion("the sky is blue") == "neutral"

    def test_analyze_returns_all_fields(self):
        result = self.analyzer.analyze("I am happy and excited")
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result
        assert "is_positive" in result
        assert "is_negative" in result
        assert "is_neutral" in result

    def test_analyze_intensity_is_abs_sentiment(self):
        result = self.analyzer.analyze("I hate everything")
        assert result["intensity"] == abs(result["sentiment"])

    def test_analyze_is_positive_flag(self):
        result = self.analyzer.analyze("I love this")
        assert result["is_positive"] is True
        assert result["is_negative"] is False

    def test_analyze_is_negative_flag(self):
        result = self.analyzer.analyze("I hate this")
        assert result["is_negative"] is True
        assert result["is_positive"] is False

    def test_analyze_is_neutral_flag(self):
        result = self.analyzer.analyze("the table is round")
        assert result["is_neutral"] is True

    def test_mixed_words_neutralize(self):
        score = self.analyzer.analyze_sentiment("good bad")
        assert score == 0.0


# ===================================================================
# EmotionalResponseGenerator
# ===================================================================

class TestEmotionalResponseGenerator:
    def setup_method(self):
        self.gen = EmotionalResponseGenerator()

    def test_empathetic_response_known_emotion(self):
        resp = self.gen.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_empathetic_response_unknown_emotion_falls_back(self):
        resp = self.gen.generate_empathetic_response("confusion", 0.0)
        assert isinstance(resp, str)

    def test_adapt_response_positive(self):
        adapted = self.gen.adapt_response("Great", "happy", 0.9)
        assert adapted.endswith("! 😊")

    def test_adapt_response_negative(self):
        adapted = self.gen.adapt_response("Sorry", "sad", -0.8)
        assert adapted.endswith(" 😔")

    def test_adapt_response_neutral(self):
        adapted = self.gen.adapt_response("Okay", "neutral", 0.0)
        assert adapted == "Okay"

    def test_format_with_empathy_non_neutral(self):
        result = self.gen.format_emotional_response("Thanks", "happy", 0.6, include_empathy=True)
        assert "Thanks" in result
        assert len(result) > len("Thanks")

    def test_format_neutral_no_empathy(self):
        result = self.gen.format_emotional_response("Thanks", "neutral", 0.0, include_empathy=True)
        assert result == "Thanks"

    def test_format_without_empathy(self):
        result = self.gen.format_emotional_response("Thanks", "happy", 0.6, include_empathy=False)
        # No empathy prepended, but adapt_response still applies emoji for high sentiment
        assert result == "Thanks! 😊"

    def test_all_emotions_have_responses(self):
        for emotion in ["happy", "sad", "angry", "fear", "surprise", "neutral"]:
            resp = self.gen.generate_empathetic_response(emotion, 0.5)
            assert isinstance(resp, str)
            assert len(resp) > 0


# ===================================================================
# RelationshipMemory
# ===================================================================

class TestRelationshipMemory:
    def setup_method(self):
        self.rm = RelationshipMemory()

    def test_get_creates_profile(self):
        profile = self.rm.get_user_profile("u1")
        assert profile["user_id"] == "u1"
        assert profile["total_interactions"] == 0

    def test_get_existing_profile(self):
        p1 = self.rm.get_user_profile("u1")
        p1["total_interactions"] = 5
        p2 = self.rm.get_user_profile("u1")
        assert p2["total_interactions"] == 5

    def test_update_increments_interactions(self):
        self.rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        assert self.rm.get_user_profile("u1")["total_interactions"] == 1

    def test_update_tracks_emotion(self):
        self.rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy")
        profile = self.rm.get_user_profile("u1")
        assert profile["emotional_tendencies"]["happy"] == 1

    def test_update_tracks_topics(self):
        self.rm.update_from_interaction(
            "u1", "programming is interesting", "tell me more", 0.0, "neutral"
        )
        profile = self.rm.get_user_profile("u1")
        topics = profile["topics_of_interest"]
        assert len(topics) > 0

    def test_update_good_feedback(self):
        self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="good")
        profile = self.rm.get_user_profile("u1")
        assert profile["satisfaction_score"] > 0.5

    def test_update_bad_feedback(self):
        self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="bad")
        profile = self.rm.get_user_profile("u1")
        assert profile["satisfaction_score"] < 0.5

    def test_satisfaction_clamped_at_1(self):
        for _ in range(20):
            self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="good")
        assert self.rm.get_user_profile("u1")["satisfaction_score"] <= 1.0

    def test_satisfaction_clamped_at_0(self):
        for _ in range(20):
            self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="bad")
        assert self.rm.get_user_profile("u1")["satisfaction_score"] >= 0.0

    def test_mood_history_capped_at_50(self):
        for i in range(60):
            self.rm.update_from_interaction("u1", f"msg {i}", "ok", 0.0, "neutral")
        profile = self.rm.get_user_profile("u1")
        assert len(profile["mood_history"]) <= 50

    def test_interaction_history_capped_at_100(self):
        for i in range(110):
            self.rm.update_from_interaction("u1", f"msg {i}", "ok", 0.0, "neutral")
        assert len(self.rm.interaction_history["u1"]) <= 100

    def test_user_summary(self):
        self.rm.update_from_interaction("u1", "hello there", "hi", 0.3, "happy")
        summary = self.rm.get_user_summary("u1")
        assert summary["user_id"] == "u1"
        assert summary["total_interactions"] == 1
        assert summary["dominant_emotion"] == "happy"

    def test_relationship_context_low_satisfaction(self):
        for _ in range(5):
            self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="bad")
        ctx = self.rm.get_relationship_context("u1", "neutral")
        assert "dissatisfied" in ctx.lower()

    def test_relationship_context_high_satisfaction(self):
        for _ in range(5):
            self.rm.update_from_interaction("u1", "hello", "hi", 0.0, "neutral", feedback="good")
        ctx = self.rm.get_relationship_context("u1", "neutral")
        assert "happy" in ctx.lower() or "positive" in ctx.lower()

    def test_relationship_context_current_emotion(self):
        ctx = self.rm.get_relationship_context("u1", "sad")
        assert "sad" in ctx.lower()

    def test_relationship_context_new_user(self):
        ctx = self.rm.get_relationship_context("u1", "neutral")
        assert isinstance(ctx, str)


# ===================================================================
# CognitiveArchitecture
# ===================================================================

class TestCognitiveArchitecture:
    def setup_method(self):
        self.arch = CognitiveArchitecture(working_capacity=5)

    def test_sensory_processing(self):
        result = self.arch.process_sensory("input1")
        assert result is True
        assert len(self.arch.sensory_buffer) == 1

    def test_sensory_buffer_capped(self):
        for i in range(120):
            self.arch.process_sensory(f"input{i}")
        assert len(self.arch.sensory_buffer) <= 100

    def test_to_working_memory(self):
        self.arch.to_working("item1")
        assert "item1" in self.arch.working_memory

    def test_working_memory_eviction(self):
        for i in range(7):
            self.arch.to_working(f"item{i}")
        assert len(self.arch.working_memory) == 5

    def test_semantic_memory_store(self):
        self.arch.to_semantic("fact1", "water boils at 100C")
        assert self.arch.retrieve_semantic("fact1") == "water boils at 100C"

    def test_semantic_memory_strengthen(self):
        self.arch.to_semantic("fact1", "value")
        self.arch.to_semantic("fact1", "value")
        assert self.arch.semantic_memory["fact1"]["strength"] > 1.0

    def test_semantic_memory_miss(self):
        assert self.arch.retrieve_semantic("nonexistent") is None

    def test_session_memory_integration(self):
        msg = self.arch.add_to_session("user", "hello")
        assert msg["role"] == "user"
        ctx = self.arch.get_session_context(1)
        assert len(ctx) == 1

    def test_save_session_as_episode(self):
        self.arch.add_to_session("user", "hello")
        self.arch.add_to_session("assistant", "hi")
        ep_id = self.arch.save_session_as_episode()
        assert ep_id is not None
        assert ep_id in self.arch.episodic_store.episodes

    def test_recall_episodes(self):
        self.arch.add_to_session("user", "remember this important thing")
        self.arch.save_session_as_episode()
        results = self.arch.recall_episodes("important")
        assert len(results) >= 1


# ===================================================================
# NeuralPlasticityEngine
# ===================================================================

class TestNeuralPlasticityEngine:
    def setup_method(self):
        self.npe = NeuralPlasticityEngine(learning_rate=0.1)

    def test_activation_recorded(self):
        self.npe.activate("A", 1.0)
        assert 1.0 in self.npe.activation_history["A"]

    def test_activation_history_capped(self):
        for i in range(120):
            self.npe.activate("A", float(i))
        assert len(self.npe.activation_history["A"]) <= 100

    def test_hebbian_learn_strengthens_connection(self):
        self.npe.activate("A", 1.0)
        self.npe.activate("B", 1.0)
        weight = self.npe.hebbian_learn("A", "B")
        assert weight > 0

    def test_hebbian_learn_accumulates(self):
        self.npe.activate("A", 1.0)
        self.npe.activate("B", 1.0)
        w1 = self.npe.hebbian_learn("A", "B")
        w2 = self.npe.hebbian_learn("A", "B")
        assert w2 > w1

    def test_get_connection_strength(self):
        self.npe.activate("A", 1.0)
        self.npe.activate("B", 1.0)
        self.npe.hebbian_learn("A", "B")
        s = self.npe.get_connection_strength("A", "B")
        assert s > 0

    def test_prune_weak_connections(self):
        self.npe.connections["X"]["Y"] = 0.001
        self.npe.connections["A"]["B"] = 0.5
        pruned = self.npe.prune_weak_connections(threshold=0.01)
        assert pruned >= 1
        assert self.npe.get_connection_strength("A", "B") == 0.5

    def test_reward_modulates_delta(self):
        self.npe.activate("A", 1.0)
        self.npe.activate("B", 1.0)
        w_low = self.npe.hebbian_learn("A", "B", reward=0.5)
        self.npe.connections["A"]["B"] = 0.0
        w_high = self.npe.hebbian_learn("A", "B", reward=2.0)
        assert w_high > w_low


# ===================================================================
# MetaLearningEngine
# ===================================================================

class TestMetaLearningEngine:
    def setup_method(self):
        self.mle = MetaLearningEngine()

    def test_initial_best_strategy(self):
        assert self.mle.get_strategy() == "spaced"

    def test_record_outcome(self):
        self.mle.record_outcome("rote", True)
        assert self.mle.strategies["rote"]["attempts"] == 1
        assert self.mle.strategies["rote"]["success"] == 1

    def test_record_outcome_unknown_strategy(self):
        self.mle.record_outcome("nonexistent", True)
        assert "nonexistent" not in self.mle.strategies

    def test_update_weights_changes_best(self):
        for _ in range(10):
            self.mle.record_outcome("rote", True)
            self.mle.record_outcome("spaced", False)
        self.mle.update_weights()
        assert self.mle.best_strategy == "rote"

    def test_weight_update_formula(self):
        self.mle.record_outcome("spaced", True)
        self.mle.record_outcome("spaced", False)
        old_weight = self.mle.strategies["spaced"]["weight"]
        self.mle.update_weights()
        new_weight = self.mle.strategies["spaced"]["weight"]
        # 0.7 * 1.0 + 0.3 * 0.5 = 0.85, so weight decreases
        assert new_weight < old_weight

    def test_no_attempts_no_weight_change(self):
        old = self.mle.strategies["spaced"]["weight"]
        self.mle.update_weights()
        assert self.mle.strategies["spaced"]["weight"] == old


# ===================================================================
# DreamProcessingEngine
# ===================================================================

class TestDreamProcessingEngine:
    def setup_method(self):
        self.dream_engine = DreamProcessingEngine()
        self.plasticity = NeuralPlasticityEngine()

    def test_dream_increments_cycle(self):
        assert self.dream_engine.dream_cycles == 0
        self.dream_engine.dream([], self.plasticity)
        assert self.dream_engine.dream_cycles == 1

    def test_dream_empty_memories(self):
        insights = self.dream_engine.dream([], self.plasticity)
        assert insights == []

    def test_dream_few_memories_no_insight(self):
        memories = [_Experience(id=f"e{i}", importance=0.5) for i in range(2)]
        insights = self.dream_engine.dream(memories, self.plasticity)
        assert insights == []

    def test_dream_many_memories_generates_insight(self):
        memories = [_Experience(id=f"e{i}", importance=0.9) for i in range(5)]
        insights = self.dream_engine.dream(memories, self.plasticity)
        assert len(insights) >= 1
        assert "Pattern detected" in insights[0]

    def test_dream_consolidates_memories(self):
        memories = [_Experience(id=f"e{i}", importance=0.8) for i in range(3)]
        self.dream_engine.dream(memories, self.plasticity)
        assert self.dream_engine.consolidated == 3

    def test_dream_replays_top_10(self):
        memories = [_Experience(id=f"e{i}", importance=i / 10.0) for i in range(15)]
        self.dream_engine.dream(memories, self.plasticity)
        assert self.dream_engine.consolidated == 10

    def test_dream_builds_plasticity_connections(self):
        memories = [_Experience(id=f"e{i}", importance=0.9) for i in range(4)]
        self.dream_engine.dream(memories, self.plasticity)
        assert self.plasticity.get_connection_strength("e0", "e1") > 0

    def test_dream_multiple_cycles(self):
        memories = [_Experience(id="e1", importance=1.0)]
        self.dream_engine.dream(memories, self.plasticity)
        self.dream_engine.dream(memories, self.plasticity)
        assert self.dream_engine.dream_cycles == 2
        assert self.dream_engine.consolidated == 2
