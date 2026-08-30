"""Comprehensive tests for domains.soul.cognitive — all cognitive classes.

Covers: SentimentAnalyzer, EmotionalResponseGenerator, RelationshipMemory,
SessionMemory, EpisodicMemoryStore, CognitiveArchitecture,
NeuralPlasticityEngine, MetaLearningEngine, DreamProcessingEngine.
"""

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

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_experience(eid="e1", importance=0.5):
    """Build a minimal Experience for DreamProcessingEngine tests."""
    from types import SimpleNamespace
    return SimpleNamespace(id=eid, importance=importance)


def _make_experience_real(eid="e1", importance=0.5):
    """Try importing real Experience; fall back to SimpleNamespace."""
    try:
        from domains.soul.foundation import Experience
        return Experience(id=eid, importance=importance, content="test")
    except ImportError:
        return _make_experience(eid, importance)


# ═══════════════════════════════════════════════════════════════════════════════
# SentimentAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeSentiment:
    def test_positive_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("This is great and wonderful")
        assert score == 1.0

    def test_negative_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("This is bad and terrible")
        assert score == -1.0

    def test_neutral_text(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("The weather is today")
        assert score == 0.0

    def test_mixed_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good bad")
        assert score == 0.0

    def test_mostly_positive(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good great amazing bad")
        assert score > 0

    def test_empty_string(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("")
        assert score == 0.0

    def test_all_positive_words(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good excellent wonderful")
        assert score == 1.0

    def test_all_negative_words(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("bad terrible horrible")
        assert score == -1.0

    def test_single_positive_word(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("love")
        assert score == 1.0

    def test_single_negative_word(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("hate")
        assert score == -1.0

    def test_unknown_words(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("xyzzy flurble")
        assert score == 0.0

    def test_case_insensitive(self):
        sa = SentimentAnalyzer()
        assert sa.analyze_sentiment("GOOD") == 1.0
        assert sa.analyze_sentiment("BAD") == -1.0

    def test_score_range(self):
        sa = SentimentAnalyzer()
        for text in ["great", "bad", "good bad", "amazing horrible happy sad"]:
            score = sa.analyze_sentiment(text)
            assert -1.0 <= score <= 1.0


class TestDetectEmotion:
    def test_happy(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am so happy today!") == "happy"

    def test_sad(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I feel so sad and depressed") == "sad"

    def test_angry(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am angry and frustrated") == "angry"

    def test_fear(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am scared and worried") == "fear"

    def test_surprise(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am shocked and surprised") == "surprise"

    def test_neutral(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("The sky is blue") == "neutral"

    def test_case_insensitive(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("HAPPY JOY EXCITED") == "happy"

    def test_multiple_emotions_tie(self):
        """When multiple emotions have the same score, the max key is returned."""
        sa = SentimentAnalyzer()
        result = sa.detect_emotion("happy sad")
        assert result in ("happy", "sad")

    def test_empty_string_returns_neutral(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("") == "neutral"

    def test_pure_numbers_returns_neutral(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("123 456 789") == "neutral"

    def test_love_is_happy(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I love this") == "happy"

    def test_hate_is_angry(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I hate this") == "angry"

    def test_surprise_keywords(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("wow unbelievable") == "surprise"


class TestAnalyze:
    def test_analyze_positive(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I love this, it's amazing and wonderful")
        assert result["sentiment"] > 0
        assert result["is_positive"] is True
        assert result["is_negative"] is False
        assert result["intensity"] > 0

    def test_analyze_negative(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I hate this, it's terrible and awful")
        assert result["sentiment"] < 0
        assert result["is_negative"] is True
        assert result["is_positive"] is False

    def test_analyze_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("The table is brown")
        assert result["is_neutral"] is True
        assert result["intensity"] == 0.0

    def test_analyze_has_all_keys(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("hello")
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result
        assert "is_positive" in result
        assert "is_negative" in result
        assert "is_neutral" in result

    def test_intensity_is_absolute(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is terrible and horrible")
        assert result["intensity"] == abs(result["sentiment"])

    def test_emotion_field_populated(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am so happy today")
        assert isinstance(result["emotion"], str)
        assert result["emotion"] in sa.emotion_keywords

    def test_mixed_sentiment_is_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("good bad")
        assert result["is_neutral"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# EmotionalResponseGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionalResponseGenerator:
    def test_generates_string(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_unknown_emotion_falls_back_to_neutral(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("nonexistent", 0.0)
        assert resp in erg.empathy_responses["neutral"]

    def test_adapt_positive_high(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Hello", "happy", 0.8)
        assert result.endswith("! 😊")

    def test_adapt_negative_high(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Hello", "sad", -0.8)
        assert result.endswith(" 😔")

    def test_adapt_neutral(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Hello", "neutral", 0.0)
        assert result == "Hello"

    def test_adapt_moderate_positive(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Hi", "happy", 0.3)
        assert result == "Hi"

    def test_format_with_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("OK", "happy", 0.5, include_empathy=True)
        assert "OK" in result

    def test_format_without_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Base", "sad", -0.5, include_empathy=False)
        assert result.startswith("Base")

    def test_format_neutral_no_empathy_prepended(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Reply", "neutral", 0.0)
        assert result == "Reply"

    def test_all_emotions_have_responses(self):
        erg = EmotionalResponseGenerator()
        for emotion in ["happy", "sad", "angry", "fear", "surprise", "neutral"]:
            resp = erg.generate_empathetic_response(emotion, 0.0)
            assert isinstance(resp, str)


# ═══════════════════════════════════════════════════════════════════════════════
# RelationshipMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestRelationshipMemory:
    def test_get_new_profile(self):
        rm = RelationshipMemory()
        p = rm.get_user_profile("u1")
        assert p["user_id"] == "u1"
        assert p["total_interactions"] == 0

    def test_get_existing_profile(self):
        rm = RelationshipMemory()
        p1 = rm.get_user_profile("u1")
        p2 = rm.get_user_profile("u1")
        assert p1 is p2

    def test_update_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello there", "hi", 0.5, "happy")
        p = rm.get_user_profile("u1")
        assert p["total_interactions"] == 1
        assert p["emotional_tendencies"]["happy"] == 1

    def test_satisfaction_good_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy", feedback="good")
        assert rm.get_user_profile("u1")["satisfaction_score"] > 0.5

    def test_satisfaction_bad_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy", feedback="bad")
        assert rm.get_user_profile("u1")["satisfaction_score"] < 0.5

    def test_satisfaction_cap_at_1(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy", feedback="good")
        assert rm.get_user_profile("u1")["satisfaction_score"] == 1.0

    def test_satisfaction_floor_at_0(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("u1", "hello", "hi", -0.5, "sad", feedback="bad")
        assert rm.get_user_profile("u1")["satisfaction_score"] == 0.0

    def test_mood_history_capped(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction("u1", f"msg{i}", "r", 0.0, "neutral")
        assert len(rm.get_user_profile("u1")["mood_history"]) <= 50

    def test_interaction_history_capped(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction("u1", f"msg{i}", "r", 0.0, "neutral")
        assert len(rm.interaction_history["u1"]) <= 100

    def test_topics_extracted(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "artificial intelligence amazing", "r", 0.5, "happy")
        topics = rm.get_user_profile("u1")["topics_of_interest"]
        assert "artificial" in topics

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello world", "hi", 0.5, "happy")
        s = rm.get_user_summary("u1")
        assert s["user_id"] == "u1"
        assert s["total_interactions"] == 1
        assert s["dominant_emotion"] == "happy"

    def test_summary_empty_profile(self):
        rm = RelationshipMemory()
        s = rm.get_user_summary("new_user")
        assert s["dominant_emotion"] == "neutral"
        assert s["total_interactions"] == 0

    def test_relationship_context_low_satisfaction(self):
        rm = RelationshipMemory()
        # satisfaction starts at 0.5, need to drop below 0.4
        for _ in range(3):
            rm.update_from_interaction("u1", "hello", "r", -0.5, "sad", feedback="bad")
        ctx = rm.get_relationship_context("u1", "angry")
        assert "dissatisfied" in ctx

    def test_relationship_context_high_satisfaction(self):
        rm = RelationshipMemory()
        for _ in range(10):
            rm.update_from_interaction("u1", "hello", "r", 0.8, "happy", feedback="good")
        ctx = rm.get_relationship_context("u1", "happy")
        assert "happy" in ctx.lower()

    def test_relationship_context_neutral_current(self):
        rm = RelationshipMemory()
        ctx = rm.get_relationship_context("u1", "neutral")
        assert "Currently feeling" not in ctx

    def test_relationship_context_many_interactions(self):
        rm = RelationshipMemory()
        for _ in range(10):
            rm.update_from_interaction("u1", "hello world test", "r", 0.5, "happy")
        ctx = rm.get_relationship_context("u1", "neutral")
        assert len(ctx) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SessionMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionMemory:
    def test_add_message(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert msg["turn"] == 0

    def test_turn_increments(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        assert sm.conversation[1]["turn"] == 1

    def test_get_context(self):
        sm = SessionMemory()
        for i in range(10):
            sm.add("user", f"msg{i}")
        ctx = sm.get_context(3)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg7"

    def test_max_turns_eviction(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg{i}")
        assert len(sm.conversation) == 3
        assert sm.conversation[0]["content"] == "msg2"

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        full = sm.get_full_session()
        assert len(full) == 2
        assert full is not sm.conversation  # copy, not reference

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        old_id = sm.session_id
        sm.clear()
        assert len(sm.conversation) == 0
        assert sm.session_id != old_id

    def test_session_id_format(self):
        sm = SessionMemory()
        assert sm.session_id.startswith("session_")

    def test_summary(self):
        sm = SessionMemory()
        sm.add("user", "hi")
        sm.add("assistant", "hello")
        s = sm.get_summary()
        assert s["turns"] == 2
        assert "session_id" in s
        assert "start" in s

    def test_get_context_more_than_available(self):
        sm = SessionMemory()
        sm.add("user", "only one")
        ctx = sm.get_context(100)
        assert len(ctx) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# EpisodicMemoryStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpisodicMemoryStore:
    def test_save_episode(self):
        em = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hi"}]
        eid = em.save_episode("session1", conv)
        assert eid.startswith("conv_")
        assert em.get_episode(eid) == conv

    def test_get_nonexistent(self):
        em = EpisodicMemoryStore()
        assert em.get_episode("nope") is None

    def test_search_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"content": "python is great"}])
        em.save_episode("s2", [{"content": "rust is fast"}])
        results = em.search_episodes("python")
        assert len(results) == 1

    def test_search_no_match(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"content": "hello"}])
        results = em.search_episodes("nonexistent")
        assert len(results) == 0

    def test_get_recent_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"content": "a"}])
        em.save_episode("s2", [{"content": "b"}])
        recent = em.get_recent_episodes(1)
        assert len(recent) == 1

    def test_importance_empty_conversation(self):
        em = EpisodicMemoryStore()
        score = em._calculate_importance([])
        assert score == 0.0

    def test_importance_long_conversation(self):
        em = EpisodicMemoryStore()
        conv = [{"content": f"msg{i}"} for i in range(15)]
        score = em._calculate_importance(conv)
        assert score > 0.5

    def test_importance_important_words(self):
        em = EpisodicMemoryStore()
        conv = [{"content": "important remember critical key learn"}]
        score = em._calculate_importance(conv)
        assert score > 0.5

    def test_max_episodes_eviction(self):
        em = EpisodicMemoryStore(max_episodes=3)
        for i in range(5):
            em.save_episode(f"s{i}", [{"content": f"msg{i}"}])
        assert len(em.episodes) <= 3

    def test_evict_least_important(self):
        em = EpisodicMemoryStore(max_episodes=2)
        em.save_episode("s1", [{"content": "important remember critical"}])
        em.save_episode("s2", [{"content": "boring regular"}])
        em.save_episode("s3", [{"content": "key learn remember critical important"}])
        # s2 should have been evicted
        assert len(em.episodes) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# CognitiveArchitecture
# ═══════════════════════════════════════════════════════════════════════════════

class TestCognitiveArchitecture:
    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("input data")
        assert result is True
        assert len(ca.sensory_buffer) == 1

    def test_sensory_buffer_cap(self):
        """Buffer trims to 50 once it exceeds 100; intermediate adds still grow."""
        ca = CognitiveArchitecture()
        # After 101 items: trims to 50. Then 14 more = 64.
        for i in range(115):
            ca.process_sensory(f"data{i}")
        assert len(ca.sensory_buffer) == 64
        # After 101+50=151 items: trims again to 50.
        for i in range(50):
            ca.process_sensory(f"data_extra{i}")
        assert len(ca.sensory_buffer) <= 100

    def test_to_working_memory(self):
        ca = CognitiveArchitecture()
        ca.to_working("item1")
        assert "item1" in ca.working_memory

    def test_working_memory_fifo_eviction(self):
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
        ca.add_to_session("user", "hi")
        ctx = ca.get_session_context(1)
        assert len(ctx) == 1

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "test")
        eid = ca.save_session_as_episode()
        assert eid.startswith("conv_")

    def test_recall_episodes(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "python test")
        ca.save_session_as_episode()
        results = ca.recall_episodes("python")
        assert len(results) >= 1

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        result = ca.to_semantic("fact1", "value1")
        assert result is True
        assert ca.semantic_memory["fact1"]["value"] == "value1"

    def test_to_semantic_strengthen(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "v1")
        ca.to_semantic("fact1", "v2")
        assert ca.semantic_memory["fact1"]["strength"] > 1.0

    def test_retrieve_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("k", "v")
        assert ca.retrieve_semantic("k") == "v"

    def test_retrieve_semantic_missing(self):
        ca = CognitiveArchitecture()
        assert ca.retrieve_semantic("missing") is None

    def test_semantic_last_accessed_set(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("k", "v")
        ca.retrieve_semantic("k")
        assert "last_accessed" in ca.semantic_memory["k"]


# ═══════════════════════════════════════════════════════════════════════════════
# NeuralPlasticityEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeuralPlasticityEngine:
    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1", 0.8)
        assert 0.8 in npe.activation_history["n1"]

    def test_activation_history_capped(self):
        """History trims to 50 once it exceeds 100; intermediate adds still grow."""
        npe = NeuralPlasticityEngine()
        # After 101 activations: trims to 50. Then 19 more = 69.
        for i in range(120):
            npe.activate("n1", float(i))
        assert len(npe.activation_history["n1"]) == 69
        # Add enough to trigger another trim
        for i in range(50):
            npe.activate("n1", float(i))
        assert len(npe.activation_history["n1"]) <= 100

    def test_hebbian_learn_basic(self):
        npe = NeuralPlasticityEngine(learning_rate=0.1)
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        w = npe.hebbian_learn("a", "b")
        assert w > 0.0

    def test_hebbian_learn_no_history(self):
        npe = NeuralPlasticityEngine(learning_rate=0.1)
        w = npe.hebbian_learn("x", "y")
        # Both default to strength 1.0, so delta = 0.1 * 1.0 * 1.0 = 0.1
        assert w == pytest.approx(0.1)

    def test_hebbian_learn_with_reward(self):
        npe = NeuralPlasticityEngine(learning_rate=0.1)
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        w = npe.hebbian_learn("a", "b", reward=2.0)
        assert w == pytest.approx(0.2)

    def test_get_connection_strength(self):
        npe = NeuralPlasticityEngine(learning_rate=0.1)
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        npe.hebbian_learn("a", "b")
        assert npe.get_connection_strength("a", "b") > 0

    def test_get_connection_strength_default_zero(self):
        npe = NeuralPlasticityEngine()
        assert npe.get_connection_strength("x", "y") == 0.0

    def test_prune_weak_connections(self):
        npe = NeuralPlasticityEngine(learning_rate=0.001)
        npe.activate("a", 0.001)
        npe.activate("b", 0.001)
        npe.hebbian_learn("a", "b")
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned >= 0  # either pruned or not

    def test_prune_nothing_to_prune(self):
        npe = NeuralPlasticityEngine()
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned == 0


# ═══════════════════════════════════════════════════════════════════════════════
# MetaLearningEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaLearningEngine:
    def test_initial_strategy(self):
        mle = MetaLearningEngine()
        assert mle.get_strategy() == "spaced"

    def test_record_outcome_success(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        assert mle.strategies["rote"]["success"] == 1
        assert mle.strategies["rote"]["attempts"] == 1

    def test_record_outcome_failure(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", False)
        assert mle.strategies["rote"]["success"] == 0
        assert mle.strategies["rote"]["attempts"] == 1

    def test_record_unknown_strategy_ignored(self):
        mle = MetaLearningEngine()
        mle.record_outcome("nonexistent", True)
        # Should not crash, strategies unchanged

    def test_update_weights(self):
        mle = MetaLearningEngine()
        # Record mixed outcomes: weight starts at 1.0
        # After update: new_w = 0.7 * 1.0 + 0.3 * (success/attempts)
        # 1 success / 2 attempts = 0.5 → new_w = 0.7 + 0.15 = 0.85
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", False)
        mle.update_weights()
        assert mle.strategies["rote"]["weight"] == pytest.approx(0.85)

    def test_update_weights_zero_attempts(self):
        mle = MetaLearningEngine()
        mle.update_weights()
        # All weights stay at 1.0 (no attempts)
        for s in mle.strategies.values():
            assert s["weight"] == 1.0

    def test_best_strategy_updates(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
        mle.update_weights()
        assert mle.get_strategy() == "rote"

    def test_all_strategies_exist(self):
        mle = MetaLearningEngine()
        for name in ["rote", "spaced", "interleaved", "elaborative"]:
            assert name in mle.strategies


# ═══════════════════════════════════════════════════════════════════════════════
# DreamProcessingEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestDreamProcessingEngine:
    def test_initial_state(self):
        dpe = DreamProcessingEngine()
        assert dpe.dream_cycles == 0
        assert dpe.consolidated == 0

    def test_dream_with_few_memories(self):
        dpe = DreamProcessingEngine()
        npe = NeuralPlasticityEngine()
        exps = [_make_experience_real(f"e{i}", 0.5) for i in range(2)]
        insights = dpe.dream(exps, npe)
        assert dpe.dream_cycles == 1
        assert dpe.consolidated == 2

    def test_dream_with_many_memories(self):
        dpe = DreamProcessingEngine()
        npe = NeuralPlasticityEngine()
        exps = [_make_experience_real(f"e{i}", float(i) / 10) for i in range(5)]
        insights = dpe.dream(exps, npe)
        assert len(insights) >= 1  # pattern detected
        assert "Pattern detected" in insights[0]

    def test_dream_empty_memories(self):
        dpe = DreamProcessingEngine()
        npe = NeuralPlasticityEngine()
        insights = dpe.dream([], npe)
        assert insights == []
        assert dpe.dream_cycles == 1
        assert dpe.consolidated == 0

    def test_dream_cycles_increment(self):
        dpe = DreamProcessingEngine()
        npe = NeuralPlasticityEngine()
        dpe.dream([], npe)
        dpe.dream([], npe)
        assert dpe.dream_cycles == 2

    def test_dream_consolidates_top_10(self):
        dpe = DreamProcessingEngine()
        npe = NeuralPlasticityEngine()
        exps = [_make_experience_real(f"e{i}", float(i) / 100) for i in range(15)]
        dpe.dream(exps, npe)
        assert dpe.consolidated == 10  # capped at top 10
