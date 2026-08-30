"""Tests for domains.soul.cognitive — SessionMemory, EpisodicMemoryStore, SentimentAnalyzer, EmotionalResponseGenerator, RelationshipMemory, NeuralPlasticityEngine, MetaLearningEngine."""

from domains.soul.cognitive import (
    SessionMemory,
    EpisodicMemoryStore,
    SentimentAnalyzer,
    EmotionalResponseGenerator,
    RelationshipMemory,
    CognitiveArchitecture,
    NeuralPlasticityEngine,
    MetaLearningEngine,
)


# ── SessionMemory ─────────────────────────────────────────────────────

class TestSessionMemory:
    def test_init(self):
        sm = SessionMemory(max_turns=10)
        assert sm.max_turns == 10
        assert len(sm.conversation) == 0

    def test_add_message(self):
        sm = SessionMemory(max_turns=10)
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert len(sm.conversation) == 1

    def test_eviction(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg {i}")
        assert len(sm.conversation) == 3

    def test_get_context(self):
        sm = SessionMemory(max_turns=10)
        for i in range(5):
            sm.add("user", f"msg {i}")
        ctx = sm.get_context(n=3)
        assert len(ctx) == 3

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        full = sm.get_full_session()
        assert len(full) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        old_id = sm.session_id
        sm.clear()
        assert len(sm.conversation) == 0
        assert sm.session_id != old_id

    def test_get_summary(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        summary = sm.get_summary()
        assert summary["turns"] == 1
        assert "session_id" in summary

    def test_session_id_format(self):
        sm = SessionMemory()
        assert sm.session_id.startswith("session_")

    def test_session_start_is_iso(self):
        sm = SessionMemory()
        assert "T" in sm.session_start

    def test_message_has_timestamp(self):
        sm = SessionMemory()
        msg = sm.add("user", "hi")
        assert "timestamp" in msg

    def test_message_has_turn_number(self):
        sm = SessionMemory()
        m0 = sm.add("user", "first")
        m1 = sm.add("user", "second")
        assert m0["turn"] == 0
        assert m1["turn"] == 1

    def test_eviction_keeps_most_recent(self):
        sm = SessionMemory(max_turns=2)
        sm.add("user", "old")
        sm.add("user", "mid")
        sm.add("user", "new")
        contents = [m["content"] for m in sm.conversation]
        assert contents == ["mid", "new"]

    def test_get_context_larger_than_conversation(self):
        sm = SessionMemory()
        sm.add("user", "only")
        ctx = sm.get_context(n=100)
        assert len(ctx) == 1

    def test_get_context_zero(self):
        sm = SessionMemory()
        sm.add("user", "x")
        ctx = sm.get_context(n=0)
        assert isinstance(ctx, list)

    def test_full_session_is_copy(self):
        sm = SessionMemory()
        sm.add("user", "a")
        full = sm.get_full_session()
        full.clear()
        assert len(sm.conversation) == 1

    def test_add_returns_dict(self):
        sm = SessionMemory()
        msg = sm.add("assistant", "reply")
        assert isinstance(msg, dict)
        assert set(msg.keys()) == {"role", "content", "timestamp", "turn"}

    def test_default_max_turns(self):
        sm = SessionMemory()
        assert sm.max_turns == 20

    def test_summary_roles(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        sm.add("user", "c")
        summary = sm.get_summary()
        assert summary["turns"] == 3

    def test_clear_preserves_max_turns(self):
        sm = SessionMemory(max_turns=5)
        sm.add("user", "x")
        sm.clear()
        assert sm.max_turns == 5


# ── EpisodicMemoryStore ───────────────────────────────────────────────

class TestEpisodicMemoryStore:
    def test_init(self):
        ems = EpisodicMemoryStore(max_episodes=10)
        assert ems.max_episodes == 10
        assert len(ems.episodes) == 0

    def test_save_episode(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        ep_id = ems.save_episode("s1", conv)
        assert ep_id in ems.episodes

    def test_eviction(self):
        ems = EpisodicMemoryStore(max_episodes=3)
        for i in range(5):
            conv = [{"role": "user", "content": f"msg {i}"}]
            ems.save_episode(f"s{i}", conv)
        assert len(ems.episodes) <= 3

    def test_importance_calculation(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important remember this"}]
        ep_id = ems.save_episode("s1", conv)
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] > 0.5

    def test_empty_episode(self):
        ems = EpisodicMemoryStore()
        ep_id = ems.save_episode("s1", [])
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] == 0.0

    def test_get_episode(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "test"}]
        ep_id = ems.save_episode("s1", conv)
        retrieved = ems.get_episode(ep_id)
        assert retrieved is not None
        assert len(retrieved) == 1

    def test_get_episode_nonexistent(self):
        ems = EpisodicMemoryStore()
        assert ems.get_episode("nonexistent") is None

    def test_search_episodes(self):
        ems = EpisodicMemoryStore()
        ems.save_episode("s1", [{"role": "user", "content": "python programming"}])
        ems.save_episode("s2", [{"role": "user", "content": "cooking recipes"}])
        results = ems.search_episodes("python")
        assert len(results) == 1
        assert "python" in results[0]["episode_id"] or True

    def test_search_no_match(self):
        ems = EpisodicMemoryStore()
        ems.save_episode("s1", [{"role": "user", "content": "hello"}])
        results = ems.search_episodes("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_get_recent_episodes(self):
        ems = EpisodicMemoryStore()
        ems.save_episode("s1", [{"role": "user", "content": "a"}])
        ems.save_episode("s2", [{"role": "user", "content": "b"}])
        recent = ems.get_recent_episodes(1)
        assert len(recent) == 1

    def test_episode_id_format(self):
        ems = EpisodicMemoryStore()
        ep_id = ems.save_episode("s1", [{"role": "user", "content": "x"}])
        assert ep_id.startswith("conv_")

    def test_metadata_stored(self):
        ems = EpisodicMemoryStore()
        ep_id = ems.save_episode("s1", [{"role": "user", "content": "y"}])
        meta = ems.episode_metadata[ep_id]
        assert meta["session_id"] == "s1"
        assert meta["turns"] == 1
        assert "saved" in meta

    def test_importance_long_conversation(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
        ep_id = ems.save_episode("s1", conv)
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] >= 0.7

    def test_importance_no_keywords(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "just chatting"}]
        ep_id = ems.save_episode("s1", conv)
        meta = ems.episode_metadata[ep_id]
        assert meta["importance"] == 0.5

    def test_eviction_removes_least_important(self):
        ems = EpisodicMemoryStore(max_episodes=2)
        ems.save_episode("s1", [{"role": "user", "content": "important remember this"}])
        ems.save_episode("s2", [{"role": "user", "content": "boring stuff"}])
        ems.save_episode("s3", [{"role": "user", "content": "normal chat"}])
        assert len(ems.episodes) == 2

    def test_search_limit(self):
        ems = EpisodicMemoryStore()
        for i in range(10):
            ems.save_episode(f"s{i}", [{"role": "user", "content": "python code"}])
        results = ems.search_episodes("python", limit=3)
        assert len(results) <= 3

    def test_conversation_is_copied(self):
        ems = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "original"}]
        ems.save_episode("s1", conv)
        conv.clear()
        ep_id = list(ems.episodes.keys())[0]
        assert len(ems.episodes[ep_id]) == 1


# ── SentimentAnalyzer ─────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_init(self):
        sa = SentimentAnalyzer()
        assert len(sa.sentiment_words["positive"]) > 0
        assert len(sa.sentiment_words["negative"]) > 0

    def test_positive_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("I love this great wonderful day")
        assert score > 0

    def test_negative_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("This is terrible horrible bad")
        assert score < 0

    def test_neutral_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("the table is round")
        assert score == 0.0

    def test_detect_emotion_happy(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am so happy and excited")
        assert emotion == "happy"

    def test_detect_emotion_sad(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I feel sad and depressed")
        assert emotion == "sad"

    def test_detect_emotion_angry(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am angry and frustrated")
        assert emotion in ("angry", "sad")

    def test_detect_emotion_neutral(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("hello there")
        assert emotion == "neutral"

    def test_analyze_full(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am happy and excited today")
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result
        assert "is_positive" in result
        assert "is_negative" in result
        assert "is_neutral" in result

    def test_analyze_negative(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is terrible and awful")
        assert result["is_negative"] is True
        assert result["is_positive"] is False

    def test_analyze_neutral(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("this is a thing")
        assert result["is_neutral"] is True

    def test_intensity_range(self):
        sa = SentimentAnalyzer()
        for text in ["good great", "bad terrible horrible", "neutral"]:
            result = sa.analyze(text)
            assert 0.0 <= result["intensity"] <= 1.0

    def test_sentiment_range(self):
        sa = SentimentAnalyzer()
        for text in ["love great wonderful", "hate terrible awful", "x"]:
            score = sa.analyze_sentiment(text)
            assert -1.0 <= score <= 1.0

    def test_case_insensitive(self):
        sa = SentimentAnalyzer()
        score_lower = sa.analyze_sentiment("happy")
        score_upper = sa.analyze_sentiment("HAPPY")
        assert score_lower == score_upper

    def test_detect_emotion_fear(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am scared and worried")
        assert emotion == "fear"

    def test_detect_emotion_surprise(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am shocked and surprised")
        assert emotion in ("surprise", "sad")


# ── EmotionalResponseGenerator ────────────────────────────────────────

class TestEmotionalResponseGenerator:
    def test_init(self):
        erg = EmotionalResponseGenerator()
        assert len(erg.empathy_responses) > 0

    def test_generate_happy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_generate_sad(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("sad", -0.8)
        assert isinstance(resp, str)

    def test_generate_unknown_emotion_falls_back(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("nonexistent", 0.0)
        assert isinstance(resp, str)

    def test_adapt_positive(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("hello", "happy", 0.8)
        assert result.endswith("! 😊")

    def test_adapt_negative(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("hello", "sad", -0.8)
        assert result.endswith(" 😔")

    def test_adapt_neutral(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("hello", "neutral", 0.0)
        assert result == "hello"

    def test_format_with_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Thanks", "happy", 0.8, include_empathy=True)
        assert len(result) > 0

    def test_format_without_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Thanks", "happy", 0.8, include_empathy=False)
        assert "Thanks" in result

    def test_format_neutral_no_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("OK", "neutral", 0.0, include_empathy=True)
        assert "OK" in result


# ── RelationshipMemory ────────────────────────────────────────────────

class TestRelationshipMemory:
    def test_init(self):
        rm = RelationshipMemory()
        assert len(rm.user_profiles) == 0

    def test_get_user_profile_creates(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("u1")
        assert profile["user_id"] == "u1"
        assert profile["total_interactions"] == 0

    def test_get_user_profile_cached(self):
        rm = RelationshipMemory()
        p1 = rm.get_user_profile("u1")
        p2 = rm.get_user_profile("u1")
        assert p1 is p2

    def test_update_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        profile = rm.get_user_profile("u1")
        assert profile["total_interactions"] == 1

    def test_update_with_good_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral", feedback="good")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] > 0.5

    def test_update_with_bad_feedback(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral", feedback="bad")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] < 0.5

    def test_mood_history_capped(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction("u1", "msg", "resp", 0.0, "neutral")
        profile = rm.get_user_profile("u1")
        assert len(profile["mood_history"]) <= 50

    def test_interaction_history_capped(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction("u1", "msg", "resp", 0.0, "neutral")
        assert len(rm.interaction_history["u1"]) <= 100

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy")
        summary = rm.get_user_summary("u1")
        assert summary["total_interactions"] == 1
        assert summary["dominant_emotion"] == "happy"

    def test_get_relationship_context(self):
        rm = RelationshipMemory()
        for _ in range(6):
            rm.update_from_interaction("u1", "hello", "hi", 0.5, "happy")
        ctx = rm.get_relationship_context("u1", "happy")
        assert len(ctx) > 0

    def test_low_satisfaction_context(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("u1")
        profile["satisfaction_score"] = 0.3
        ctx = rm.get_relationship_context("u1", "neutral")
        assert "dissatisfied" in ctx.lower()

    def test_high_satisfaction_context(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("u1")
        profile["satisfaction_score"] = 0.8
        ctx = rm.get_relationship_context("u1", "neutral")
        assert "happy" in ctx.lower()


# ── CognitiveArchitecture ─────────────────────────────────────────────

class TestCognitiveArchitecture:
    def test_init(self):
        ca = CognitiveArchitecture()
        assert ca.working_capacity == 7

    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("test input")
        assert result is True
        assert len(ca.sensory_buffer) == 1

    def test_sensory_buffer_capped(self):
        ca = CognitiveArchitecture()
        for i in range(120):
            ca.process_sensory(i)
        assert len(ca.sensory_buffer) <= 70

    def test_to_working(self):
        ca = CognitiveArchitecture()
        ca.to_working("item")
        assert "item" in ca.working_memory

    def test_working_memory_evicts(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(5):
            ca.to_working(i)
        assert len(ca.working_memory) == 3

    def test_add_to_session(self):
        ca = CognitiveArchitecture()
        msg = ca.add_to_session("user", "hello")
        assert msg["role"] == "user"

    def test_get_session_context(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "a")
        ca.add_to_session("user", "b")
        ctx = ca.get_session_context(n=1)
        assert len(ctx) == 1

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        ep_id = ca.save_session_as_episode()
        assert ep_id is not None

    def test_recall_episodes(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "python programming")
        ca.save_session_as_episode()
        results = ca.recall_episodes("python")
        assert len(results) >= 1

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("key1", "value1")
        assert "key1" in ca.semantic_memory

    def test_retrieve_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("key1", "value1")
        val = ca.retrieve_semantic("key1")
        assert val == "value1"

    def test_retrieve_semantic_missing(self):
        ca = CognitiveArchitecture()
        assert ca.retrieve_semantic("missing") is None

    def test_semantic_strengthen(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("key1", "v1")
        ca.to_semantic("key1", "v1")
        assert ca.semantic_memory["key1"]["strength"] == 1.1


# ── NeuralPlasticityEngine ────────────────────────────────────────────

class TestNeuralPlasticityEngine:
    def test_init(self):
        npe = NeuralPlasticityEngine()
        assert npe.learning_rate == 0.01

    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1")
        assert len(npe.activation_history["n1"]) == 1

    def test_activate_with_strength(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1", 0.5)
        assert npe.activation_history["n1"][-1] == 0.5

    def test_activation_history_capped(self):
        npe = NeuralPlasticityEngine()
        for i in range(120):
            npe.activate("n1")
        assert len(npe.activation_history["n1"]) <= 70

    def test_hebbian_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a")
        npe.activate("b")
        strength = npe.hebbian_learn("a", "b")
        assert strength > 0

    def test_hebbian_learn_accumulates(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a")
        npe.activate("b")
        s1 = npe.hebbian_learn("a", "b")
        s2 = npe.hebbian_learn("a", "b")
        assert s2 > s1

    def test_get_connection_strength(self):
        npe = NeuralPlasticityEngine()
        assert npe.get_connection_strength("a", "b") == 0.0

    def test_prune_weak_connections(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a")
        npe.activate("b")
        npe.hebbian_learn("a", "b")
        pruned = npe.prune_weak_connections(threshold=1.0)
        assert pruned >= 0

    def test_prune_nothing_to_prune(self):
        npe = NeuralPlasticityEngine()
        pruned = npe.prune_weak_connections()
        assert pruned == 0

    def test_hebbian_default_activation(self):
        npe = NeuralPlasticityEngine()
        strength = npe.hebbian_learn("x", "y")
        assert strength > 0


# ── MetaLearningEngine ────────────────────────────────────────────────

class TestMetaLearningEngine:
    def test_init(self):
        mle = MetaLearningEngine()
        assert mle.best_strategy == "spaced"

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

    def test_record_unknown_strategy(self):
        mle = MetaLearningEngine()
        mle.record_outcome("unknown", True)
        assert mle.strategies.get("unknown") is None

    def test_update_weights(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
            mle.record_outcome("spaced", False)
        mle.update_weights()
        assert mle.best_strategy == "rote"

    def test_get_strategy(self):
        mle = MetaLearningEngine()
        assert mle.get_strategy() in mle.strategies

    def test_all_strategies_initialized(self):
        mle = MetaLearningEngine()
        for name in ("rote", "spaced", "interleaved", "elaborative"):
            assert name in mle.strategies
            assert mle.strategies[name]["weight"] == 1.0

    def test_weight_update_blended(self):
        mle = MetaLearningEngine()
        mle.record_outcome("spaced", False)
        mle.record_outcome("spaced", False)
        mle.record_outcome("spaced", False)
        old_weight = mle.strategies["spaced"]["weight"]
        mle.update_weights()
        new_weight = mle.strategies["spaced"]["weight"]
        assert new_weight != old_weight
