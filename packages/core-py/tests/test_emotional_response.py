"""Meaningful tests for EmotionalResponseGenerator — empathetic responses, adaptation, formatting."""

from domains.soul.cognitive import (
    EmotionalResponseGenerator,
    SentimentAnalyzer,
    RelationshipMemory,
    SessionMemory,
    EpisodicMemoryStore,
    CognitiveArchitecture,
    NeuralPlasticityEngine,
    MetaLearningEngine,
    DreamProcessingEngine,
)


# ── EmotionalResponseGenerator ────────────────────────────────────────────────


class TestGenerateEmpatheticResponse:
    def test_happy_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("happy", 0.8)
        assert len(resp) > 0
        # Should be one of the happy responses
        assert any(kw in resp.lower() for kw in ["glad", "wonderful", "happy", "great"])

    def test_sad_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("sad", -0.8)
        assert any(kw in resp.lower() for kw in ["sorry", "difficult", "tough", "care"])

    def test_unknown_emotion_falls_back_to_neutral(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("confused", 0.0)
        assert any(kw in resp.lower() for kw in ["understand", "got it", "see", "alright"])

    def test_angry_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("angry", -0.9)
        assert any(kw in resp.lower() for kw in ["frustration", "upsetting", "hear", "work through"])

    def test_fear_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("fear", -0.6)
        assert any(kw in resp.lower() for kw in ["worried", "help", "step", "alone"])

    def test_surprise_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("surprise", 0.3)
        assert any(kw in resp.lower() for kw in ["surprising", "shock", "unexpected"])

    def test_neutral_response(self):
        gen = EmotionalResponseGenerator()
        resp = gen.generate_empathetic_response("neutral", 0.0)
        assert any(kw in resp.lower() for kw in ["understand", "got it", "see", "alright"])

    def test_response_nonempty(self):
        gen = EmotionalResponseGenerator()
        for emotion in ["happy", "sad", "angry", "fear", "surprise", "neutral"]:
            resp = gen.generate_empathetic_response(emotion, 0.0)
            assert len(resp) > 0

    def test_randomness_of_responses(self):
        gen = EmotionalResponseGenerator()
        responses = {gen.generate_empathetic_response("happy", 0.8) for _ in range(20)}
        assert len(responses) > 1

    def test_sentiment_ignored_for_empathy(self):
        gen = EmotionalResponseGenerator()
        r1 = gen.generate_empathetic_response("happy", 0.9)
        r2 = gen.generate_empathetic_response("happy", 0.1)
        # Both should return happy-family responses (sentiment doesn't change the pool)
        assert any(kw in r1.lower() for kw in ["glad", "wonderful", "happy", "great"])
        assert any(kw in r2.lower() for kw in ["glad", "wonderful", "happy", "great"])


class TestAdaptResponse:
    def test_positive_sentiment_adds_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Great news", "happy", 0.8)
        assert resp.endswith("😊")

    def test_negative_sentiment_adds_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("That's bad", "sad", -0.8)
        assert resp.endswith("😔")

    def test_neutral_sentiment_no_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Ok", "neutral", 0.0)
        assert resp == "Ok"

    def test_boundary_neutral_no_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Test", "neutral", 0.3)
        assert resp == "Test"

    def test_boundary_negative_no_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Test", "sad", -0.3)
        assert resp == "Test"

    def test_strong_positive_gets_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Yes", "happy", 1.0)
        assert resp.endswith("😊")

    def test_strong_negative_gets_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("No", "sad", -1.0)
        assert resp.endswith("😔")

    def test_exactly_at_threshold_no_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Test", "happy", 0.5)
        assert resp == "Test"

    def test_just_above_threshold_gets_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Test", "happy", 0.51)
        assert resp.endswith("😊")

    def test_just_below_negative_threshold_no_emoji(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("Test", "sad", -0.49)
        assert resp == "Test"

    def test_empty_string_positive(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("", "happy", 0.8)
        assert resp == "! 😊"

    def test_empty_string_negative(self):
        gen = EmotionalResponseGenerator()
        resp = gen.adapt_response("", "sad", -0.8)
        assert resp == " 😔"


class TestFormatEmotionalResponse:
    def test_with_empathy(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Thanks", "happy", 0.8, include_empathy=True)
        # Should have empathy prefix + base response
        assert "Thanks" in resp
        assert len(resp) > len("Thanks")

    def test_without_empathy(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Thanks", "happy", 0.8, include_empathy=False)
        # Without empathy, just adapt
        assert "Thanks" in resp

    def test_neutral_no_empathy_prefix(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Ok", "neutral", 0.0, include_empathy=True)
        # Neutral should not have empathy prefix
        assert resp == "Ok"

    def test_with_empathy_and_negative(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Sorry to hear", "sad", -0.8, include_empathy=True)
        assert "Sorry to hear" in resp
        assert len(resp) > len("Sorry to hear")

    def test_with_empathy_and_high_positive(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Great", "happy", 0.9, include_empathy=True)
        assert "Great" in resp

    def test_without_empathy_neutral(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Hello", "neutral", 0.0, include_empathy=False)
        assert resp == "Hello"

    def test_empathy_for_angry(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Calm down", "angry", -0.7, include_empathy=True)
        assert "Calm down" in resp

    def test_empathy_for_fear(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("It's ok", "fear", -0.5, include_empathy=True)
        assert "It's ok" in resp

    def test_empathy_for_surprise(self):
        gen = EmotionalResponseGenerator()
        resp = gen.format_emotional_response("Wow", "surprise", 0.5, include_empathy=True)
        assert "Wow" in resp

    def test_format_returns_string(self):
        gen = EmotionalResponseGenerator()
        for emotion in ["happy", "sad", "angry", "fear", "surprise", "neutral"]:
            resp = gen.format_emotional_response("Test", emotion, 0.0)
            assert isinstance(resp, str)


# ── SentimentAnalyzer ─────────────────────────────────────────────────────────


class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("I love this great wonderful day")
        assert score > 0.0

    def test_negative_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("This is terrible and awful")
        assert score < 0.0

    def test_neutral_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("The weather is today")
        assert score == 0.0

    def test_detect_happy_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am so happy and excited!")
        assert emotion == "happy"

    def test_detect_sad_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I feel sad and depressed")
        assert emotion == "sad"

    def test_detect_angry_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am angry and frustrated")
        assert emotion == "angry"

    def test_detect_neutral_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("It is okay and normal")
        assert emotion == "neutral"

    def test_analyze_returns_dict(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am happy today")
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result

    def test_analyze_is_positive_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is wonderful and great")
        assert result["is_positive"] is True

    def test_analyze_is_negative_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is terrible and awful")
        assert result["is_negative"] is True

    def test_analyze_is_neutral_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("normal stuff")
        assert result["is_neutral"] is True

    def test_intensity_is_abs_sentiment(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("happy great wonderful")
        assert result["intensity"] == abs(result["sentiment"])

    def test_empty_text(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("")
        assert score == 0.0

    def test_mixed_sentiment(self):
        sa = SentimentAnalyzer()
        score = sa.analyze_sentiment("good bad")
        assert score == 0.0

    def test_detect_fear_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am afraid and scared")
        assert emotion == "fear"

    def test_detect_surprise_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am surprised and shocked")
        assert emotion == "surprise"


# ── RelationshipMemory ────────────────────────────────────────────────────────


class TestRelationshipMemory:
    def test_get_user_profile_creates_new(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("user1")
        assert profile["user_id"] == "user1"
        assert profile["total_interactions"] == 0

    def test_get_user_profile_returns_existing(self):
        rm = RelationshipMemory()
        p1 = rm.get_user_profile("user1")
        p2 = rm.get_user_profile("user1")
        assert p1 is p2

    def test_update_from_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            user_id="user1", user_input="hello", response="hi",
            sentiment=0.5, emotion="happy",
        )
        profile = rm.get_user_profile("user1")
        assert profile["total_interactions"] == 1

    def test_update_satisfaction_good(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            user_id="user1", user_input="hello", response="hi",
            sentiment=0.5, emotion="happy", feedback="good",
        )
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] == 0.6

    def test_update_satisfaction_bad(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            user_id="user1", user_input="hello", response="hi",
            sentiment=-0.5, emotion="sad", feedback="bad",
        )
        profile = rm.get_user_profile("user1")
        assert profile["satisfaction_score"] == 0.4

    def test_mood_history_capped_at_50(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction(
                user_id="user1", user_input="hello", response="hi",
                sentiment=0.5, emotion="happy",
            )
        profile = rm.get_user_profile("user1")
        assert len(profile["mood_history"]) <= 50

    def test_interaction_history_capped_at_100(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction(
                user_id="user1", user_input="hello", response="hi",
                sentiment=0.5, emotion="happy",
            )
        assert len(rm.interaction_history["user1"]) <= 100

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            user_id="user1", user_input="hello world", response="hi",
            sentiment=0.8, emotion="happy",
        )
        summary = rm.get_user_summary("user1")
        assert summary["user_id"] == "user1"
        assert summary["total_interactions"] == 1
        assert "dominant_emotion" in summary

    def test_get_relationship_context(self):
        rm = RelationshipMemory()
        for i in range(6):
            rm.update_from_interaction(
                user_id="user1", user_input="hello", response="hi",
                sentiment=0.5, emotion="happy",
            )
        ctx = rm.get_relationship_context("user1", "sad")
        assert isinstance(ctx, str)

    def test_low_satisfaction_context(self):
        rm = RelationshipMemory()
        for _ in range(6):
            rm.update_from_interaction(
                user_id="user1", user_input="hello", response="hi",
                sentiment=-0.8, emotion="sad", feedback="bad",
            )
        ctx = rm.get_relationship_context("user1", "neutral")
        assert "dissatisfied" in ctx.lower()

    def test_high_satisfaction_context(self):
        rm = RelationshipMemory()
        rm.update_from_interaction(
            user_id="user1", user_input="hello", response="hi",
            sentiment=0.8, emotion="happy", feedback="good",
        )
        for _ in range(5):
            rm.update_from_interaction(
                user_id="user1", user_input="hello", response="hi",
                sentiment=0.8, emotion="happy", feedback="good",
            )
        ctx = rm.get_relationship_context("user1", "neutral")
        assert "happy" in ctx.lower()


# ── SessionMemory ─────────────────────────────────────────────────────────────


class TestSessionMemory:
    def test_init(self):
        sm = SessionMemory()
        assert sm.conversation == []
        assert sm.session_id.startswith("session_")

    def test_add_message(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert len(sm.conversation) == 1

    def test_max_turns_eviction(self):
        sm = SessionMemory(max_turns=5)
        for i in range(10):
            sm.add("user", f"msg {i}")
        assert len(sm.conversation) == 5
        assert sm.conversation[0]["content"] == "msg 5"

    def test_get_context(self):
        sm = SessionMemory()
        for i in range(10):
            sm.add("user", f"msg {i}")
        ctx = sm.get_context(3)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg 7"

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        full = sm.get_full_session()
        assert len(full) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.clear()
        assert len(sm.conversation) == 0

    def test_get_summary(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        summary = sm.get_summary()
        assert summary["turns"] == 2

    def test_session_id_changes_on_clear(self):
        sm = SessionMemory()
        old_id = sm.session_id
        sm.clear()
        assert sm.session_id != old_id


# ── EpisodicMemoryStore ───────────────────────────────────────────────────────


class TestEpisodicMemoryStore:
    def test_save_episode(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}]
        episode_id = store.save_episode("session_1", conv)
        assert episode_id.startswith("conv_")

    def test_get_episode(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}]
        eid = store.save_episode("session_1", conv)
        episode = store.get_episode(eid)
        assert episode is not None
        assert len(episode) == 1

    def test_search_episodes(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important reminder"}]
        store.save_episode("session_1", conv)
        results = store.search_episodes("important")
        assert len(results) >= 1

    def test_eviction_on_max_episodes(self):
        store = EpisodicMemoryStore(max_episodes=3)
        for i in range(5):
            store.save_episode(f"session_{i}", [{"role": "user", "content": f"msg {i}"}])
        assert len(store.episodes) <= 3

    def test_importance_calculation(self):
        store = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "important critical key learn"}]
        importance = store._calculate_importance(conv)
        assert importance > 0.5

    def test_empty_conversation_importance(self):
        store = EpisodicMemoryStore()
        importance = store._calculate_importance([])
        assert importance == 0.0

    def test_get_recent_episodes(self):
        store = EpisodicMemoryStore()
        for i in range(3):
            store.save_episode(f"session_{i}", [{"role": "user", "content": "hello"}])
        recent = store.get_recent_episodes(2)
        assert len(recent) == 2

    def test_get_nonexistent_episode(self):
        store = EpisodicMemoryStore()
        assert store.get_episode("nonexistent") is None


# ── CognitiveArchitecture ─────────────────────────────────────────────────────


class TestCognitiveArchitecture:
    def test_init(self):
        ca = CognitiveArchitecture()
        assert ca.working_capacity == 7
        assert ca.sensory_buffer == []

    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        result = ca.process_sensory("input data")
        assert result is True
        assert len(ca.sensory_buffer) == 1

    def test_to_working(self):
        ca = CognitiveArchitecture()
        ca.to_working("item1")
        assert len(ca.working_memory) == 1

    def test_working_memory_eviction(self):
        ca = CognitiveArchitecture(working_capacity=3)
        for i in range(5):
            ca.to_working(f"item_{i}")
        assert len(ca.working_memory) == 3

    def test_add_to_session(self):
        ca = CognitiveArchitecture()
        msg = ca.add_to_session("user", "hello")
        assert msg["role"] == "user"

    def test_get_session_context(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        ctx = ca.get_session_context(1)
        assert len(ctx) == 1

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "value1")
        assert ca.retrieve_semantic("fact1") == "value1"

    def test_semantic_strengthening(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("fact1", "value1")
        ca.to_semantic("fact1", "value1")
        assert ca.semantic_memory["fact1"]["strength"] > 1.0

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        eid = ca.save_session_as_episode()
        assert eid.startswith("conv_")

    def test_recall_episodes(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "remember this")
        ca.save_session_as_episode()
        results = ca.recall_episodes("remember")
        assert len(results) >= 1


# ── NeuralPlasticityEngine ────────────────────────────────────────────────────


class TestNeuralPlasticityEngine:
    def test_init(self):
        npe = NeuralPlasticityEngine()
        assert npe.learning_rate == 0.01

    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("neuron1", 0.8)
        assert len(npe.activation_history["neuron1"]) == 1

    def test_hebbian_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("pre", 1.0)
        npe.activate("post", 1.0)
        weight = npe.hebbian_learn("pre", "post")
        assert weight > 0.0

    def test_connection_strength(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        npe.hebbian_learn("a", "b", reward=2.0)
        strength = npe.get_connection_strength("a", "b")
        assert strength > 0.0

    def test_prune_weak_connections(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 0.001)
        npe.activate("b", 0.001)
        npe.hebbian_learn("a", "b", reward=0.001)
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned >= 1

    def test_activation_history_capped(self):
        npe = NeuralPlasticityEngine()
        for i in range(150):
            npe.activate("n", float(i))
        assert len(npe.activation_history["n"]) <= 100

    def test_hebbian_no_activation_defaults(self):
        npe = NeuralPlasticityEngine()
        weight = npe.hebbian_learn("x", "y")
        assert weight > 0.0

    def test_prune_preserves_strong(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 10.0)
        npe.activate("b", 10.0)
        npe.hebbian_learn("a", "b")
        pruned = npe.prune_weak_connections(threshold=0.01)
        assert pruned == 0


# ── MetaLearningEngine ────────────────────────────────────────────────────────


class TestMetaLearningEngine:
    def test_init(self):
        mle = MetaLearningEngine()
        assert mle.best_strategy == "spaced"

    def test_record_outcome_success(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        assert mle.strategies["rote"]["success"] == 1

    def test_record_outcome_failure(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", False)
        assert mle.strategies["rote"]["success"] == 0
        assert mle.strategies["rote"]["attempts"] == 1

    def test_update_weights(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
        mle.update_weights()
        # weight = 0.7 * 1.0 + 0.3 * 1.0 = 1.0 (capped at 1.0)
        assert mle.strategies["rote"]["weight"] == 1.0

    def test_get_strategy(self):
        mle = MetaLearningEngine()
        strategy = mle.get_strategy()
        assert strategy in mle.strategies

    def test_unknown_strategy_ignored(self):
        mle = MetaLearningEngine()
        mle.record_outcome("unknown", True)
        # Unknown strategy should not affect anything

    def test_best_strategy_updates_on_failure(self):
        mle = MetaLearningEngine()
        # Degrade all strategies except elaborative
        for strat in ["rote", "spaced", "interleaved"]:
            for _ in range(20):
                mle.record_outcome(strat, False)
        for _ in range(20):
            mle.record_outcome("elaborative", True)
        mle.update_weights()
        assert mle.best_strategy == "elaborative"


# ── DreamProcessingEngine ─────────────────────────────────────────────────────


class TestDreamProcessingEngine:
    def test_init(self):
        dpe = DreamProcessingEngine()
        assert dpe.dream_cycles == 0
        assert dpe.consolidated == 0

    def test_dream_increments_cycles(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        dpe.dream([], plasticity)
        assert dpe.dream_cycles == 1

    def test_dream_returns_insights(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        # Create mock experiences
        class MockExp:
            def __init__(self, eid, imp):
                self.id = eid
                self.importance = imp
        memories = [MockExp(f"e{i}", float(i)) for i in range(5)]
        insights = dpe.dream(memories, plasticity)
        assert isinstance(insights, list)

    def test_dream_consolidates_memories(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()

        class MockExp:
            def __init__(self, eid, imp):
                self.id = eid
                self.importance = imp
        memories = [MockExp(f"e{i}", float(i)) for i in range(10)]
        dpe.dream(memories, plasticity)
        assert dpe.consolidated == 10

    def test_dream_generates_pattern_insight(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()

        class MockExp:
            def __init__(self, eid, imp):
                self.id = eid
                self.importance = imp
        memories = [MockExp(f"e{i}", float(i)) for i in range(5)]
        insights = dpe.dream(memories, plasticity)
        assert len(insights) > 0

    def test_dream_empty_memories(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()
        insights = dpe.dream([], plasticity)
        assert insights == []
        assert dpe.consolidated == 0

    def test_dream_hebbian_connections(self):
        dpe = DreamProcessingEngine()
        plasticity = NeuralPlasticityEngine()

        class MockExp:
            def __init__(self, eid, imp):
                self.id = eid
                self.importance = imp
        memories = [MockExp("a", 1.0), MockExp("b", 2.0)]
        dpe.dream(memories, plasticity)
        assert plasticity.get_connection_strength("a", "b") > 0.0
