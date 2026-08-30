"""Tests for domains.soul.cognitive — all classes except DreamProcessingEngine.

Covers: NeuralPlasticityEngine, MetaLearningEngine, SentimentAnalyzer,
EmotionalResponseGenerator, RelationshipMemory, SessionMemory,
EpisodicMemoryStore, CognitiveArchitecture.
"""

from domains.soul.cognitive import (
    NeuralPlasticityEngine,
    MetaLearningEngine,
    SentimentAnalyzer,
    EmotionalResponseGenerator,
    RelationshipMemory,
    SessionMemory,
    EpisodicMemoryStore,
    CognitiveArchitecture,
)


# ── NeuralPlasticityEngine ──────────────────────────────────────────────

class TestNeuralPlasticityEngine:
    def test_init(self):
        npe = NeuralPlasticityEngine()
        assert npe.learning_rate == 0.01
        assert len(npe.connections) == 0

    def test_init_custom_lr(self):
        npe = NeuralPlasticityEngine(learning_rate=0.05)
        assert npe.learning_rate == 0.05

    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1", 0.8)
        assert len(npe.activation_history["n1"]) == 1
        assert npe.activation_history["n1"][0] == 0.8

    def test_activate_default_strength(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1")
        assert npe.activation_history["n1"][0] == 1.0

    def test_activate_multiple(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1", 0.5)
        npe.activate("n1", 0.9)
        assert len(npe.activation_history["n1"]) == 2

    def test_activate_history_trimming(self):
        npe = NeuralPlasticityEngine()
        for i in range(120):
            npe.activate("n1", float(i % 10) / 10)
        assert len(npe.activation_history["n1"]) <= 100

    def test_hebbian_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("pre", 1.0)
        npe.activate("post", 1.0)
        weight = npe.hebbian_learn("pre", "post")
        assert weight > 0.0

    def test_hebbian_learn_no_prior_activation(self):
        npe = NeuralPlasticityEngine()
        weight = npe.hebbian_learn("x", "y")
        assert weight > 0.0

    def test_hebbian_learn_with_reward(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        w1 = npe.hebbian_learn("a", "b", reward=1.0)
        w2 = npe.hebbian_learn("a", "b", reward=2.0)
        assert w2 > w1

    def test_hebbian_learn_accumulates(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        w1 = npe.hebbian_learn("a", "b")
        w2 = npe.hebbian_learn("a", "b")
        assert w2 > w1

    def test_get_connection_strength(self):
        npe = NeuralPlasticityEngine()
        assert npe.get_connection_strength("a", "b") == 0.0

    def test_get_connection_strength_after_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        npe.hebbian_learn("a", "b")
        assert npe.get_connection_strength("a", "b") > 0.0

    def test_prune_weak_connections(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 0.001)
        npe.activate("b", 0.001)
        npe.hebbian_learn("a", "b")
        pruned = npe.prune_weak_connections(threshold=1.0)
        assert pruned >= 1
        assert npe.get_connection_strength("a", "b") == 0.0

    def test_prune_preserves_strong(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 1.0)
        npe.hebbian_learn("a", "b")
        npe.hebbian_learn("a", "b")
        npe.hebbian_learn("a", "b")
        npe.hebbian_learn("a", "b")
        pruned = npe.prune_weak_connections(threshold=0.0001)
        assert pruned == 0
        assert npe.get_connection_strength("a", "b") > 0.0

    def test_multiple_neuron_pairs(self):
        npe = NeuralPlasticityEngine()
        npe.activate("a", 1.0)
        npe.activate("b", 0.5)
        npe.activate("c", 0.8)
        npe.hebbian_learn("a", "b")
        npe.hebbian_learn("a", "c")
        assert npe.get_connection_strength("a", "b") > 0.0
        assert npe.get_connection_strength("a", "c") > 0.0


# ── MetaLearningEngine ──────────────────────────────────────────────────

class TestMetaLearningEngine:
    def test_init(self):
        mle = MetaLearningEngine()
        assert len(mle.strategies) == 4
        assert mle.best_strategy == "spaced"

    def test_init_strategies(self):
        mle = MetaLearningEngine()
        assert "rote" in mle.strategies
        assert "spaced" in mle.strategies
        assert "interleaved" in mle.strategies
        assert "elaborative" in mle.strategies

    def test_init_strategy_weights(self):
        mle = MetaLearningEngine()
        for s in mle.strategies.values():
            assert s["weight"] == 1.0
            assert s["success"] == 0
            assert s["attempts"] == 0

    def test_record_outcome(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", False)
        assert mle.strategies["rote"]["attempts"] == 2
        assert mle.strategies["rote"]["success"] == 1

    def test_record_outcome_unknown_strategy(self):
        mle = MetaLearningEngine()
        original_keys = set(mle.strategies.keys())
        mle.record_outcome("nonexistent", True)
        assert set(mle.strategies.keys()) == original_keys

    def test_update_weights(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
        mle.update_weights()
        assert mle.strategies["rote"]["weight"] == 1.0
        assert mle.strategies["rote"]["success"] == 10

    def test_update_weights_changes_best(self):
        mle = MetaLearningEngine()
        for _ in range(20):
            mle.record_outcome("rote", True)
        mle.record_outcome("spaced", False)
        mle.update_weights()
        assert mle.best_strategy == "rote"

    def test_get_strategy(self):
        mle = MetaLearningEngine()
        s = mle.get_strategy()
        assert s in mle.strategies

    def test_weight_update_formula(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", False)
        old_w = mle.strategies["rote"]["weight"]
        mle.update_weights()
        new_w = mle.strategies["rote"]["weight"]
        success_rate = 2 / 3
        expected = 0.7 * old_w + 0.3 * success_rate
        assert abs(new_w - expected) < 1e-9

    def test_strategies_with_zero_attempts(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        mle.update_weights()
        assert mle.strategies["interleaved"]["weight"] == 1.0
        assert mle.strategies["elaborative"]["weight"] == 1.0


# ── SentimentAnalyzer ───────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        sa = SentimentAnalyzer()
        s = sa.analyze_sentiment("good great wonderful love")
        assert s > 0.0

    def test_negative_sentiment(self):
        sa = SentimentAnalyzer()
        s = sa.analyze_sentiment("bad terrible awful hate")
        assert s < 0.0

    def test_neutral_sentiment(self):
        sa = SentimentAnalyzer()
        s = sa.analyze_sentiment("the cat sat on the mat")
        assert s == 0.0

    def test_mixed_sentiment(self):
        sa = SentimentAnalyzer()
        s = sa.analyze_sentiment("good bad")
        assert s == 0.0

    def test_detect_emotion_happy(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am happy and excited") == "happy"

    def test_detect_emotion_sad(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I feel sad and depressed") == "sad"

    def test_detect_emotion_angry(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am so angry and frustrated") == "angry"

    def test_detect_emotion_fear(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("I am scared and afraid") == "fear"

    def test_detect_emotion_neutral(self):
        sa = SentimentAnalyzer()
        assert sa.detect_emotion("the weather is normal") == "neutral"

    def test_analyze_returns_dict(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am happy today")
        assert "sentiment" in result
        assert "emotion" in result
        assert "intensity" in result
        assert "is_positive" in result
        assert "is_negative" in result
        assert "is_neutral" in result

    def test_analyze_intensity(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I am happy")
        assert result["intensity"] >= 0.0
        assert result["intensity"] <= 1.0

    def test_analyze_is_positive_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("good great wonderful")
        assert result["is_positive"] is True

    def test_analyze_is_negative_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("bad terrible awful")
        assert result["is_negative"] is True

    def test_analyze_is_neutral_flag(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("the table is wood")
        assert result["is_neutral"] is True

    def test_analyze_empty_text(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("")
        assert result["sentiment"] == 0.0
        assert result["emotion"] == "neutral"


# ── EmotionalResponseGenerator ──────────────────────────────────────────

class TestEmotionalResponseGenerator:
    def test_generate_happy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_generate_sad(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("sad", -0.8)
        assert isinstance(resp, str)

    def test_generate_unknown_emotion(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("unknown_emotion", 0.5)
        assert isinstance(resp, str)

    def test_adapt_response_positive(self):
        erg = EmotionalResponseGenerator()
        resp = erg.adapt_response("Hello", "happy", 0.8)
        assert "!" in resp

    def test_adapt_response_negative(self):
        erg = EmotionalResponseGenerator()
        resp = erg.adapt_response("Hello", "sad", -0.8)
        assert "Hello" in resp
        assert resp != "Hello"

    def test_adapt_response_neutral(self):
        erg = EmotionalResponseGenerator()
        resp = erg.adapt_response("Hello", "neutral", 0.0)
        assert resp == "Hello"

    def test_format_with_empathy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.format_emotional_response("Sure", "happy", 0.8, include_empathy=True)
        assert "Sure" in resp
        assert len(resp) > len("Sure")

    def test_format_without_empathy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.format_emotional_response("Sure", "happy", 0.8, include_empathy=False)
        assert "Sure" in resp

    def test_format_neutral_no_empathy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.format_emotional_response("Sure", "neutral", 0.0)
        assert resp == "Sure"


# ── RelationshipMemory ──────────────────────────────────────────────────

class TestRelationshipMemory:
    def test_get_user_profile(self):
        rm = RelationshipMemory()
        p = rm.get_user_profile("u1")
        assert p["user_id"] == "u1"
        assert p["total_interactions"] == 0

    def test_get_user_profile_creates(self):
        rm = RelationshipMemory()
        rm.get_user_profile("u1")
        assert "u1" in rm.user_profiles

    def test_update_from_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        p = rm.get_user_profile("u1")
        assert p["total_interactions"] == 1
        assert p["emotional_tendencies"]["neutral"] == 1

    def test_update_satisfaction_good(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral", feedback="good")
        p = rm.get_user_profile("u1")
        assert p["satisfaction_score"] == 0.6

    def test_update_satisfaction_bad(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral", feedback="bad")
        p = rm.get_user_profile("u1")
        assert p["satisfaction_score"] == 0.4

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        s = rm.get_user_summary("u1")
        assert s["user_id"] == "u1"
        assert s["total_interactions"] == 1

    def test_get_relationship_context(self):
        rm = RelationshipMemory()
        ctx = rm.get_relationship_context("u1", "happy")
        assert isinstance(ctx, str)

    def test_mood_history_trimming(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        p = rm.get_user_profile("u1")
        assert len(p["mood_history"]) <= 50

    def test_interaction_history_trimming(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction("u1", "hello", "hi", 0.5, "neutral")
        assert len(rm.interaction_history["u1"]) <= 100

    def test_dominant_emotion_in_summary(self):
        rm = RelationshipMemory()
        for _ in range(5):
            rm.update_from_interaction("u1", "happy", "hi", 0.8, "happy")
        rm.update_from_interaction("u1", "sad", "hi", -0.5, "sad")
        s = rm.get_user_summary("u1")
        assert s["dominant_emotion"] == "happy"


# ── SessionMemory ───────────────────────────────────────────────────────

class TestSessionMemory:
    def test_init(self):
        sm = SessionMemory()
        assert len(sm.conversation) == 0
        assert sm.session_id.startswith("session_")

    def test_add(self):
        sm = SessionMemory()
        msg = sm.add("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == "hello"
        assert len(sm.conversation) == 1

    def test_add_multiple(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        assert len(sm.conversation) == 2

    def test_get_context(self):
        sm = SessionMemory()
        for i in range(10):
            sm.add("user", f"msg{i}")
        ctx = sm.get_context(3)
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg7"

    def test_get_full_session(self):
        sm = SessionMemory()
        sm.add("user", "a")
        sm.add("assistant", "b")
        full = sm.get_full_session()
        assert len(full) == 2

    def test_clear(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        old_id = sm.session_id
        sm.clear()
        assert len(sm.conversation) == 0
        assert sm.session_id != old_id

    def test_max_turns(self):
        sm = SessionMemory(max_turns=3)
        for i in range(5):
            sm.add("user", f"msg{i}")
        assert len(sm.conversation) == 3
        assert sm.conversation[0]["content"] == "msg2"

    def test_get_summary(self):
        sm = SessionMemory()
        sm.add("user", "hello")
        sm.add("assistant", "hi")
        s = sm.get_summary()
        assert s["turns"] == 2
        assert "session_id" in s


# ── EpisodicMemoryStore ─────────────────────────────────────────────────

class TestEpisodicMemoryStore:
    def test_save_episode(self):
        em = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        eid = em.save_episode("session1", conv)
        assert eid.startswith("conv_")
        assert eid in em.episodes

    def test_get_episode(self):
        em = EpisodicMemoryStore()
        conv = [{"role": "user", "content": "hello"}]
        eid = em.save_episode("session1", conv)
        ep = em.get_episode(eid)
        assert ep is not None
        assert len(ep) == 1

    def test_get_episode_missing(self):
        em = EpisodicMemoryStore()
        assert em.get_episode("nonexistent") is None

    def test_search_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"role": "user", "content": "hello world"}])
        results = em.search_episodes("hello")
        assert len(results) == 1

    def test_search_episodes_no_match(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"role": "user", "content": "hello"}])
        results = em.search_episodes("xyz")
        assert len(results) == 0

    def test_get_recent_episodes(self):
        em = EpisodicMemoryStore()
        em.save_episode("s1", [{"role": "user", "content": "a"}])
        em.save_episode("s2", [{"role": "user", "content": "b"}])
        recent = em.get_recent_episodes(1)
        assert len(recent) == 1

    def test_importance_with_keywords(self):
        em = EpisodicMemoryStore()
        imp = em._calculate_importance([{"content": "important remember critical"}])
        assert imp > 0.5

    def test_importance_empty(self):
        em = EpisodicMemoryStore()
        assert em._calculate_importance([]) == 0.0

    def test_importance_short(self):
        em = EpisodicMemoryStore()
        imp = em._calculate_importance([{"content": "hi"}])
        assert imp == 0.5

    def test_eviction(self):
        em = EpisodicMemoryStore(max_episodes=2)
        em.save_episode("s1", [{"content": "a"}])
        em.save_episode("s2", [{"content": "b important remember"}])
        em.save_episode("s3", [{"content": "c"}])
        assert len(em.episodes) == 2


# ── CognitiveArchitecture ───────────────────────────────────────────────

class TestCognitiveArchitecture:
    def test_init(self):
        ca = CognitiveArchitecture()
        assert len(ca.sensory_buffer) == 0
        assert len(ca.working_memory) == 0
        assert ca.working_capacity == 7

    def test_process_sensory(self):
        ca = CognitiveArchitecture()
        assert ca.process_sensory("input") is True
        assert len(ca.sensory_buffer) == 1

    def test_process_sensory_trimming(self):
        ca = CognitiveArchitecture()
        for i in range(110):
            ca.process_sensory(f"input{i}")
        assert len(ca.sensory_buffer) <= 100

    def test_to_working(self):
        ca = CognitiveArchitecture()
        ca.to_working("item")
        assert "item" in ca.working_memory

    def test_to_working_evicts(self):
        ca = CognitiveArchitecture(working_capacity=2)
        ca.to_working("a")
        ca.to_working("b")
        ca.to_working("c")
        assert len(ca.working_memory) == 2
        assert "a" not in ca.working_memory

    def test_add_to_session(self):
        ca = CognitiveArchitecture()
        msg = ca.add_to_session("user", "hello")
        assert msg["role"] == "user"

    def test_get_session_context(self):
        ca = CognitiveArchitecture()
        for i in range(5):
            ca.add_to_session("user", f"msg{i}")
        ctx = ca.get_session_context(3)
        assert len(ctx) == 3

    def test_save_session_as_episode(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello")
        eid = ca.save_session_as_episode()
        assert eid.startswith("conv_")

    def test_recall_episodes(self):
        ca = CognitiveArchitecture()
        ca.add_to_session("user", "hello world")
        ca.save_session_as_episode()
        results = ca.recall_episodes("hello")
        assert len(results) >= 1

    def test_to_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("key1", "value1")
        assert ca.to_semantic("key1", "value1") is True

    def test_retrieve_semantic(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("key1", "value1")
        v = ca.retrieve_semantic("key1")
        assert v == "value1"

    def test_retrieve_semantic_missing(self):
        ca = CognitiveArchitecture()
        assert ca.retrieve_semantic("missing") is None

    def test_semantic_strength_increases(self):
        ca = CognitiveArchitecture()
        ca.to_semantic("k", "v")
        s1 = ca.semantic_memory["k"]["strength"]
        ca.to_semantic("k", "v")
        s2 = ca.semantic_memory["k"]["strength"]
        assert s2 > s1
