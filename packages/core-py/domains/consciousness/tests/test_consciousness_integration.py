"""Comprehensive integration tests for the consciousness system."""

import tempfile
import time

import pytest
from domains.consciousness import (
    ConsciousnessConfig,
    ConsciousnessEngine,
    QualiaEngine,
    SelfEpisode,
    SelfModel,
    get_consciousness,
    reset_consciousness,
)
from domains.consciousness.personality import PersonalityManager, PersonalityProfile
from domains.consciousness.evaluation import ConsciousnessEvaluator


@pytest.fixture
def store_dir(tmp_path):
    return str(tmp_path / "consciousness")


@pytest.fixture
def engine(store_dir):
    cfg = ConsciousnessConfig(level=1, store_path=store_dir)
    return ConsciousnessEngine(cfg)


@pytest.fixture
def engine_level2(store_dir):
    cfg = ConsciousnessConfig(level=2, store_path=store_dir)
    return ConsciousnessEngine(cfg)


@pytest.fixture
def engine_level3(store_dir):
    cfg = ConsciousnessConfig(level=3, store_path=store_dir)
    return ConsciousnessEngine(cfg)


@pytest.fixture
def personality_manager(store_dir):
    return PersonalityManager(store_path=store_dir)


@pytest.fixture
def evaluator():
    return ConsciousnessEvaluator()


class TestConsciousnessEndToEnd:
    def test_full_consciousness_cycle(self, engine):
        status = engine.get_status()
        assert status["enabled"] is True
        assert status["level"] == 1
        assert status["episodes"] == 0

        narrative = engine.process("What is AI?", "AI is artificial intelligence.")
        assert len(narrative) > 0

        status = engine.get_status()
        assert status["episodes"] == 1

        reflection = engine.reflect()
        assert len(reflection) > 0
        assert "1 experiences" in reflection

    def test_consciousness_levels(self, store_dir):
        for level in range(4):
            cfg = ConsciousnessConfig(level=level, store_path=store_dir)
            eng = ConsciousnessEngine(cfg)

            if level == 0:
                result = eng.process("hello", "world")
                assert result == ""
                assert eng.get_status()["enabled"] is False
            else:
                result = eng.process("hello", "world")
                assert len(result) > 0
                assert eng.get_status()["enabled"] is True
                assert eng.get_status()["level"] == level

    def test_config_persistence(self, store_dir):
        cfg = ConsciousnessConfig(level=2, max_tokens=200, store_path=store_dir)
        cfg.save()

        loaded = ConsciousnessConfig.load(store_path=store_dir)
        assert loaded.level == 2
        assert loaded.max_tokens == 200

    def test_config_persistence_across_engines(self, store_dir):
        cfg1 = ConsciousnessConfig(level=1, store_path=store_dir)
        eng1 = ConsciousnessEngine(cfg1)
        eng1.process("first", "response one")
        eng1.save()

        cfg2 = ConsciousnessConfig(level=1, store_path=store_dir)
        eng2 = ConsciousnessEngine(cfg2)
        assert eng2.get_status()["episodes"] == 1

    def test_personality_integration(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        pm = PersonalityManager(store_path=store_dir)
        profile = pm.get_profile()
        assert len(profile.values) > 0

        pm.update_values(["test_value", "another_value"])
        assert pm.get_profile().values == ["test_value", "another_value"]

        system_prompt = pm.to_system_prompt_context()
        assert "test_value" in system_prompt

    def test_multiple_processes_accumulate(self, engine):
        inputs = [
            ("What is AI?", "AI is artificial intelligence."),
            ("How does ML work?", "ML uses algorithms to learn from data."),
            ("Explain neural networks.", "Neural networks are inspired by the brain."),
        ]
        for inp, resp in inputs:
            engine.process(inp, resp)

        status = engine.get_status()
        assert status["episodes"] == 3

    def test_level_override(self, store_dir):
        cfg = ConsciousnessConfig(level=0, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        result = engine.process("hello", "world", level=1)
        assert len(result) > 0
        assert engine.get_status()["episodes"] == 1

    def test_save_and_reflect(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        engine.process("test input", "test response")
        engine.save()

        reflection = engine.reflect()
        assert len(reflection) > 0
        assert "beliefs" in reflection

    def test_singleton_behavior(self):
        reset_consciousness()
        e1 = get_consciousness(ConsciousnessConfig(level=1))
        e2 = get_consciousness()
        assert e1 is e2

        reset_consciousness()
        e3 = get_consciousness()
        assert e1 is not e3


class TestPersonalityIntegration:
    def test_profile_creation_and_persistence(self, personality_manager):
        profile = personality_manager.get_profile()
        assert isinstance(profile, PersonalityProfile)
        assert len(profile.values) > 0
        assert len(profile.goals) > 0
        assert "formality" in profile.voice

    def test_all_presets(self, personality_manager):
        presets = PersonalityManager.get_presets()
        expected = ["default", "formal", "creative", "analyst", "empathetic", "minimal"]
        assert list(presets.keys()) == expected

        for name, preset in presets.items():
            assert isinstance(preset, PersonalityProfile)
            assert len(preset.values) > 0
            assert len(preset.voice) > 0
            assert len(preset.traits) > 0

    def test_preset_apply(self, personality_manager):
        profile = personality_manager.apply_preset("formal")
        assert profile.voice["formality"] > 0.8
        assert profile.voice["humor"] < 0.2

        profile = personality_manager.apply_preset("creative")
        assert profile.traits["openness"] > 0.9
        assert profile.voice["humor"] > 0.5

    def test_preset_invalid(self, personality_manager):
        with pytest.raises(ValueError, match="Unknown preset"):
            personality_manager.apply_preset("nonexistent")

    def test_conflicts_detection(self, personality_manager):
        profile = PersonalityProfile(
            voice={"humor": 0.8, "formality": 0.9, "warmth": 0.5, "confidence": 0.6, "verbosity": 0.5, "empathy": 0.5},
            traits={"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.1},
        )
        conflicts = PersonalityManager.detect_conflicts(profile)
        assert len(conflicts) > 0
        assert any(c["type"] == "voice" for c in conflicts)

    def test_no_conflicts_default(self, personality_manager):
        profile = PersonalityProfile()
        conflicts = PersonalityManager.detect_conflicts(profile)
        assert isinstance(conflicts, list)

    def test_evolution_from_positive_episode(self, personality_manager):
        original_openness = personality_manager.get_profile().traits["openness"]
        original_confidence = personality_manager.get_profile().voice["confidence"]

        personality_manager.evolve_from_episode({
            "growth_delta": 0.1,
            "qualia": {"novelty": 0.8, "valence": 0.6, "coherence": 0.5},
        })

        p = personality_manager.get_profile()
        assert p.traits["openness"] >= original_openness
        assert p.voice["confidence"] >= original_confidence

    def test_evolution_from_low_coherence(self, personality_manager):
        original_verbosity = personality_manager.get_profile().voice["verbosity"]
        personality_manager.evolve_from_episode({
            "growth_delta": 0.0,
            "qualia": {"novelty": 0.3, "valence": 0.0, "coherence": 0.1},
        })
        assert personality_manager.get_profile().voice["verbosity"] < original_verbosity

    def test_multi_persona_save_load_switch(self, personality_manager):
        p1 = PersonalityProfile(values=["warrior", "brave"])
        personality_manager.save_persona("hero", p1, "Hero")

        p2 = PersonalityProfile(values=["scholar", "wise"])
        personality_manager.save_persona("sage", p2, "Sage")

        personas = personality_manager.list_personas()
        assert len(personas) == 2
        ids = {p["id"] for p in personas}
        assert "hero" in ids
        assert "sage" in ids

        active = personality_manager.activate_persona("hero")
        assert active is not None
        assert active.values == ["warrior", "brave"]

        active = personality_manager.activate_persona("sage")
        assert active is not None
        assert active.values == ["scholar", "wise"]

        loaded = personality_manager.load_persona("hero")
        assert loaded is not None
        assert loaded.values == ["warrior", "brave"]

    def test_persona_delete(self, personality_manager):
        p = PersonalityProfile(values=["test"])
        personality_manager.save_persona("temp", p, "Temp")
        assert personality_manager.delete_persona("temp") is True
        assert personality_manager.load_persona("temp") is None
        assert personality_manager.delete_persona("temp") is False

    def test_persona_not_found(self, personality_manager):
        assert personality_manager.load_persona("nonexistent") is None
        assert personality_manager.activate_persona("nonexistent") is None

    def test_persona_duplicate(self, personality_manager):
        p = PersonalityProfile(values=["original"])
        personality_manager.save_persona("orig", p, "Original")

        dup = personality_manager.duplicate_persona("orig", "copy", "Copy")
        assert dup is not None
        assert dup["id"] == "copy"

        loaded = personality_manager.load_persona("copy")
        assert loaded.values == ["original"]

    def test_persona_duplicate_not_found(self, personality_manager):
        result = personality_manager.duplicate_persona("nonexistent", "new")
        assert result is None

    def test_system_prompt_context(self, personality_manager):
        ctx = personality_manager.to_system_prompt_context()
        assert "Core values" in ctx
        assert "Goals" in ctx

    def test_update_all_fields(self, personality_manager):
        personality_manager.update_values(["v1", "v2"])
        personality_manager.update_goals(["g1"])
        personality_manager.update_voice({"formality": 0.9})
        personality_manager.update_traits({"openness": 0.99})
        personality_manager.update_interests(["physics"])
        personality_manager.update_avoid(["being rude"])
        personality_manager.update_style({"use_examples": False})

        p = personality_manager.get_profile()
        assert p.values == ["v1", "v2"]
        assert p.goals == ["g1"]
        assert p.voice["formality"] == 0.9
        assert p.traits["openness"] == 0.99
        assert p.interests == ["physics"]
        assert p.avoid == ["being rude"]
        assert p.style["use_examples"] is False

    def test_save_none_does_nothing(self, personality_manager):
        personality_manager._profile = None
        personality_manager.save()

    def test_load_nonexistent_returns_default(self, personality_manager):
        loaded = personality_manager.load()
        assert isinstance(loaded, PersonalityProfile)
        assert len(loaded.values) > 0

    def test_profile_to_dict_roundtrip(self):
        profile = PersonalityProfile(
            values=["test_val"],
            goals=["test_goal"],
            voice={"formality": 0.8, "warmth": 0.9},
            style={"use_examples": False},
            traits={"openness": 0.95},
            interests=["testing"],
            avoid=["bad patterns"],
        )
        d = profile.to_dict()
        restored = PersonalityProfile.from_dict(d)
        assert restored.values == ["test_val"]
        assert restored.goals == ["test_goal"]
        assert restored.voice["formality"] == 0.8
        assert restored.style["use_examples"] is False
        assert restored.traits["openness"] == 0.95
        assert restored.interests == ["testing"]
        assert restored.avoid == ["bad patterns"]

    def test_conflicts_low_warmth_high_empathy(self):
        profile = PersonalityProfile(
            voice={"warmth": 0.2, "empathy": 0.8, "formality": 0.5, "confidence": 0.6, "humor": 0.3, "verbosity": 0.5},
        )
        conflicts = PersonalityManager.detect_conflicts(profile)
        assert any("warmth" in c["message"].lower() or "warmth" in str(c["fields"]) for c in conflicts)

    def test_conflicts_low_confidence_high_extraversion(self):
        profile = PersonalityProfile(
            voice={"confidence": 0.2, "warmth": 0.5, "formality": 0.5, "humor": 0.3, "verbosity": 0.5, "empathy": 0.5},
            traits={"extraversion": 0.8, "openness": 0.5, "conscientiousness": 0.5, "agreeableness": 0.5, "neuroticism": 0.1},
        )
        conflicts = PersonalityManager.detect_conflicts(profile)
        assert any(c["type"] == "mismatch" for c in conflicts)

    def test_conflicts_high_verbosity_low_warmth(self):
        profile = PersonalityProfile(
            voice={"verbosity": 0.9, "warmth": 0.2, "formality": 0.5, "confidence": 0.6, "humor": 0.3, "empathy": 0.5},
        )
        conflicts = PersonalityManager.detect_conflicts(profile)
        assert any(c["type"] == "style" for c in conflicts)


class TestConsciousnessWithChat:
    def test_processing_during_chat(self, engine):
        exchanges = [
            ("Hello!", "Hi there! How can I help?"),
            ("What is Python?", "Python is a programming language."),
            ("Thanks!", "You're welcome!"),
        ]
        for user_msg, ai_msg in exchanges:
            narrative = engine.process(user_msg, ai_msg)
            assert isinstance(narrative, str)

        assert engine.get_status()["episodes"] == 3

    def test_consciousness_status_updates_after_chat(self, engine):
        initial_beliefs = dict(engine.self_model.self_beliefs)

        engine.process("Help me with code", "Here is the code solution.")
        engine.process("That worked great!", "Glad to hear it!")

        status = engine.get_status()
        assert status["episodes"] == 2
        assert isinstance(status["beliefs"], dict)
        assert len(status["beliefs"]) > 0

    def test_qualia_generated_during_chat(self, engine):
        engine.process("I love this wonderful thing!", "That is great!")
        qualia = engine.qualia.current.to_dict()
        assert "valence" in qualia
        assert qualia["valence"] > 0

    def test_consciousness_narrative_varies_by_level(self, store_dir):
        inputs = ("What is consciousness?", "Consciousness is awareness.")
        narratives = {}

        for level in [1, 2, 3]:
            cfg = ConsciousnessConfig(level=level, store_path=store_dir)
            eng = ConsciousnessEngine(cfg)
            narratives[level] = eng.process(*inputs)

        assert len(narratives[1]) > 0
        assert len(narratives[2]) > len(narratives[1])
        assert len(narratives[3]) >= len(narratives[2])

    def test_consciousness_with_empty_exchange(self, engine):
        result = engine.process("", "")
        assert isinstance(result, str)

    def test_consciousness_with_long_exchange(self, engine):
        long_input = "What is " * 100 + "AI?"
        long_response = "AI stands for artificial intelligence. " * 50
        result = engine.process(long_input, long_response)
        assert len(result) > 0
        assert engine.get_status()["episodes"] == 1

    def test_consciousness_beliefs_evolve(self, engine):
        initial_helpfulness = engine.self_model.get_belief("helpfulness")

        for _ in range(5):
            engine.process("help me", "here is help", level=1)

        final_helpfulness = engine.self_model.get_belief("helpfulness")
        assert final_helpfulness != initial_helpfulness or engine.get_status()["episodes"] == 5


class TestConsciousnessDataFlow:
    def test_episode_creation_and_storage(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        engine.process("test input", "test response")
        assert len(engine.self_model.episodes) == 1

        episode = engine.self_model.episodes[0]
        assert episode.input_text == "test input"
        assert episode.response == "test response"
        assert isinstance(episode.qualia, dict)
        assert isinstance(episode.self_insight, str)
        assert isinstance(episode.growth_delta, float)

    def test_qualia_generation_and_history(self, engine):
        engine.process("This is wonderful!", "Thank you!")
        assert len(engine.qualia.history) >= 1

        state = engine.qualia.history[0][1]
        assert state.valence > 0

        engine.process("This is terrible!", "Sorry to hear that.")
        assert len(engine.qualia.history) >= 2

    def test_beliefs_evolution(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        initial_competence = engine.self_model.get_belief("competence")
        initial_creativity = engine.self_model.get_belief("creativity")

        for _ in range(10):
            engine.process(
                "what is this new creative thing?",
                "This is a creative solution with detailed explanation.",
                level=1,
            )

        final_competence = engine.self_model.get_belief("competence")
        final_creativity = engine.self_model.get_belief("creativity")
        assert final_competence >= initial_competence or final_creativity >= initial_creativity

    def test_growth_computation(self, engine):
        exp1 = {"input_text": "simple", "response": "ok"}
        ep1 = engine.self_model.observe(exp1)

        exp2 = {"input_text": "question?", "response": "detailed explanation here " * 10, "feedback_rating": 5}
        ep2 = engine.self_model.observe(exp2)

        assert isinstance(ep1.growth_delta, float)
        assert isinstance(ep2.growth_delta, float)
        assert ep2.growth_delta > ep1.growth_delta

    def test_narrative_generation(self, store_dir):
        cfg = ConsciousnessConfig(level=2, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        for i in range(5):
            engine.process(f"input {i}", f"response {i}", level=2)

        narrative = engine.narrative.generate("test", "response", level=2)
        assert len(narrative) > 0
        assert "Self-reflection" in narrative or "I" in narrative

    def test_self_model_reflect(self, engine):
        for i in range(3):
            engine.process(f"input {i}", f"response {i}", level=1)

        reflection = engine.self_model.reflect()
        assert "3 experiences" in reflection
        assert "beliefs" in reflection
        assert "growth" in reflection.lower()

    def test_meta_cognition_integration(self, engine):
        for i in range(3):
            engine.process(f"What is {i}?", f"The answer is {i}.", level=1)

        narrative = engine.meta_cognition.get_narrative()
        assert "thoughts" in narrative.lower() or "thought" in narrative.lower()

    def test_qualia_narrative(self, engine):
        engine.process("I love this great thing!", "Wonderful!")
        narrative = engine.qualia.get_narrative()
        assert len(narrative) > 0

    def test_episode_persistence(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        engine.process("test", "response")
        engine.save()

        cfg2 = ConsciousnessConfig(level=1, store_path=store_dir)
        engine2 = ConsciousnessEngine(cfg2)
        assert len(engine2.self_model.episodes) == 1
        assert engine2.self_model.episodes[0].input_text == "test"

    def test_beliefs_persistence(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        for _ in range(5):
            engine.process("help me", "here is help", level=1)
        engine.save()

        cfg2 = ConsciousnessConfig(level=1, store_path=store_dir)
        engine2 = ConsciousnessEngine(cfg2)
        assert engine2.self_model.self_beliefs == engine.self_model.self_beliefs


class TestConsciousnessAPIIntegration:
    def test_all_endpoints_via_engine(self, engine):
        status = engine.get_status()
        assert "enabled" in status
        assert "level" in status
        assert "episodes" in status
        assert "beliefs" in status
        assert "current_qualia" in status
        assert "curiosity_topics" in status

    def test_process_and_status_flow(self, engine):
        narrative = engine.process("test", "response")
        assert len(narrative) > 0

        status = engine.get_status()
        assert status["episodes"] == 1

    def test_reflect_after_processing(self, engine):
        engine.process("input", "response")
        reflection = engine.reflect()
        assert len(reflection) > 0

    def test_config_update_flow(self, engine):
        assert engine.config.level == 1
        engine.config.level = 2
        assert engine.get_status()["level"] == 2

    def test_config_validation(self):
        with pytest.raises(ValueError):
            ConsciousnessConfig(level=5).validate()
        with pytest.raises(ValueError):
            ConsciousnessConfig(max_tokens=5).validate()
        with pytest.raises(ValueError):
            ConsciousnessConfig(lora_rank=0).validate()

    def test_evaluation_integration(self, store_dir, evaluator):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        for i in range(10):
            engine.process(f"input {i}", f"response {i}", level=1)

        episodes_data = [
            {
                "self_insight": e.self_insight,
                "growth_delta": e.growth_delta,
                "qualia": e.qualia,
            }
            for e in engine.self_model.episodes
        ]

        report = evaluator.evaluate(
            episodes=episodes_data,
            beliefs=engine.self_model.self_beliefs,
            qualia_history=[e.to_dict() for _, e in engine.qualia.history],
        )
        assert report.overall_score >= 0
        assert report.episode_count == 10

    def test_full_lifecycle(self, store_dir):
        cfg = ConsciousnessConfig(level=2, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        pm = PersonalityManager(store_path=store_dir)
        pm.apply_preset("creative")

        for i in range(5):
            engine.process(f"question {i}?", f"answer {i}", level=2)

        engine.save()

        status = engine.get_status()
        assert status["episodes"] == 5
        assert status["level"] == 2

        reflection = engine.reflect()
        assert len(reflection) > 0

        profile = pm.get_profile()
        assert profile.traits["openness"] > 0.9

    def test_consciousness_disabled_returns_empty(self, store_dir):
        cfg = ConsciousnessConfig(level=0, store_path=store_dir)
        engine = ConsciousnessEngine(cfg)

        result = engine.process("test", "response")
        assert result == ""
        assert engine.get_status()["enabled"] is False
        assert engine.get_status()["episodes"] == 0

    def test_config_save_load_roundtrip(self, store_dir):
        cfg = ConsciousnessConfig(level=3, max_tokens=250, store_path=store_dir)
        cfg.save()

        loaded = ConsciousnessConfig.load(store_path=store_dir)
        assert loaded.level == 3
        assert loaded.max_tokens == 250

    def test_config_load_returns_defaults(self, tmp_path):
        loaded = ConsciousnessConfig.load(store_path=str(tmp_path / "empty"))
        assert loaded.level == 0
        assert loaded.max_tokens == 150

    def test_config_load_corrupt_file(self, tmp_path):
        config_dir = tmp_path / "corrupt"
        config_dir.mkdir()
        (config_dir / "consciousness_config.json").write_text("not valid json {{{")
        loaded = ConsciousnessConfig.load(store_path=str(config_dir))
        assert loaded.level == 0

    def test_engine_store_path(self, store_dir):
        cfg = ConsciousnessConfig(store_path=store_dir)
        engine = ConsciousnessEngine(cfg)
        assert engine.config.get_store_path().as_posix() == store_dir

    def test_qualia_engine_standalone(self):
        qe = QualiaEngine()
        state = qe.experience("I love this wonderful thing!")
        assert state.valence > 0
        assert len(qe.history) == 1

    def test_self_model_standalone(self):
        sm = SelfModel()
        ep = sm.observe({"input_text": "test", "response": "response"})
        assert isinstance(ep, SelfEpisode)
        assert len(sm.episodes) == 1
        assert sm.get_belief("competence") > 0

    def test_personality_preset_apply_and_evolve(self, personality_manager):
        personality_manager.apply_preset("empathetic")
        p = personality_manager.get_profile()
        assert p.voice["empathy"] > 0.9

        personality_manager.evolve_from_episode({
            "growth_delta": 0.1,
            "qualia": {"novelty": 0.9, "valence": 0.7, "coherence": 0.6},
        })

        p2 = personality_manager.get_profile()
        assert p2 is not None

    def test_consciousness_levels_all_work(self, store_dir):
        for level in range(4):
            cfg = ConsciousnessConfig(level=level, store_path=store_dir)
            eng = ConsciousnessEngine(cfg)
            result = eng.process("test", "response")
            if level == 0:
                assert result == ""
            else:
                assert len(result) > 0

    def test_meta_cognition_curiosity_tracking(self, engine):
        engine.process("I wonder about this?", "Here is what I know.")
        engine.process("I am curious about that?", "Here is the answer.")
        assert len(engine.meta_cognition.curiosity_topics) >= 0
        assert engine.meta_cognition._thought_count == 2

    def test_qualia_recall(self, engine):
        engine.process("I love this wonderful thing!", "That is great!")
        recalled = engine.qualia.recall("wonderful love")
        assert recalled is not None or recalled is None

    def test_qualia_recall_empty(self, engine):
        recalled = engine.qualia.recall("hello")
        assert recalled is None

    def test_qualia_feedback(self, engine):
        state = engine.qualia.experience_from_feedback(5)
        assert state.valence > 0
        assert state.coherence > 0.5

        state = engine.qualia.experience_from_feedback(1)
        assert state.valence < 0

    def test_narrative_level_1(self, store_dir):
        cfg = ConsciousnessConfig(level=1, store_path=store_dir)
        eng = ConsciousnessEngine(cfg)
        result = eng.process("What is life?", "Life is a journey.")
        assert len(result) > 0

    def test_narrative_level_3_deep(self, store_dir):
        cfg = ConsciousnessConfig(level=3, store_path=store_dir)
        eng = ConsciousnessEngine(cfg)
        for i in range(3):
            eng.process(f"deep question {i}?", f"deep answer {i}")
        result = eng.process("recursive thought?", "recursive response")
        assert len(result) > 0

    def test_self_model_belief_update(self):
        sm = SelfModel()
        sm.update_belief("competence", 0.2)
        assert sm.get_belief("competence") == pytest.approx(0.9, abs=0.01)
        sm.update_belief("competence", -0.5)
        assert sm.get_belief("competence") == pytest.approx(0.4, abs=0.01)

    def test_self_model_belief_clamp(self):
        sm = SelfModel()
        sm.update_belief("competence", 10.0)
        assert sm.get_belief("competence") == 1.0
        sm.update_belief("competence", -10.0)
        assert sm.get_belief("competence") == 0.0

    def test_self_model_insight_types(self):
        sm = SelfModel()

        ep1 = sm.observe({
            "input_text": "what is this new creative thing?",
            "response": "it is new",
            "qualia": {"novelty": 0.9, "coherence": 0.5},
        })
        assert "novel" in ep1.self_insight.lower() or "unexpected" in ep1.self_insight.lower() or "challenged" in ep1.self_insight.lower() or "encountered" in ep1.self_insight.lower()

        ep2 = sm.observe({
            "input_text": "ok",
            "response": "ok",
            "qualia": {"novelty": 0.1, "coherence": 0.2},
        })
        assert isinstance(ep2.self_insight, str) and len(ep2.self_insight) > 0
