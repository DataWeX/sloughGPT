"""Tests for QualiaEngine."""

import pytest
from domains.consciousness.qualia import QualiaEngine, QualiaState


class TestQualiaState:
    def test_defaults(self):
        s = QualiaState()
        assert s.valence == 0.0
        assert s.arousal == 0.0
        assert s.novelty == 0.0

    def test_decay(self):
        s = QualiaState(valence=0.8, arousal=0.6, novelty=0.9)
        s.decay(rate=0.5)
        assert s.valence == pytest.approx(0.4, abs=0.01)
        assert s.arousal == pytest.approx(0.3, abs=0.01)
        assert s.novelty == pytest.approx(0.45, abs=0.01)

    def test_to_dict(self):
        s = QualiaState(valence=0.5, arousal=0.3)
        d = s.to_dict()
        assert d["valence"] == 0.5
        assert d["arousal"] == 0.3
        assert "novelty" in d

    def test_magnitude(self):
        s = QualiaState(valence=1.0, arousal=1.0, dominance=1.0,
                        novelty=1.0, coherence=1.0, beauty=1.0)
        assert s.magnitude() == pytest.approx(1.0, abs=0.01)

    def test_magnitude_zero(self):
        s = QualiaState()
        assert s.magnitude() == 0.0


class TestQualiaEngine:
    def test_init(self):
        engine = QualiaEngine()
        assert engine.current.valence == 0.0
        assert len(engine.history) == 0

    def test_experience_positive(self):
        engine = QualiaEngine()
        state = engine.experience("I love this great wonderful thing!")
        assert state.valence > 0

    def test_experience_negative(self):
        engine = QualiaEngine()
        state = engine.experience("This is terrible and awful and bad.")
        assert state.valence < 0

    def test_experience_neutral(self):
        engine = QualiaEngine()
        state = engine.experience("the weather is cloudy today")
        assert abs(state.valence) < 0.5

    def test_experience_arousal_from_exclamation(self):
        engine = QualiaEngine()
        state = engine.experience("Wow!!! This is amazing!!!")
        assert state.arousal > 0.3

    def test_experience_question_increases_novelty(self):
        engine = QualiaEngine()
        state = engine.experience("What is the meaning of life?")
        assert state.novelty > 0.2

    def test_experience_stores_history(self):
        engine = QualiaEngine()
        engine.experience("hello")
        engine.experience("world")
        assert len(engine.history) == 2

    def test_experience_from_feedback_positive(self):
        engine = QualiaEngine()
        state = engine.experience_from_feedback(5)
        assert state.valence > 0
        assert state.coherence > 0.5

    def test_experience_from_feedback_negative(self):
        engine = QualiaEngine()
        state = engine.experience_from_feedback(1)
        assert state.valence < 0

    def test_recall_returns_none_empty(self):
        engine = QualiaEngine()
        assert engine.recall("hello") is None

    def test_recall_finds_similar(self):
        engine = QualiaEngine()
        engine.experience("I love this beautiful thing")
        result = engine.recall("beautiful love")
        assert result is not None

    def test_get_narrative_default(self):
        engine = QualiaEngine()
        engine.current = QualiaState(valence=0.8)
        narrative = engine.get_narrative()
        assert "positive" in narrative.lower()

    def test_get_narrative_calm(self):
        engine = QualiaEngine()
        engine.current = QualiaState(arousal=0.1)
        narrative = engine.get_narrative()
        assert "calm" in narrative.lower()

    def test_get_narrative_novel(self):
        engine = QualiaEngine()
        engine.current = QualiaState(novelty=0.8)
        narrative = engine.get_narrative()
        assert "novel" in narrative.lower() or "new" in narrative.lower()

    def test_history_capped(self):
        engine = QualiaEngine()
        for i in range(250):
            engine.experience(f"test {i}")
        assert len(engine.history) == 200
