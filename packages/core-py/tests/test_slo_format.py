"""Tests for domains.inference.slo_format — GenerationParams, ContextParams, PersonalityCore, BehavioralTraits, CognitiveSignature, EmotionalRange, SloProfile."""

from domains.inference.slo_format import (
    GenerationParams, ContextParams, PersonalityCore, BehavioralTraits,
    CognitiveSignature, EmotionalRange, SloProfile,
)


class TestGenerationParams:
    def test_defaults(self):
        gp = GenerationParams()
        assert gp.temperature == 0.7
        assert gp.top_p == 0.9
        assert gp.top_k == 40
        assert gp.max_tokens == 2048
        assert gp.repeat_penalty == 1.1

    def test_to_dict(self):
        d = GenerationParams().to_dict()
        assert isinstance(d, dict)
        assert d["temperature"] == 0.7
        assert "stop" in d

    def test_custom(self):
        gp = GenerationParams(temperature=1.2, max_tokens=512)
        assert gp.temperature == 1.2
        assert gp.max_tokens == 512


class TestContextParams:
    def test_defaults(self):
        cp = ContextParams()
        assert cp.context_window == 4096
        assert cp.num_ctx == 4096

    def test_to_dict(self):
        d = ContextParams().to_dict()
        assert d["context_window"] == 4096


class TestPersonalityCore:
    def test_defaults(self):
        pc = PersonalityCore()
        assert pc.warmth == 0.5
        assert pc.creativity == 0.5
        assert pc.humor == 0.5

    def test_to_dict(self):
        d = PersonalityCore().to_dict()
        assert "warmth" in d
        assert "curiosity" in d


class TestBehavioralTraits:
    def test_defaults(self):
        bt = BehavioralTraits()
        assert bt.speaking_style == "conversational"
        assert bt.reasoning_approach == "balanced"

    def test_to_dict(self):
        d = BehavioralTraits().to_dict()
        assert "speaking_style" in d
        assert "emotional_expressiveness" in d


class TestCognitiveSignature:
    def test_defaults(self):
        cs = CognitiveSignature()
        assert cs.pattern_recognition == 0.5
        assert cs.abstract_reasoning == 0.5

    def test_to_dict(self):
        d = CognitiveSignature().to_dict()
        assert "pattern_recognition" in d


class TestEmotionalRange:
    def test_defaults(self):
        er = EmotionalRange()
        assert er.empathy_depth == 0.5
        assert er.mood_responsiveness == 0.5

    def test_to_dict(self):
        d = EmotionalRange().to_dict()
        assert "empathy_depth" in d


class TestSloProfile:
    def test_defaults(self):
        sp = SloProfile(name="test")
        assert sp.name == "test"
        assert sp.version == "1.0.0"
        assert sp.born_at != ""
        assert sp.personality.warmth == 0.5

    def test_to_dict(self):
        sp = SloProfile(name="test")
        d = sp.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert "personality" in d
        assert "generation" in d

    def test_custom(self):
        sp = SloProfile(name="my_model", version="2.0", tags=["chat", "instruct"])
        assert sp.tags == ["chat", "instruct"]
