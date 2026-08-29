"""Tests for domains.soul.cognitive — SentimentAnalyzer, EmotionalResponseGenerator, RelationshipMemory."""

from domains.soul.cognitive import (
    SentimentAnalyzer, EmotionalResponseGenerator, RelationshipMemory,
)


class TestSentimentAnalyzer:
    def test_analyze_positive(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("I love this, it is absolutely amazing!")
        assert result["sentiment"] > 0

    def test_analyze_negative(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("This is terrible and I hate it.")
        assert result["sentiment"] < 0

    def test_detect_emotion(self):
        sa = SentimentAnalyzer()
        emotion = sa.detect_emotion("I am so happy today!")
        assert emotion in ("happy", "neutral", "surprise")

    def test_analyze_returns_dict(self):
        sa = SentimentAnalyzer()
        result = sa.analyze("hello world")
        assert "sentiment" in result
        assert "emotion" in result


class TestEmotionalResponseGenerator:
    def test_happy_empathy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("happy", 0.8)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_sad_empathy(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("sad", -0.6)
        assert isinstance(resp, str)

    def test_unknown_emotion_fallback(self):
        erg = EmotionalResponseGenerator()
        resp = erg.generate_empathetic_response("nonexistent", 0.0)
        assert isinstance(resp, str)

    def test_adapt_positive(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Great", "happy", 0.9)
        assert result != "Great"

    def test_adapt_negative(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("Sorry", "sad", -0.9)
        assert result != "Sorry"

    def test_adapt_neutral(self):
        erg = EmotionalResponseGenerator()
        result = erg.adapt_response("OK", "neutral", 0.0)
        assert result == "OK"

    def test_format_emotional_response(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Sure", "happy", 0.8)
        assert isinstance(result, str)
        assert "Sure" in result

    def test_format_no_empathy(self):
        erg = EmotionalResponseGenerator()
        result = erg.format_emotional_response("Sure", "neutral", 0.0, include_empathy=False)
        assert result == "Sure"


class TestRelationshipMemory:
    def test_init(self):
        rm = RelationshipMemory()
        assert rm.user_profiles == {}

    def test_get_user_profile(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("user1")
        assert "user_id" in profile
        assert profile["user_id"] == "user1"

    def test_update_from_interaction(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "Hello!", "Hi there!", sentiment=0.5, emotion="happy")
        profile = rm.get_user_profile("user1")
        assert profile["total_interactions"] >= 1

    def test_get_user_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("user1", "a", "b", sentiment=0.0, emotion="neutral")
        summary = rm.get_user_summary("user1")
        assert isinstance(summary, dict)
