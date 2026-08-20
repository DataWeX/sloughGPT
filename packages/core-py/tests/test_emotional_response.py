"""Meaningful tests for EmotionalResponseGenerator — empathetic responses, adaptation, formatting."""

from domains.soul.cognitive import EmotionalResponseGenerator


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
