"""Meaningful tests for SentimentAnalyzer — sentiment scoring, emotion detection, analyze."""

from domains.soul.cognitive import SentimentAnalyzer


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
