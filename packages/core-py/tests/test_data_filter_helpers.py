"""Tests for domains.learner.data_filter — pure helper functions."""

from domains.learner.data_filter import (
    _hashed, _score_quality, _score_relevance,
    _matches_blacklist, _matches_whitelist,
)


class TestHashed:
    def test_deterministic(self):
        assert _hashed(1.0) == _hashed(1.0)

    def test_different_inputs(self):
        assert _hashed(1.0) != _hashed(2.0)

    def test_output_range(self):
        for i in range(100):
            v = _hashed(float(i))
            assert 0.0 <= v < 1.0


class TestScoreQuality:
    def test_short_text(self):
        assert _score_quality("hi") == 0.0

    def test_good_prose(self):
        text = "The quick brown fox jumps over the lazy dog. It ran swiftly across the green meadow. Birds sang in the trees above."
        score = _score_quality(text)
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    def test_all_caps_low(self):
        text = "THIS IS ALL CAPS TEXT WITH MANY SENTENCES. IT SHOULD SCORE LOW. BECAUSE CAPS ARE ANNOYING."
        score = _score_quality(text)
        assert score < 0.8


class TestScoreRelevance:
    def test_empty_whitelist(self):
        assert _score_relevance("any text", []) == 1.0

    def test_matching_topic(self):
        score = _score_relevance("python programming is fun", ["python"])
        assert score > 0.0

    def test_no_match(self):
        score = _score_relevance("cats are cute", ["python"])
        assert score == 0.0


class TestMatchesBlacklist:
    def test_match(self):
        assert _matches_blacklist("buy now!!!", ["buy"]) is True

    def test_no_match(self):
        assert _matches_blacklist("hello world", ["buy"]) is False

    def test_empty(self):
        assert _matches_blacklist("anything", []) is False


class TestMatchesWhitelist:
    def test_match(self):
        assert _matches_whitelist("python programming", ["python"]) is True

    def test_no_match(self):
        assert _matches_whitelist("cats are cute", ["python"]) is False

    def test_empty_whitelist(self):
        assert _matches_whitelist("anything", []) is True
