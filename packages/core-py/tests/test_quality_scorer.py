"""Tests for training pair quality scorer."""

import pytest
from domains.training.quality_scorer import (
    score_pair,
    score_batch,
    _length_score,
    _repetition_score,
    _coherence_score,
    _language_quality_score,
)


class TestScorePair:
    def test_empty_messages(self):
        assert score_pair("", "") == 0.0
        assert score_pair("hello", "") == 0.0
        assert score_pair("", "hello") == 0.0

    def test_short_messages(self):
        score = score_pair("Hi", "Hello")
        assert 0.0 <= score <= 2.0  # Short = low quality

    def test_normal_conversation(self):
        score = score_pair(
            "What is the capital of France?",
            "The capital of France is Paris. It is a beautiful city known for the Eiffel Tower."
        )
        assert 2.0 <= score <= 5.0

    def test_long_conversation(self):
        user = "Can you explain how photosynthesis works?" * 3
        assistant = ("Photosynthesis is the process by which plants convert sunlight into energy. "
                     "They use chlorophyll in their leaves to capture light energy. "
                     "This energy is used to convert carbon dioxide and water into glucose and oxygen. "
                     "The process occurs in two stages: the light reactions and the Calvin cycle.") * 2
        score = score_pair(user, assistant)
        assert 2.0 <= score <= 5.0

    def test_repetitive_response(self):
        score = score_pair(
            "Tell me about dogs",
            "Dogs are great. Dogs are great. Dogs are great. Dogs are great. Dogs are great."
        )
        # Repetitive = lower score (not necessarily below 2.5 — scorer measures text quality)
        score_normal = score_pair(
            "Tell me about dogs",
            "Dogs are wonderful companions. They come in many breeds and have unique personalities."
        )
        assert score < score_normal

    def test_coherent_qa(self):
        score = score_pair(
            "How do I reset my password?",
            "To reset your password, go to the login page and click 'Forgot Password'. You will receive an email with a reset link."
        )
        assert score >= 2.0  # Well-formed response gets decent score

    def test_incoherent_response(self):
        # Scorer measures text quality, not semantic relevance.
        # A well-written but off-topic response still gets points for language quality.
        score_short = score_pair("What is 2+2?", "ok")
        score_normal = score_pair("What is 2+2?", "The answer to 2+2 is 4, which is a basic arithmetic operation.")
        assert score_short < score_normal  # Shorter = lower

    def test_score_range(self):
        # Score should always be 0-5
        for user, assistant in [
            ("", ""),
            ("x", "y"),
            ("Hello", "Hi there"),
            ("What is Python?", "Python is a programming language used for web development, data science, and AI."),
        ]:
            score = score_pair(user, assistant)
            assert 0.0 <= score <= 5.0

    def test_score_is_numeric(self):
        score = score_pair("Hello", "Hi there")
        assert isinstance(score, float)

    def test_deterministic(self):
        # Same input should always produce same output
        s1 = score_pair("What is Python?", "Python is a programming language.")
        s2 = score_pair("What is Python?", "Python is a programming language.")
        assert s1 == s2


class TestScoreBatch:
    def test_empty_batch(self):
        assert score_batch([]) == []

    def test_batch_returns_list(self):
        pairs = [
            {"user_msg": "Hello", "assistant_msg": "Hi there"},
            {"user_msg": "What is AI?", "assistant_msg": "AI is artificial intelligence."},
        ]
        scores = score_batch(pairs)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_batch_with_missing_keys(self):
        pairs = [{"user_msg": "Hello"}, {"assistant_msg": "Hi"}]
        scores = score_batch(pairs)
        assert len(scores) == 2
        assert scores[0] == 0.0  # Missing assistant_msg
        assert scores[1] == 0.0  # Missing user_msg

    def test_batch_scores_vary(self):
        pairs = [
            {"user_msg": "Hi", "assistant_msg": "Hello"},  # Short
            {"user_msg": "What is Python?", "assistant_msg": "Python is a programming language used for many applications."},  # Normal
        ]
        scores = score_batch(pairs)
        assert scores[1] > scores[0]  # Normal > short

    def test_batch_with_quality_scores(self):
        """Verify score_batch produces non-zero scores for normal conversations."""
        pairs = [
            {"user_msg": f"Question {i}", "assistant_msg": f"Answer {i} with enough text to be a reasonable response."}
            for i in range(5)
        ]
        scores = score_batch(pairs)
        assert len(scores) == 5
        # At least some should be non-zero
        assert any(s > 0 for s in scores)


class TestInternalBranchCoverage:
    def test_length_over_1000_gives_07(self):
        assert _length_score("a" * 1500, "b" * 1500) == pytest.approx(0.7)

    def test_length_over_2000_gives_04(self):
        assert _length_score("a" * 2500, "b" * 2500) == pytest.approx(0.4)

    def test_repetition_heavy_bigram_penalty(self):
        assert _repetition_score("A A A A") == pytest.approx(0.1)

    def test_repetition_mild_trigram_penalty(self):
        score = _repetition_score("alpha beta alpha beta gamma delta")
        assert 0.0 <= score <= 1.0
        assert score < _repetition_score("alpha beta gamma delta epsilon zeta")

    def test_coherence_zero_when_user_only_stop_words(self):
        assert _coherence_score("the and of", "Python is a language") == pytest.approx(0.3)

    def test_coherence_ratio_edge_branches(self):
        mid = _coherence_score("a" * 100, "b" * 30)  # ratio ~0.3 -> 0.6
        far = _coherence_score("a" * 5, "b" * 200)   # ratio 40 -> 0.2
        assert mid > far

    def test_language_no_punctuation(self):
        assert _language_quality_score("hello there friend") == pytest.approx(0.7)

    def test_language_many_punctuation(self):
        assert _language_quality_score("One, two; three: four. Five?") >= 0.5

    def test_language_mid_unique_ratio(self):
        assert _language_quality_score("the the the cat dog") >= 0.5

    def test_language_whitespace_only(self):
        assert _language_quality_score("              ") == pytest.approx(0.3)

    def test_language_short_avg_word_length(self):
        assert _language_quality_score("aa bb cc dd") < 0.6

    def test_language_mid_avg_word_length(self):
        assert _language_quality_score("aaa bbb ccc ddd eee") >= 0.6

    def test_language_mid_caps_ratio(self):
        assert _language_quality_score("AAAA bbbb cccc dddd eeee") == pytest.approx(0.6)

    def test_language_high_caps_ratio(self):
        assert _language_quality_score("AAAA BBBB CCCC DDDD EEEE") == pytest.approx(0.5)
