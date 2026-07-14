"""Tests for training pair quality scorer."""

import pytest
from domains.training.quality_scorer import score_pair, score_batch


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
        assert score < 2.5  # Repetitive = low quality

    def test_coherent_qa(self):
        score = score_pair(
            "How do I reset my password?",
            "To reset your password, go to the login page and click 'Forgot Password'. You will receive an email with a reset link."
        )
        assert score >= 3.0

    def test_incoherent_response(self):
        score = score_pair(
            "What is 2+2?",
            "The weather is nice today and I like pizza with extra cheese on top."
        )
        assert score < 3.0

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
