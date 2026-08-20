"""Meaningful tests for knowledge_augmenter — small talk detection, web search need, content tokens, topical relatedness."""

import pytest
from domains.learner.knowledge_augmenter import (
    _is_casual_small_talk, _needs_web_search, _content_tokens,
    _topically_related, MIN_RELEVANCE_SCORE, _QUERY_SIGNALS,
    _CASUAL_GREETINGS, _CASUAL_PATTERNS,
)


class TestIsCasualSmallTalk:
    def test_hello(self):
        assert _is_casual_small_talk("Hello!") is True

    def test_hi(self):
        assert _is_casual_small_talk("Hi") is True

    def test_hey(self):
        assert _is_casual_small_talk("Hey there") is True

    def test_how_are_you(self):
        assert _is_casual_small_talk("How are you?") is True

    def test_whats_up(self):
        assert _is_casual_small_talk("What's up?") is True

    def test_good_morning(self):
        assert _is_casual_small_talk("Good morning!") is True

    def test_nice_to_meet(self):
        assert _is_casual_small_talk("Nice to meet you") is True

    def test_not_casual(self):
        assert _is_casual_small_talk("What is the capital of France?") is False

    def test_empty(self):
        assert _is_casual_small_talk("") is False

    def test_greeting_with_punctuation(self):
        assert _is_casual_small_talk("Hello, how are you?") is True


class TestNeedsWebSearch:
    def test_needs_search(self):
        assert _needs_web_search("What is the latest news about AI?") is True

    def test_how_to(self):
        assert _needs_web_search("How do I train a model?") is True

    def test_compare(self):
        assert _needs_web_search("Compare Python and Rust") is True

    def test_no_search_for_greeting(self):
        assert _needs_web_search("Hello") is False

    def test_no_search_for_casual(self):
        assert _needs_web_search("How are you?") is False

    def test_no_search_for_factual(self):
        assert _needs_web_search("2 + 2 equals 4") is False


class TestContentTokens:
    def test_long_words(self):
        tokens = _content_tokens("Python is a programming language")
        assert "python" in tokens
        assert "programming" in tokens
        assert "language" in tokens

    def test_short_words_excluded(self):
        tokens = _content_tokens("is the of to for")
        assert len(tokens) == 0

    def test_mixed(self):
        tokens = _content_tokens("the quick brown foxes jump")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "foxes" in tokens
        assert "the" not in tokens

    def test_numbers(self):
        tokens = _content_tokens("version 2024 release")
        assert "2024" in tokens

    def test_empty(self):
        tokens = _content_tokens("")
        assert len(tokens) == 0


class TestTopicallyRelated:
    def test_related(self):
        assert _topically_related(
            "What is Python?",
            "Python is a programming language"
        ) is True

    def test_not_related(self):
        assert _topically_related(
            "What color is the sky?",
            "Paris is the capital of France"
        ) is False

    def test_shared_short_words_not_enough(self):
        assert _topically_related(
            "is the",
            "is the"
        ) is False

    def test_empty(self):
        assert _topically_related("", "") is False


class TestConstants:
    def test_min_relevance_score(self):
        assert 0.0 < MIN_RELEVANCE_SCORE < 1.0

    def test_query_signals(self):
        assert "what" in _QUERY_SIGNALS
        assert "latest" in _QUERY_SIGNALS
        assert len(_QUERY_SIGNALS) >= 10

    def test_casual_greetings(self):
        assert "hello" in _CASUAL_GREETINGS
        assert "hi" in _CASUAL_GREETINGS

    def test_casual_patterns(self):
        assert "how are you" in _CASUAL_PATTERNS
        assert "good morning" in _CASUAL_PATTERNS
