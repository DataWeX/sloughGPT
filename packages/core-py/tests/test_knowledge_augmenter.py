"""Tests for knowledge_augmenter — _needs_web_search, enrich_with_knowledge."""

import pytest
from unittest.mock import patch, MagicMock

from domains.learner.knowledge_augmenter import (
    _needs_web_search,
    _is_casual_small_talk,
    _content_tokens,
    _topically_related,
    enrich_with_knowledge,
    MIN_RELEVANCE_SCORE,
)


# ── _needs_web_search ────────────────────────────────────────────────────

class TestNeedsWebSearch:

    def test_casual_greeting_returns_false(self):
        assert _needs_web_search("hello") is False
        assert _needs_web_search("Hi") is False
        assert _needs_web_search("hey") is False
        assert _needs_web_search("sup") is False

    def test_casual_pattern_returns_false(self):
        assert _needs_web_search("how are you") is False
        assert _needs_web_search("How's it going?") is False
        assert _needs_web_search("how are things") is False
        assert _needs_web_search("what's up") is False

    def test_query_signal_returns_true(self):
        assert _needs_web_search("what is the capital of France") is True
        assert _needs_web_search("Who won the election?") is True
        assert _needs_web_search("latest news about AI") is True
        assert _needs_web_search("Explain quantum computing") is True
        assert _needs_web_search("compare Python and Rust") is True
        assert _needs_web_search("how do black holes form") is True

    def test_no_signal_returns_false(self):
        assert _needs_web_search("I like pizza") is False
        assert _needs_web_search("The sky is blue") is False
        assert _needs_web_search("That is interesting") is False
        assert _needs_web_search("I think therefore I am") is False

    def test_empty_string_returns_false(self):
        assert _needs_web_search("") is False

    def test_punctuation_is_stripped(self):
        assert _needs_web_search("hello!") is False
        assert _needs_web_search("what?") is True


# ── _is_casual_small_talk ─────────────────────────────────────────────────

class TestIsCasualSmallTalk:

    def test_exact_greeting(self):
        assert _is_casual_small_talk("hello") is True
        assert _is_casual_small_talk("Hi") is True
        assert _is_casual_small_talk("HEY") is True

    def test_greeting_with_punctuation(self):
        assert _is_casual_small_talk("hello!") is True
        assert _is_casual_small_talk("Hi?") is True

    def test_greeting_with_trailing_words(self):
        assert _is_casual_small_talk("hello there") is True
        assert _is_casual_small_talk("hi friend") is True
        assert _is_casual_small_talk("hey you") is True

    def test_casual_patterns(self):
        assert _is_casual_small_talk("how are you") is True
        assert _is_casual_small_talk("How's it going?") is True
        assert _is_casual_small_talk("good morning") is True
        assert _is_casual_small_talk("nice to meet you") is True

    def test_real_query_is_not_small_talk(self):
        assert _is_casual_small_talk("what is the capital of France") is False
        assert _is_casual_small_talk("Explain quantum computing") is False
        assert _is_casual_small_talk("what color is the sky") is False

    def test_empty_string(self):
        assert _is_casual_small_talk("") is False


# ── _content_tokens / _topically_related ────────────────────────────────

class TestTopicalOverlap:

    def test_content_tokens_exclude_short_function_words(self):
        assert _content_tokens("What color is the sky?") == {"what", "color"}
        assert _content_tokens("is it ok?") == set()

    def test_content_tokens_lowercase_and_alphanumeric(self):
        assert _content_tokens("Tell ME about Python!") == {"tell", "about", "python"}

    def test_topically_related_shared_content_word(self):
        assert _topically_related(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence.",
        ) is True

    def test_topically_related_function_word_overlap_only(self):
        assert _topically_related(
            "What color is the sky?",
            "Paris is the capital of France.",
        ) is False

    def test_topically_related_case_insensitive(self):
        assert _topically_related(
            "TELL ME ABOUT PYTHON",
            "Python is a programming language.",
        ) is True

    def test_topically_related_no_stemming(self):
        assert _topically_related("tell me facts", "Fact number one.") is False

    def test_topically_related_empty_query(self):
        assert _topically_related("", "Python is a programming language.") is False


# ── enrich_with_knowledge ────────────────────────────────────────────────

class TestEnrichWithKnowledge:

    def test_returns_none_when_no_facts(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Hi there", auto_search=False)
            assert result == {"facts": [], "source": "none", "topics": []}

    def test_returns_memory_facts_when_found(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Paris is the capital of France.", "score": 0.95},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is the capital of France?", auto_search=False)
            assert len(result["facts"]) == 1
            assert "Paris" in result["facts"][0]
            assert result["source"] == "memory"

    def test_sanitizes_short_facts(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "short", "score": 0.9},
            {"content": "This is a sufficiently long fact that should pass the length filter.", "score": 0.8},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Tell me a fact", auto_search=False)
            assert len(result["facts"]) == 1
            assert "sufficiently long" in result["facts"][0]

    def test_deduplicates_facts(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Water is H2O. " * 10, "score": 0.9},
            {"content": "Water is H2O. " * 10, "score": 0.85},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is water?", auto_search=False)
            assert len(result["facts"]) == 1

    def test_respects_max_facts(self):
        mock_memory = MagicMock()
        facts = [{"content": f"Fact number {i}. " * 10, "score": 1.0 - i * 0.1} for i in range(10)]
        mock_memory.search.return_value = facts
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Fact number", auto_search=False, max_facts=3)
            assert len(result["facts"]) == 3

    def test_web_search_triggered_when_no_facts_and_needs_search(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []  # no facts in memory
        mock_ingestor = MagicMock()

        with (
            patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory),
            patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor", return_value=mock_ingestor),
        ):
            mock_memory.search.side_effect = [
                [],  # first call — no facts
                [{"content": "Python is a programming language.", "score": 0.9}],  # second call after ingest
            ]
            result = enrich_with_knowledge("What is Python?", auto_search=True)
            mock_ingestor.search_and_ingest.assert_called_once()
            assert result["source"] == "web"
            assert len(result["facts"]) == 1

    def test_web_search_not_triggered_for_casual(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        mock_ingestor = MagicMock()

        with (
            patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory),
            patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor", return_value=mock_ingestor),
        ):
            result = enrich_with_knowledge("Hello there!", auto_search=True)
            mock_ingestor.search_and_ingest.assert_not_called()
            assert result["source"] == "none"

    def test_auto_search_false_skips_web(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is AI?", auto_search=False)
            assert result["source"] == "none"

    def test_web_search_failure_graceful_fallback(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []
        mock_ingestor = MagicMock()
        mock_ingestor.search_and_ingest.side_effect = RuntimeError("API down")

        with (
            patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory),
            patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor", return_value=mock_ingestor),
        ):
            result = enrich_with_knowledge("What happened today?", auto_search=True)
            assert result == {"facts": [], "source": "none", "topics": []}

    # ── relevance gate ──────────────────────────────────────────────────

    def test_casual_greeting_skips_memory_search(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Paris is the capital of France. " * 3, "score": 0.95},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Hello!", auto_search=False)
            assert result == {"facts": [], "source": "none", "topics": []}
            mock_memory.search.assert_not_called()

    def test_casual_small_talk_skips_memory_search(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Machine learning is a subset of artificial intelligence. " * 2, "score": 0.9},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("How are you?", auto_search=False)
            assert result["source"] == "none"
            mock_memory.search.assert_not_called()

    def test_low_score_fact_is_filtered_out(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Machine learning is a subset of artificial intelligence. " * 2, "score": 0.05},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is machine learning?", auto_search=False)
            assert result == {"facts": [], "source": "none", "topics": []}

    def test_high_score_fact_is_kept(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Machine learning is a subset of artificial intelligence. " * 2, "score": 0.7},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is machine learning?", auto_search=False)
            assert result["source"] == "memory"
            assert len(result["facts"]) == 1

    def test_score_at_floor_is_kept(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Python is a programming language. " * 3, "score": MIN_RELEVANCE_SCORE},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Tell me about Python", auto_search=False)
            assert result["source"] == "memory"

    def test_custom_min_score_override(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Python is a programming language. " * 3, "score": 0.2},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Tell me about Python", auto_search=False, min_score=0.3)
            assert result["source"] == "none"

    def test_mixed_scores_filtered_by_relevance(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Paris is the capital of France. " * 3, "score": 0.733},
            {"content": "Python is a programming language. " * 3, "score": 0.08},
            {"content": "Machine learning is a subset of AI. " * 2, "score": 0.0},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Is Paris the capital of France?", auto_search=False)
            assert result["source"] == "memory"
            assert len(result["facts"]) == 1
            assert "Paris" in result["facts"][0]

    def test_web_results_also_filtered_by_score(self):
        mock_memory = MagicMock()
        mock_ingestor = MagicMock()
        mock_memory.search.side_effect = [
            [],  # first call — no facts
            [{"content": "Python is a programming language. " * 3, "score": 0.02}],  # post-ingest, weak
        ]
        with (
            patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory),
            patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor", return_value=mock_ingestor),
        ):
            result = enrich_with_knowledge("What is Python?", auto_search=True)
            mock_ingestor.search_and_ingest.assert_called_once()
            assert result["source"] == "none"

    # ── topical-overlap gate ─────────────────────────────────────────────

    def test_no_shared_content_token_skips_score_passing_fact(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Paris is the capital of France. " * 3, "score": 0.224},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What color is the sky?", auto_search=False)
            assert result == {"facts": [], "source": "none", "topics": []}

    def test_shared_content_token_keeps_score_passing_fact(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Machine learning is a subset of artificial intelligence. " * 2, "score": 0.224},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("What is machine learning?", auto_search=False)
            assert result["source"] == "memory"
            assert len(result["facts"]) == 1

    def test_query_without_content_tokens_skips_fact(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Python is a programming language. " * 3, "score": 0.9},
        ]
        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory):
            result = enrich_with_knowledge("Is it ok?", auto_search=False)
            assert result["source"] == "none"

    def test_web_results_also_gated_by_topical_overlap(self):
        mock_memory = MagicMock()
        mock_ingestor = MagicMock()
        mock_memory.search.side_effect = [
            [],  # first call — no facts
            [{"content": "Paris is the capital of France. " * 3, "score": 0.9}],  # post-ingest, unrelated
        ]
        with (
            patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=mock_memory),
            patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor", return_value=mock_ingestor),
        ):
            result = enrich_with_knowledge("What color is the sky?", auto_search=True)
            mock_ingestor.search_and_ingest.assert_called_once()
            assert result["source"] == "none"
