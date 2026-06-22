"""Tests for knowledge_augmenter — _needs_web_search, enrich_with_knowledge."""

import pytest
from unittest.mock import patch, MagicMock

from domains.learner.knowledge_augmenter import (
    _needs_web_search,
    enrich_with_knowledge,
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
            result = enrich_with_knowledge("Tell me something", auto_search=False)
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
            result = enrich_with_knowledge("Tell me facts", auto_search=False, max_facts=3)
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
