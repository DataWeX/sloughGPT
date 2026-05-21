"""Tests for knowledge_augmenter — chat enrichment with vector-retrieved facts."""
import pytest
from unittest.mock import patch, MagicMock
from domains.learner.knowledge_augmenter import (
    _needs_web_search,
    enrich_with_knowledge,
)


class TestNeedsWebSearch:
    def test_what_question(self):
        assert _needs_web_search("What is the weather?") is True

    def test_explain(self):
        assert _needs_web_search("Explain quantum physics") is True

    def test_news(self):
        assert _needs_web_search("latest news on AI") is True

    def test_casual_chat(self):
        assert _needs_web_search("Hello, how are you?") is False

    def test_statement(self):
        assert _needs_web_search("I like programming.") is False

    def test_empty(self):
        assert _needs_web_search("") is False


class TestEnrichWithKnowledge:
    def test_returns_facts_from_memory(self, monkeypatch):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "Python is a programming language created in 1991."},
        ]

        def mock_get_memory():
            return mock_memory

        monkeypatch.setattr(
            "domains.learner.knowledge_augmenter.get_knowledge_memory",
            mock_get_memory,
        )

        result = enrich_with_knowledge("Tell me about Python", auto_search=False)
        assert result["source"] == "memory"
        assert len(result["facts"]) > 0
        assert "Python" in result["facts"][0]

    def test_no_facts_returns_none(self, monkeypatch):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []

        def mock_get_memory():
            return mock_memory

        monkeypatch.setattr(
            "domains.learner.knowledge_augmenter.get_knowledge_memory",
            mock_get_memory,
        )

        result = enrich_with_knowledge("Hello", auto_search=False)
        assert result["source"] == "none"
        assert result["facts"] == []

    def test_auto_search_casual_does_not_trigger(self, monkeypatch):
        mock_memory = MagicMock()
        mock_memory.search.return_value = []

        def mock_get_memory():
            return mock_memory

        monkeypatch.setattr(
            "domains.learner.knowledge_augmenter.get_knowledge_memory",
            mock_get_memory,
        )

        result = enrich_with_knowledge("Hello, how are you?", auto_search=True)
        assert result["source"] == "none"

    def test_memory_filters_short_content(self, monkeypatch):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"content": "short"},
            {"content": "This is a sufficiently long fact about something interesting."},
        ]

        def mock_get_memory():
            return mock_memory

        monkeypatch.setattr(
            "domains.learner.knowledge_augmenter.get_knowledge_memory",
            mock_get_memory,
        )

        result = enrich_with_knowledge("test", auto_search=False)
        assert result["source"] == "memory"
        assert "short" not in result["facts"][0]
