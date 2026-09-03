"""Tests for KnowledgeStorage adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from domains.learner.knowledge_storage import KnowledgeStorage


class TestKnowledgeStorage:
    def test_add_and_get_fact(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        fact_id = storage.add_fact(
            content="Python is a programming language",
            topic="tech",
            source="manual",
        )
        assert fact_id.startswith("fact_")
        fact = storage.get_fact(fact_id)
        assert fact is not None
        assert fact["content"] == "Python is a programming language"
        assert fact["topic"] == "tech"

    def test_list_facts(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        storage.add_fact(content="Fact 1", topic="tech")
        storage.add_fact(content="Fact 2", topic="food")
        storage.add_fact(content="Fact 3", topic="tech")

        all_facts = storage.list_facts()
        assert len(all_facts) == 3

        tech_facts = storage.list_facts(topic="tech")
        assert len(tech_facts) == 2

    def test_search_facts(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        storage.add_fact(content="Python rocks")
        storage.add_fact(content="Java is okay")
        results = storage.search_facts("Python")
        assert len(results) == 1
        assert results[0]["content"] == "Python rocks"

    def test_delete_fact(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        fact_id = storage.add_fact(content="To delete")
        assert storage.delete_fact(fact_id) is True
        assert storage.get_fact(fact_id) is None
        assert storage.delete_fact(fact_id) is False

    def test_count_facts(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        assert storage.count_facts() == 0
        storage.add_fact(content="A")
        storage.add_fact(content="B")
        assert storage.count_facts() == 2

    def test_visited_urls(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        storage.mark_visited("https://example.com")
        storage.mark_visited("https://python.org")
        visited = storage.get_visited()
        assert "https://example.com" in visited
        assert "https://python.org" in visited

    def test_feeds(self, tmp_path: Path):
        storage = KnowledgeStorage(tmp_path / "knowledge")
        storage.add_feed("https://example.com/rss", title="Example")
        feed = storage.get_feed("https://example.com/rss")
        assert feed is not None
        assert feed["title"] == "Example"

        feeds = storage.list_feeds()
        assert len(feeds) == 1

        storage.update_feed_last_fetched("https://example.com/rss")
        feed = storage.get_feed("https://example.com/rss")
        assert feed["last_fetched"] > 0

        assert storage.remove_feed("https://example.com/rss") is True
        assert storage.get_feed("https://example.com/rss") is None

    def test_persistence(self, tmp_path: Path):
        data_dir = tmp_path / "knowledge"
        storage1 = KnowledgeStorage(data_dir)
        fact_id = storage1.add_fact(content="Persisted fact")
        storage1.add_feed("https://persist.com/rss")

        storage2 = KnowledgeStorage(data_dir)
        fact = storage2.get_fact(fact_id)
        assert fact is not None
        assert fact["content"] == "Persisted fact"
        feeds = storage2.list_feeds()
        assert len(feeds) == 1
