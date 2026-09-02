"""Tests for domains.learner.knowledge_storage — KnowledgeStorage adapter."""

from __future__ import annotations

import pytest
from pathlib import Path

from domains.learner.knowledge_storage import KnowledgeStorage


@pytest.fixture
def storage(tmp_path):
    return KnowledgeStorage(tmp_path / "knowledge")


# ── Facts ─────────────────────────────────────────────────────────────────────

class TestFacts:
    def test_add_and_get(self, storage):
        fid = storage.add_fact("Python is great", topic="python", source="manual")
        assert fid.startswith("fact_")
        fact = storage.get_fact(fid)
        assert fact is not None
        assert fact["content"] == "Python is great"
        assert fact["topic"] == "python"

    def test_add_with_defaults(self, storage):
        fid = storage.add_fact("A fact")
        fact = storage.get_fact(fid)
        assert fact["topic"] == "general"
        assert fact["source"] == "manual"
        assert fact["importance"] == 0.5

    def test_add_with_tags(self, storage):
        fid = storage.add_fact("Tagged", tags=["important", "todo"])
        fact = storage.get_fact(fid)
        assert "important" in fact["tags"]

    def test_get_nonexistent(self, storage):
        assert storage.get_fact("nonexistent") is None

    def test_list_all(self, storage):
        storage.add_fact("Fact 1")
        storage.add_fact("Fact 2")
        storage.add_fact("Fact 3")
        facts = storage.list_facts()
        assert len(facts) == 3

    def test_list_by_topic(self, storage):
        storage.add_fact("Py fact", topic="python")
        storage.add_fact("Rust fact", topic="rust")
        storage.add_fact("Py tip", topic="python")
        py_facts = storage.list_facts(topic="python")
        assert len(py_facts) == 2

    def test_search(self, storage):
        storage.add_fact("Python is great")
        storage.add_fact("Rust is fast")
        results = storage.search_facts("python")
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_delete(self, storage):
        fid = storage.add_fact("To delete")
        assert storage.delete_fact(fid) is True
        assert storage.get_fact(fid) is None

    def test_delete_nonexistent(self, storage):
        assert storage.delete_fact("nope") is False

    def test_count(self, storage):
        assert storage.count_facts() == 0
        storage.add_fact("One")
        storage.add_fact("Two")
        assert storage.count_facts() == 2


# ── Visited URLs ──────────────────────────────────────────────────────────────

class TestVisited:
    def test_mark_and_get_visited(self, storage):
        storage.mark_visited("https://example.com")
        visited = storage.get_visited()
        assert "https://example.com" in visited

    def test_mark_visited_creates_fact(self, storage):
        storage.mark_visited("https://example.com")
        facts = storage.list_facts(topic="visited")
        assert len(facts) == 1
        assert "Visited:" in facts[0]["content"]

    def test_get_visited_empty(self, storage):
        assert storage.get_visited() == []

    def test_multiple_visited(self, storage):
        storage.mark_visited("https://a.com")
        storage.mark_visited("https://b.com")
        visited = storage.get_visited()
        assert len(visited) == 2


# ── Feeds ─────────────────────────────────────────────────────────────────────

class TestFeeds:
    def test_add_and_get_feed(self, storage):
        storage.add_feed("https://feed.xml", title="My Feed")
        feed = storage.get_feed("https://feed.xml")
        assert feed is not None
        assert feed["title"] == "My Feed"
        assert feed["enabled"] is True

    def test_add_feed_default_interval(self, storage):
        storage.add_feed("https://feed.xml")
        feed = storage.get_feed("https://feed.xml")
        assert feed["poll_interval"] == 3600.0

    def test_get_feed_nonexistent(self, storage):
        assert storage.get_feed("https://nope.xml") is None

    def test_list_feeds(self, storage):
        storage.add_feed("https://a.xml", title="A")
        storage.add_feed("https://b.xml", title="B")
        feeds = storage.list_feeds()
        assert len(feeds) == 2

    def test_update_feed_last_fetched(self, storage):
        storage.add_feed("https://feed.xml")
        storage.update_feed_last_fetched("https://feed.xml")
        feed = storage.get_feed("https://feed.xml")
        assert feed["last_fetched"] > 0

    def test_remove_feed(self, storage):
        storage.add_feed("https://feed.xml")
        assert storage.remove_feed("https://feed.xml") is True
        assert storage.get_feed("https://feed.xml") is None

    def test_remove_feed_nonexistent(self, storage):
        assert storage.remove_feed("https://nope.xml") is False


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_fact_persistence(self, tmp_path):
        s1 = KnowledgeStorage(tmp_path / "k")
        fid = s1.add_fact("Persistent fact")
        
        s2 = KnowledgeStorage(tmp_path / "k")
        fact = s2.get_fact(fid)
        assert fact is not None
        assert fact["content"] == "Persistent fact"

    def test_feed_persistence(self, tmp_path):
        s1 = KnowledgeStorage(tmp_path / "k")
        s1.add_feed("https://feed.xml", title="Persistent")
        
        s2 = KnowledgeStorage(tmp_path / "k")
        feed = s2.get_feed("https://feed.xml")
        assert feed is not None
        assert feed["title"] == "Persistent"
