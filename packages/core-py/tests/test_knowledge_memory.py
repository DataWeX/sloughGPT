"""Tests for KnowledgeMemory — fact CRUD + search + context."""

import time
import pytest
from domains.learner.knowledge import KnowledgeMemory, KnowledgeFact


@pytest.fixture
def km():
    mem = KnowledgeMemory()
    mem.clear_all()
    return mem


class TestKnowledgeMemory:
    def test_add_fact_returns_true(self, km):
        fact = KnowledgeFact(content="Python is a language", topic="code", source="test", timestamp=time.time(), importance=0.8)
        assert km.add_fact(fact) is True

    def test_add_duplicate_content_returns_false(self, km):
        fact = KnowledgeFact(content="Duplicate content", topic="test", source="test", timestamp=time.time(), importance=0.5)
        assert km.add_fact(fact) is True
        assert km.add_fact(fact) is False

    def test_list_all_returns_added_facts(self, km):
        km.add_fact(KnowledgeFact(content="Fact A", topic="topic_a", source="test", timestamp=time.time(), importance=0.6))
        km.add_fact(KnowledgeFact(content="Fact B", topic="topic_b", source="test", timestamp=time.time(), importance=0.9))
        items = km.list_all()
        assert len(items) == 2
        contents = {i["content"] for i in items}
        assert "Fact A" in contents
        assert "Fact B" in contents

    def test_list_all_includes_all_fields(self, km):
        ts = time.time()
        km.add_fact(KnowledgeFact(content="Full fields", topic="test", source="manual", url="https://example.com", timestamp=ts, importance=0.7))
        items = km.list_all()
        item = items[0]
        assert item["id"]
        assert item["content"] == "Full fields"
        assert item["topic"] == "test"
        assert item["source"] == "manual"
        assert item["url"] == "https://example.com"
        assert item["importance"] == 0.7
        assert isinstance(item["score"], float)

    def test_delete_by_id(self, km):
        km.add_fact(KnowledgeFact(content="Delete me", topic="test", source="test", timestamp=time.time()))
        items = km.list_all()
        assert len(items) == 1
        item_id = items[0]["id"]
        assert km.delete_by_id(item_id) is True
        assert len(km.list_all()) == 0

    def test_delete_nonexistent_id_returns_false(self, km):
        assert km.delete_by_id("nonexistent-id") is False

    def test_delete_frees_content_hash(self, km):
        """After deleting, the same content can be re-added."""
        km.add_fact(KnowledgeFact(content="Re-addable content", topic="test", source="test", timestamp=time.time()))
        item_id = km.list_all()[0]["id"]
        km.delete_by_id(item_id)
        re_added = km.add_fact(KnowledgeFact(content="Re-addable content", topic="test", source="test", timestamp=time.time()))
        assert re_added is True

    def test_search_finds_relevant_facts(self, km):
        km.add_fact(KnowledgeFact(content="Python programming guide", topic="code", source="test", timestamp=time.time(), importance=0.8))
        km.add_fact(KnowledgeFact(content="Cooking recipes for pasta", topic="food", source="test", timestamp=time.time(), importance=0.5))
        results = km.search("Python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_search_empty_store_returns_empty(self, km):
        results = km.search("anything")
        assert results == []

    def test_query_by_topic(self, km):
        km.add_fact(KnowledgeFact(content="React hooks tutorial", topic="frontend", source="test", timestamp=time.time()))
        km.add_fact(KnowledgeFact(content="SQL joins explained", topic="database", source="test", timestamp=time.time()))
        results = km.query("frontend", top_k=10)
        assert len(results) >= 1
        assert all(r["topic"] == "frontend" for r in results)

    def test_get_context_string(self, km):
        km.add_fact(KnowledgeFact(content="Context fact", topic="test", source="test", timestamp=time.time(), importance=0.9))
        context = km.get_context_string(max_items=10)
        assert "[KNOWN_FACTS]" in context
        assert "Context fact" in context

    def test_clear_all_empties_store(self, km):
        km.add_fact(KnowledgeFact(content="Item to clear", topic="test", source="test", timestamp=time.time()))
        assert km.clear_all() == 1
        assert len(km.list_all()) == 0

    def test_clear_all_also_clears_visited(self, km):
        km.add_fact(KnowledgeFact(content="Clear visited test", topic="test", source="test", timestamp=time.time()))
        km.clear_all()
        re_added = km.add_fact(KnowledgeFact(content="Clear visited test", topic="test", source="test", timestamp=time.time()))
        assert re_added is True

    def test_stats_returns_dict(self, km):
        km.add_fact(KnowledgeFact(content="Stats fact", topic="test", source="test", timestamp=time.time()))
        stats = km.stats()
        assert stats["total_facts"] >= 1
        assert stats["visited_urls"] >= 1

    def test_many_facts(self, km):
        for i in range(50):
            km.add_fact(KnowledgeFact(content=f"Fact number {i}", topic="bulk", source="test", timestamp=time.time()))
        items = km.list_all(top_k=100)
        assert len(items) == 50

    def test_importance_scored_results(self, km):
        km.add_fact(KnowledgeFact(content="Low importance fact", topic="test", source="test", timestamp=time.time(), importance=0.1))
        km.add_fact(KnowledgeFact(content="High importance fact", topic="test", source="test", timestamp=time.time(), importance=0.9))
        results = km.query("test", top_k=10)
        assert len(results) >= 2
        # Should be sorted by importance descending
        assert results[0]["importance"] >= results[1]["importance"]


class TestAutoIngestFromChat:
    """Tests for auto_ingest_from_chat feature."""

    def test_extract_facts_from_text_with_facts(self):
        from domains.learner.knowledge import _extract_facts_from_text
        text = "Python is a programming language. It has dynamic typing. The current version is 3.12."
        facts = _extract_facts_from_text(text)
        assert len(facts) >= 1
        # Should extract at least one declarative fact
        assert any("Python" in f or "typing" in f or "version" in f for f in facts)

    def test_extract_facts_skips_questions(self):
        from domains.learner.knowledge import _extract_facts_from_text
        text = "What is Python? How does it work? Is it fast?"
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_extract_facts_skips_short(self):
        from domains.learner.knowledge import _extract_facts_from_text
        text = "Yes. No. Maybe."
        facts = _extract_facts_from_text(text)
        assert len(facts) == 0

    def test_extract_facts_empty(self):
        from domains.learner.knowledge import _extract_facts_from_text
        assert _extract_facts_from_text("") == []
        assert _extract_facts_from_text("short") == []

    def test_auto_ingest_adds_facts(self, km):
        response = "Python is a versatile programming language. It supports multiple paradigms. The language was created in 1991."
        added = km.auto_ingest_from_chat("Tell me about Python", response)
        assert added >= 1
        # Query by inferred topic (may be "programming" or "general")
        items = km.list_all()
        assert len(items) >= 1

    def test_auto_ingest_empty_response(self, km):
        added = km.auto_ingest_from_chat("Hi", "")
        assert added == 0

    def test_auto_ingest_respects_max_facts(self, km):
        response = "Fact one about science. Fact two about science. Fact three about science. Fact four about science."
        added = km.auto_ingest_from_chat("science facts", response, max_facts=2)
        assert added <= 2

    def test_auto_ingest_source_is_chat(self, km):
        km.auto_ingest_from_chat("test", "Python is a language used for web development.")
        items = km.list_all()
        chat_items = [i for i in items if i.get("source") == "chat"]
        assert len(chat_items) >= 1
