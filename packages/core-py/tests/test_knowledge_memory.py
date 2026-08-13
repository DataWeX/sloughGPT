"""Tests for KnowledgeMemory — fact CRUD + search + context."""

import json
import time
import pytest
from domains.learner.knowledge import KnowledgeMemory, KnowledgeFact


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Keep persistence off the real data dir (repo-root anchored)."""
    from domains.learner import knowledge as K
    monkeypatch.setattr(K, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(K, "FEED_STATE_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(K, "VISITED_PATH", tmp_path / "visited.json")
    monkeypatch.setattr(K, "ENTRIES_PATH", tmp_path / "entries.json")


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
        fact = KnowledgeFact(content="The earth is round", topic="science", source="test")
        assert km.add_fact(fact) is True
        assert km.add_fact(fact) is False


class TestAddFactsBatch:
    def test_add_facts_returns_new_count(self, km):
        facts = [
            KnowledgeFact(content=f"Batch fact {i} with enough length", topic="bulk", source="test")
            for i in range(3)
        ]
        assert km.add_facts(facts) == 3
        assert len(km.list_all()) == 3

    def test_add_facts_empty_batch(self, km):
        assert km.add_facts([]) == 0

    def test_add_facts_dedup_within_batch(self, km):
        fact = KnowledgeFact(content="identical batch fact content", topic="t", source="test")
        assert km.add_facts([fact, fact]) == 1

    def test_add_facts_dedup_against_existing(self, km):
        km.add_fact(KnowledgeFact(content="existing fact content here", topic="t", source="test"))
        assert km.add_facts([KnowledgeFact(content="existing fact content here", topic="t", source="test")]) == 0

    def test_add_facts_precomputed_vectors_skip_embedding(self, km, monkeypatch):
        def boom(text):
            raise AssertionError("embedding should be skipped when vectors are precomputed")

        monkeypatch.setattr(km, "_get_embedding", boom)
        facts = [
            KnowledgeFact(content=f"precomputed fact {i} long enough", topic="t", source="test")
            for i in range(2)
        ]
        vecs = [[0.5] * km._vector_store.dimension for _ in facts]
        assert km.add_facts(facts, vectors=vecs) == 2

    def test_add_facts_persists_store_once(self, km, monkeypatch):
        saves = []
        monkeypatch.setattr(km, "_save_entries", lambda: saves.append(1))
        facts = [
            KnowledgeFact(content=f"persist fact {i} long enough", topic="t", source="test")
            for i in range(5)
        ]
        assert km.add_facts(facts) == 5
        assert len(saves) == 1

    def test_add_facts_requires_aligned_vectors(self, km):
        with pytest.raises(ValueError):
            km.add_facts(
                [KnowledgeFact(content="x", topic="t", source="s")],
                vectors=[[0.5, 0.5], [0.5, 0.5]],
            )

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

    def test_update_fact_changes_content_and_topic(self, km):
        km.add_fact(KnowledgeFact(content="Original fact text", topic="old_topic", source="test", timestamp=time.time()))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Edited fact text", topic="new_topic") is True
        items = km.list_all()
        assert len(items) == 1
        assert items[0]["content"] == "Edited fact text"
        assert items[0]["topic"] == "new_topic"
        assert items[0]["id"] == item_id

    def test_update_fact_keeps_topic_when_omitted(self, km):
        km.add_fact(KnowledgeFact(content="Keep topic fact", topic="retained", source="test", timestamp=time.time()))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Keep topic fact v2") is True
        items = km.list_all()
        assert items[0]["content"] == "Keep topic fact v2"
        assert items[0]["topic"] == "retained"

    def test_update_fact_unknown_id_returns_false(self, km):
        km.add_fact(KnowledgeFact(content="Some fact", topic="test", source="test", timestamp=time.time()))
        assert km.update_fact("nonexistent-id", "New text") is False
        assert km.list_all()[0]["content"] == "Some fact"

    def test_update_fact_empty_content_returns_false(self, km):
        km.add_fact(KnowledgeFact(content="Some fact", topic="test", source="test", timestamp=time.time()))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "   ") is False
        assert km.list_all()[0]["content"] == "Some fact"

    def test_update_fact_to_duplicate_returns_false(self, km):
        km.add_fact(KnowledgeFact(content="First fact", topic="test", source="test", timestamp=time.time()))
        km.add_fact(KnowledgeFact(content="Second fact", topic="test", source="test", timestamp=time.time()))
        first_id = km.list_all()[0]["id"]
        assert km.update_fact(first_id, "Second fact") is False
        assert km.list_all()[0]["content"] == "First fact"

    def test_updated_fact_can_be_restored_to_original_content(self, km):
        km.add_fact(KnowledgeFact(content="Original", topic="test", source="test", timestamp=time.time()))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Revised") is True
        assert km.update_fact(item_id, "Original") is True
        assert km.list_all()[0]["content"] == "Original"

    def test_update_fact_changes_importance(self, km):
        km.add_fact(KnowledgeFact(content="Scored fact", topic="test", source="test", timestamp=time.time(), importance=0.5))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Scored fact", importance=0.9) is True
        assert km.list_all()[0]["importance"] == 0.9

    def test_update_fact_keeps_importance_when_omitted(self, km):
        km.add_fact(KnowledgeFact(content="Scored fact", topic="test", source="test", timestamp=time.time(), importance=0.5))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Scored fact v2") is True
        assert km.list_all()[0]["importance"] == 0.5

    def test_update_fact_clamps_importance_out_of_range(self, km):
        km.add_fact(KnowledgeFact(content="Scored fact", topic="test", source="test", timestamp=time.time(), importance=0.5))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Scored fact", importance=1.5) is True
        assert km.list_all()[0]["importance"] == 1.0
        assert km.update_fact(item_id, "Scored fact", importance=-0.2) is True
        assert km.list_all()[0]["importance"] == 0.0

    def test_update_fact_importance_only_leaves_text_untouched(self, km):
        km.add_fact(KnowledgeFact(content="Same text", topic="retained", source="test", timestamp=time.time(), importance=0.5))
        item_id = km.list_all()[0]["id"]
        assert km.update_fact(item_id, "Same text", importance=0.7) is True
        items = km.list_all()
        assert len(items) == 1
        assert items[0]["content"] == "Same text"
        assert items[0]["topic"] == "retained"
        assert items[0]["importance"] == 0.7

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

    def test_clear_all_persists_empty_file(self, km):
        from domains.learner import knowledge as K
        km.add_fact(KnowledgeFact(content="Persist clear test", topic="test", source="test", timestamp=time.time()))
        assert K.ENTRIES_PATH.exists()
        assert km.clear_all() == 1
        assert K.ENTRIES_PATH.exists()
        assert json.loads(K.ENTRIES_PATH.read_text()) == []

    def test_clear_all_survives_restart(self, km):
        km.add_fact(KnowledgeFact(content="Restart clear test", topic="test", source="test", timestamp=time.time()))
        assert km.clear_all() == 1
        fresh = KnowledgeMemory()
        assert len(fresh.list_all()) == 0


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
