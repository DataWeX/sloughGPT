"""Tests for KnowledgeMemoryProvider — memory CRUD backed by KnowledgeMemory."""

import pytest
from unittest.mock import MagicMock, patch
from domains.memory.memory_provider import KnowledgeMemoryProvider


@pytest.fixture
def mock_store():
    """Fake KnowledgeMemory store."""
    store = MagicMock()
    store.auto_ingest_from_chat.return_value = 0
    store.ingest_from_chat.return_value = []
    store.add_fact.return_value = True
    store.search.return_value = []
    store.list_all.return_value = []
    store.clear_all.return_value = 0
    store.delete_by_id.return_value = False
    store.stats.return_value = {"total_facts": 0}
    return store


@pytest.fixture
def provider(mock_store):
    return KnowledgeMemoryProvider(store=mock_store)


class TestStoreTurn:
    def test_returns_false_on_empty_user(self, provider):
        assert provider.store_turn("", "response") is False

    def test_returns_false_on_empty_assistant(self, provider):
        assert provider.store_turn("user", "") is False

    def test_returns_false_on_both_empty(self, provider):
        assert provider.store_turn("", "") is False

    def test_delegates_to_auto_ingest(self, provider, mock_store):
        mock_store.auto_ingest_from_chat.return_value = 3
        result = provider.store_turn("hello", "world")
        mock_store.auto_ingest_from_chat.assert_called_once_with("hello", "world")
        assert result is True

    def test_returns_false_when_no_facts_extracted(self, provider, mock_store):
        mock_store.auto_ingest_from_chat.return_value = 0
        assert provider.store_turn("hi", "ok") is False

    def test_returns_true_when_facts_extracted(self, provider, mock_store):
        mock_store.auto_ingest_from_chat.return_value = 1
        assert provider.store_turn("tell me", "facts") is True

    def test_handles_store_exception(self, provider, mock_store):
        mock_store.auto_ingest_from_chat.side_effect = RuntimeError("db down")
        assert provider.store_turn("q", "a") is False


class TestStoreTurnFacts:
    def test_returns_stored_fact_texts(self, provider, mock_store):
        mock_store.ingest_from_chat.return_value = [
            "The capital of France is Paris, a major European city.",
            "The Seine runs through Paris.",
        ]
        result = provider.store_turn_facts("what is the capital", "response")
        mock_store.ingest_from_chat.assert_called_once_with("what is the capital", "response")
        assert result == [
            "The capital of France is Paris, a major European city.",
            "The Seine runs through Paris.",
        ]

    def test_empty_on_blank_inputs(self, provider):
        assert provider.store_turn_facts("", "response") == []
        assert provider.store_turn_facts("message", "") == []

    def test_empty_when_no_facts_extracted(self, provider, mock_store):
        mock_store.ingest_from_chat.return_value = []
        assert provider.store_turn_facts("hi", "ok") == []

    def test_handles_store_exception(self, provider, mock_store):
        mock_store.ingest_from_chat.side_effect = RuntimeError("db down")
        assert provider.store_turn_facts("q", "a") == []


class TestStore:
    def test_returns_false_on_empty_content(self, provider):
        assert provider.store("", "topic", "source") is False

    def test_returns_false_on_whitespace_content(self, provider):
        assert provider.store("   ", "topic", "source") is False

    def test_delegates_to_add_fact(self, provider, mock_store):
        mock_store.add_fact.return_value = True
        result = provider.store("fact text", "science", "manual")
        mock_store.add_fact.assert_called_once()
        call_args = mock_store.add_fact.call_args[0][0]
        assert call_args.content == "fact text"
        assert call_args.topic == "science"
        assert call_args.source == "manual"
        assert result is True

    def test_returns_false_on_store_failure(self, provider, mock_store):
        mock_store.add_fact.side_effect = Exception("oops")
        assert provider.store("fact", "t", "s") is False


class TestRetrieve:
    def test_delegates_to_search(self, provider, mock_store):
        mock_store.search.return_value = [{"content": "result"}]
        result = provider.retrieve("query", limit=5)
        mock_store.search.assert_called_once_with("query", top_k=5)
        assert len(result) == 1

    def test_returns_empty_on_exception(self, provider, mock_store):
        mock_store.search.side_effect = Exception("fail")
        assert provider.retrieve("q", 5) == []


class TestStats:
    def test_delegates_to_stats(self, provider, mock_store):
        mock_store.stats.return_value = {"total_facts": 42}
        assert provider.stats() == {"total_facts": 42}

    def test_returns_empty_on_exception(self, provider, mock_store):
        mock_store.stats.side_effect = Exception("err")
        assert provider.stats() == {}


class TestListAll:
    def test_delegates_to_list_all(self, provider, mock_store):
        mock_store.list_all.return_value = [{"content": "a"}]
        result = provider.list_all(limit=10)
        mock_store.list_all.assert_called_once_with(top_k=10)
        assert len(result) == 1

    def test_returns_empty_on_exception(self, provider, mock_store):
        mock_store.list_all.side_effect = Exception("err")
        assert provider.list_all(5) == []


class TestClear:
    def test_returns_zero_on_empty(self, provider, mock_store):
        mock_store.clear_all.return_value = 0
        assert provider.clear() == 0

    def test_returns_count(self, provider, mock_store):
        mock_store.clear_all.return_value = 5
        assert provider.clear() == 5

    def test_returns_zero_on_exception(self, provider, mock_store):
        mock_store.clear_all.side_effect = Exception("err")
        assert provider.clear() == 0


class TestDelete:
    def test_returns_zero_on_empty_ids(self, provider):
        assert provider.delete([]) == 0

    def test_delegates_to_delete_by_id(self, provider, mock_store):
        mock_store.delete_by_id.return_value = True
        result = provider.delete(["id1", "id2"])
        assert result == 2

    def test_counts_partial_deletes(self, provider, mock_store):
        mock_store.delete_by_id.side_effect = [True, False, True]
        result = provider.delete(["id1", "id2", "id3"])
        assert result == 2

    def test_handles_exception_per_id(self, provider, mock_store):
        mock_store.delete_by_id.side_effect = [True, Exception("fail"), True]
        result = provider.delete(["id1", "id2", "id3"])
        assert result == 2


class TestGetStore:
    def test_injected_store(self, provider, mock_store):
        assert provider._get_store() is mock_store

    def test_falls_back_to_singleton(self):
        p = KnowledgeMemoryProvider(store=None)
        with patch("domains.memory.memory_provider.get_knowledge_memory") as mock_get:
            mock_get.return_value = MagicMock()
            store = p._get_store()
            mock_get.assert_called_once()
            assert store is not None
