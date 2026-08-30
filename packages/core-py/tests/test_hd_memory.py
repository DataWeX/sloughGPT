"""Tests for domains.soul.hd_memory — HDMemoryItem, HDMemoryStore."""

import pytest
import time
from domains.soul.hd_memory import HDMemoryItem, HDMemoryStore


class TestHDMemoryItem:
    def test_fields(self):
        item = HDMemoryItem(
            id="1", content="hello", hypervector=[0.1, 0.2],
            metadata={}, timestamp=1.0, role="user"
        )
        assert item.id == "1"
        assert item.content == "hello"
        assert item.role == "user"

    def test_all_fields(self):
        item = HDMemoryItem(
            id="2", content="world", hypervector=[0.3, 0.4, 0.5],
            metadata={"key": "val"}, timestamp=2.0, role="assistant"
        )
        assert item.content == "world"
        assert len(item.hypervector) == 3
        assert item.metadata["key"] == "val"
        assert item.timestamp == 2.0
        assert item.role == "assistant"

    def test_system_role(self):
        item = HDMemoryItem(
            id="3", content="sys", hypervector=[], metadata={},
            timestamp=0.0, role="system"
        )
        assert item.role == "system"

    def test_metadata_mutable(self):
        item = HDMemoryItem(
            id="4", content="c", hypervector=[], metadata={},
            timestamp=0.0, role="user"
        )
        item.metadata["new"] = "data"
        assert item.metadata["new"] == "data"

    def test_empty_content(self):
        item = HDMemoryItem(
            id="5", content="", hypervector=[], metadata={},
            timestamp=0.0, role="user"
        )
        assert item.content == ""

    def test_equality(self):
        a = HDMemoryItem(id="1", content="c", hypervector=[1], metadata={}, timestamp=0.0, role="user")
        b = HDMemoryItem(id="1", content="c", hypervector=[1], metadata={}, timestamp=0.0, role="user")
        assert a == b

    def test_repr(self):
        item = HDMemoryItem(id="6", content="c", hypervector=[], metadata={}, timestamp=0.0, role="user")
        r = repr(item)
        assert "6" in r


class TestHDMemoryStoreInit:
    def test_init(self):
        store = HDMemoryStore(dim=100, max_items=10)
        assert store.dim == 100
        assert store.max_items == 10

    def test_init_defaults(self):
        store = HDMemoryStore()
        assert store.dim == 10000
        assert store.max_items == 1000

    def test_init_empty_items(self):
        store = HDMemoryStore(dim=50, max_items=5)
        assert store.items == []

    def test_init_not_initialized(self):
        store = HDMemoryStore()
        assert store._initialized is False

    def test_init_empty_role_vectors(self):
        store = HDMemoryStore()
        assert store.role_vectors == {}


class TestHDMemoryStoreAdd:
    def test_add_and_search(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("hello world", role="user", metadata={})
        results = store.search("hello", top_k=5)
        assert len(results) >= 1

    def test_add_returns_id(self):
        store = HDMemoryStore(dim=100, max_items=10)
        item_id = store.add("test content", role="user")
        assert item_id.startswith("mem_")

    def test_add_multiple(self):
        store = HDMemoryStore(dim=100, max_items=10)
        id1 = store.add("first", role="user")
        id2 = store.add("second", role="assistant")
        assert id1 != id2
        assert len(store.items) == 2

    def test_add_with_metadata(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("content", role="user", metadata={"source": "chat"})
        assert store.items[0].metadata["source"] == "chat"

    def test_add_default_role(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("content")
        assert store.items[0].role == "user"

    def test_add_roles(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("u", role="user")
        store.add("a", role="assistant")
        store.add("s", role="system")
        assert store.items[0].role == "user"
        assert store.items[1].role == "assistant"
        assert store.items[2].role == "system"


class TestHDMemoryStoreMaxItems:
    def test_max_items(self):
        store = HDMemoryStore(dim=50, max_items=3)
        for i in range(5):
            store.add(f"item {i}", role="user", metadata={})
        assert len(store.items) <= 3

    def test_eviction_keeps_newest(self):
        store = HDMemoryStore(dim=50, max_items=2)
        store.add("old1", role="user")
        store.add("old2", role="user")
        store.add("new1", role="user")
        assert len(store.items) == 2
        contents = [item.content for item in store.items]
        assert "old1" not in contents
        assert "new1" in contents

    def test_no_eviction_under_limit(self):
        store = HDMemoryStore(dim=50, max_items=10)
        for i in range(5):
            store.add(f"item {i}", role="user")
        assert len(store.items) == 5

    def test_exact_limit(self):
        store = HDMemoryStore(dim=50, max_items=3)
        store.add("a", role="user")
        store.add("b", role="user")
        store.add("c", role="user")
        assert len(store.items) == 3

    def test_one_over_limit(self):
        store = HDMemoryStore(dim=50, max_items=3)
        store.add("a", role="user")
        store.add("b", role="user")
        store.add("c", role="user")
        store.add("d", role="user")
        assert len(store.items) == 3


class TestHDMemoryStoreSearch:
    def test_search_empty(self):
        store = HDMemoryStore(dim=100, max_items=10)
        results = store.search("query")
        assert results == []

    def test_search_returns_tuples(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("hello", role="user")
        results = store.search("hello")
        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 3

    def test_search_tuple_format(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("test", role="user")
        results = store.search("test")
        item_id, content, similarity = results[0]
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert isinstance(similarity, float)

    def test_search_top_k(self):
        store = HDMemoryStore(dim=100, max_items=20)
        for i in range(10):
            store.add(f"item {i}", role="user")
        results = store.search("item", top_k=3)
        assert len(results) <= 3

    def test_search_sorted_by_similarity(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("python programming language", role="user")
        store.add("java programming language", role="user")
        results = store.search("python")
        if len(results) >= 2:
            assert results[0][2] >= results[1][2]

    def test_search_role_filter(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("user content", role="user")
        store.add("assistant content", role="assistant")
        results = store.search("content", role_filter="user")
        for _, _, sim in results:
            assert sim is not None

    def test_search_exact_match(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("exact match test", role="user")
        results = store.search("exact match test")
        assert len(results) >= 1


class TestHDMemoryStoreStats:
    def test_stats(self):
        store = HDMemoryStore(dim=50, max_items=10)
        stats = store.get_stats()
        assert "total_items" in stats
        assert stats["total_items"] == 0

    def test_stats_with_items(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.add("b", role="assistant")
        stats = store.get_stats()
        assert stats["total_items"] == 2
        assert stats["max_items"] == 10
        assert stats["dimension"] == 50

    def test_stats_roles(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.add("b", role="user")
        store.add("c", role="assistant")
        stats = store.get_stats()
        assert stats["roles"]["user"] == 2
        assert stats["roles"]["assistant"] == 1

    def test_stats_initialized(self):
        store = HDMemoryStore(dim=50, max_items=10)
        stats = store.get_stats()
        assert stats["initialized"] is False

    def test_stats_keys(self):
        store = HDMemoryStore(dim=50, max_items=10)
        stats = store.get_stats()
        expected_keys = {"total_items", "max_items", "dimension", "initialized", "roles"}
        assert set(stats.keys()) == expected_keys


class TestHDMemoryStoreClear:
    def test_clear(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("test", role="user", metadata={})
        n = store.clear()
        assert n >= 0

    def test_clear_returns_count(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.add("b", role="user")
        count = store.clear()
        assert count == 2

    def test_clear_empties_store(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.add("b", role="user")
        store.clear()
        assert len(store.items) == 0

    def test_clear_empty_store(self):
        store = HDMemoryStore(dim=50, max_items=10)
        count = store.clear()
        assert count == 0

    def test_clear_allows_readd(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.clear()
        store.add("b", role="user")
        assert len(store.items) == 1


class TestHDMemoryStoreGetContext:
    def test_get_context(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("hello world", role="user", metadata={})
        ctx = store.get_context("hello")
        assert isinstance(ctx, str)

    def test_get_context_empty(self):
        store = HDMemoryStore(dim=50, max_items=10)
        ctx = store.get_context("anything")
        assert ctx == ""

    def test_get_context_has_content(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("test content here", role="user")
        ctx = store.get_context("test")
        assert "test content" in ctx

    def test_get_context_max_chars(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a" * 200, role="user")
        ctx = store.get_context("a", max_chars=50)
        assert len(ctx) <= 200

    def test_get_context_with_roles(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("content", role="user")
        ctx = store.get_context("content", include_roles=True)
        assert isinstance(ctx, str)

    def test_get_context_without_roles(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("content", role="user")
        ctx = store.get_context("content", include_roles=False)
        assert isinstance(ctx, str)


class TestHDMemoryStoreBundleRecent:
    def test_bundle_empty(self):
        store = HDMemoryStore(dim=50, max_items=10)
        result = store.bundle_recent()
        assert len(result) == 50
        assert all(v == 0 for v in result)

    def test_bundle_with_items(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.add("b", role="user")
        result = store.bundle_recent(n=5)
        assert len(result) == 50

    def test_bundle_respects_n(self):
        store = HDMemoryStore(dim=50, max_items=10)
        for i in range(10):
            store.add(f"item {i}", role="user")
        result = store.bundle_recent(n=3)
        assert len(result) == 50

    def test_bundle_default_n(self):
        store = HDMemoryStore(dim=50, max_items=10)
        for i in range(5):
            store.add(f"item {i}", role="user")
        result = store.bundle_recent()
        assert len(result) == 50


class TestHDMemoryStorePrune:
    def test_prune_empty(self):
        store = HDMemoryStore(dim=50, max_items=10)
        pruned = store.prune()
        assert pruned == 0

    def test_prune_single_item(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("only one", role="user")
        pruned = store.prune()
        assert pruned == 0

    def test_prune_no_duplicates(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("completely different text one", role="user")
        store.add("totally unrelated content two", role="user")
        pruned = store.prune(similarity_threshold=0.99)
        assert pruned == 0

    def test_prune_returns_int(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        result = store.prune()
        assert isinstance(result, int)


class TestHDMemoryStoreEdgeCases:
    def test_add_many_items(self):
        store = HDMemoryStore(dim=50, max_items=1000)
        for i in range(100):
            store.add(f"item {i}", role="user")
        assert len(store.items) == 100

    def test_search_after_clear(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("test", role="user")
        store.clear()
        results = store.search("test")
        assert results == []

    def test_get_stats_after_clear(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("a", role="user")
        store.clear()
        stats = store.get_stats()
        assert stats["total_items"] == 0

    def test_get_context_after_clear(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("test", role="user")
        store.clear()
        ctx = store.get_context("test")
        assert ctx == ""
