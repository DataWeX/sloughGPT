"""Tests for domains.soul.hd_memory: hyperdimensional memory store."""

import random

import pytest

from domains.soul.hd_memory import HDMemoryItem, HDMemoryStore

DIM = 64


@pytest.fixture
def store():
    random.seed(42)
    return HDMemoryStore(dim=DIM, max_items=1000)


class TestHDMemoryItem:
    def test_fields(self):
        item = HDMemoryItem(
            id="mem_1", content="hi", hypervector=[1.0], metadata={"k": "v"},
            timestamp=1.0, role="user",
        )
        assert item.id == "mem_1"
        assert item.content == "hi"
        assert item.role == "user"
        assert item.metadata == {"k": "v"}


class TestEncode:
    def test_encode_content_dimension(self, store):
        assert len(store.encode_content("hello world")) == DIM

    def test_encode_content_deterministic(self, store):
        assert store.encode_content("repeat me") == store.encode_content("repeat me")

    def test_encode_with_role_dimension(self, store):
        assert len(store.encode_with_role("hello world", "user")) == DIM

    def test_roles_initialized(self, store):
        store.encode_content("trigger init")
        assert set(store.role_vectors) == {"user", "assistant", "system"}


class TestAdd:
    def test_returns_id(self, store):
        item_id = store.add("hello there", role="user")
        assert item_id.startswith("mem_")

    def test_stores_item(self, store):
        store.add("hello there", role="assistant", metadata={"model": "gpt2"})
        item = store.items[0]
        assert item.content == "hello there"
        assert item.role == "assistant"
        assert item.metadata == {"model": "gpt2"}
        assert item.timestamp > 0
        assert len(item.hypervector) == DIM

    def test_default_metadata(self, store):
        store.add("hello")
        assert store.items[0].metadata == {}

    def test_evicts_oldest_over_capacity(self):
        random.seed(7)
        store = HDMemoryStore(dim=DIM, max_items=2)
        store.add("first")
        first_id = store.items[0].id
        store.add("second")
        store.add("third")
        assert len(store.items) == 2
        assert all(item.id != first_id for item in store.items)


class TestSearch:
    def test_empty(self, store):
        assert store.search("anything") == []

    def test_exact_match_top_result(self, store):
        store.add("the sky is blue today", role="user")
        store.add("I love pizza with cheese", role="assistant")
        results = store.search("the sky is blue today", top_k=5)
        assert results[0][1] == "the sky is blue today"
        assert results[0][2] == pytest.approx(1.0, abs=1e-6)

    def test_sorted_descending(self, store):
        store.add("alpha beta gamma delta")
        store.add("alpha beta gamma delta")
        results = store.search("alpha beta gamma delta", top_k=5)
        assert results[0][2] >= results[1][2]

    def test_role_filter(self, store):
        store.add("shared words content here", role="user")
        store.add("shared words content here", role="assistant")
        results = store.search("shared words content here", top_k=5, role_filter="assistant")
        assert len(results) == 1
        assert results[0][0] == store.items[1].id

    def test_top_k_limit(self, store):
        for i in range(5):
            store.add(f"query token number {i}")
        results = store.search("query token number", top_k=2)
        assert len(results) == 2

    def test_result_shape(self, store):
        store.add("hello world")
        result = store.search("hello world")[0]
        assert len(result) == 3
        item_id, content, sim = result
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert isinstance(sim, float)


class TestGetContext:
    def test_empty_when_no_items(self, store):
        assert store.get_context("anything") == ""

    def test_include_roles(self, store):
        store.add("the sky is blue today", role="user")
        context = store.get_context("the sky is blue today")
        assert context == "[User]: the sky is blue today"

    def test_no_roles(self, store):
        store.add("the sky is blue today", role="user")
        context = store.get_context("the sky is blue today", include_roles=False)
        assert context == "the sky is blue today"

    def test_respects_max_chars(self, store):
        store.add("the sky is blue today", role="user")
        context = store.get_context("the sky is blue today", max_chars=10)
        assert len(context) <= 10

    def test_filters_low_similarity(self, store, monkeypatch):
        store.add("the sky is blue today", role="user")
        monkeypatch.setattr(store, "search", lambda *a, **k: [("mem_x", "low sim", 0.005)])
        assert store.get_context("anything") == ""


class TestBundleRecent:
    def test_empty_returns_zeros(self, store):
        assert store.bundle_recent(5) == [0] * DIM

    def test_bundles_items(self, store):
        store.add("one two three")
        store.add("four five six")
        vector = store.bundle_recent(10)
        assert len(vector) == DIM
        assert all(x in (-1.0, 1.0) for x in vector)

    def test_bundles_last_n(self, store):
        for i in range(5):
            store.add(f"item number {i}")
        assert len(store.bundle_recent(2)) == DIM


class TestStatsAndClear:
    def test_stats(self, store):
        store.add("hello", role="user")
        store.add("hi", role="assistant")
        stats = store.get_stats()
        assert stats["total_items"] == 2
        assert stats["max_items"] == 1000
        assert stats["dimension"] == DIM
        assert stats["initialized"] is True
        assert stats["roles"] == {"user": 1, "assistant": 1}

    def test_stats_before_init(self):
        store = HDMemoryStore(dim=DIM)
        stats = store.get_stats()
        assert stats["initialized"] is False
        assert stats["total_items"] == 0

    def test_clear(self, store):
        store.add("hello")
        store.add("world")
        assert store.clear() == 2
        assert store.items == []

    def test_clear_empty(self, store):
        assert store.clear() == 0


class TestPrune:
    def test_duplicates_pruned(self, store):
        store.add("the sky is blue today", role="user")
        store.add("the sky is blue today", role="user")
        assert store.prune(similarity_threshold=0.95) == 1
        assert len(store.items) == 1

    def test_distinct_kept(self, store):
        store.add("the sky is blue today", role="user")
        store.add("I love pizza with cheese", role="assistant")
        assert store.prune(similarity_threshold=0.95) == 0
        assert len(store.items) == 2

    def test_single_item_no_prune(self, store):
        store.add("only one item here")
        assert store.prune() == 0

    def test_empty_no_prune(self, store):
        assert store.prune() == 0
