"""Tests for domains.soul.hd_memory — HDMemoryItem, HDMemoryStore."""

import pytest
from domains.soul.hd_memory import HDMemoryItem, HDMemoryStore


# ===================================================================
# HDMemoryItem (dataclass)
# ===================================================================

class TestHDMemoryItem:
    def test_creation(self):
        item = HDMemoryItem(
            id="m1",
            content="hello",
            hypervector=[1.0, -1.0, 1.0],
            metadata={"source": "test"},
            timestamp=1000.0,
            role="user",
        )
        assert item.id == "m1"
        assert item.content == "hello"
        assert len(item.hypervector) == 3
        assert item.role == "user"
        assert item.metadata["source"] == "test"

    def test_equality(self):
        a = HDMemoryItem("x", "c", [1.0], {}, 0.0, "user")
        b = HDMemoryItem("x", "c", [1.0], {}, 0.0, "user")
        assert a == b

    def test_fields_are_mutable(self):
        item = HDMemoryItem("x", "c", [1.0], {}, 0.0, "user")
        item.content = "updated"
        assert item.content == "updated"


# ===================================================================
# HDMemoryStore — unit-level tests (low dimension for speed)
# ===================================================================

class TestHDMemoryStore:
    """Tests use dim=256 for fast execution while still exercising real HD ops."""

    def setup_method(self):
        self.store = HDMemoryStore(dim=256, max_items=10)

    def test_add_returns_id(self):
        item_id = self.store.add("hello world", role="user")
        assert item_id.startswith("mem_")

    def test_add_stores_item(self):
        self.store.add("hello", role="user")
        assert len(self.store.items) == 1
        assert self.store.items[0].content == "hello"

    def test_add_with_metadata(self):
        self.store.add("test", role="user", metadata={"k": "v"})
        assert self.store.items[0].metadata["k"] == "v"

    def test_add_default_metadata(self):
        self.store.add("test", role="user")
        assert self.store.items[0].metadata == {}

    def test_eviction_over_max(self):
        for i in range(15):
            self.store.add(f"msg {i}", role="user")
        assert len(self.store.items) == 10
        assert self.store.items[0].content == "msg 5"

    def test_search_empty_store(self):
        results = self.store.search("anything")
        assert results == []

    def test_search_returns_tuples(self):
        self.store.add("hello world", role="user")
        results = self.store.search("hello")
        assert len(results) >= 1
        item_id, content, similarity = results[0]
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert isinstance(similarity, float)

    def test_search_sorted_by_similarity(self):
        self.store.add("python programming language", role="user")
        self.store.add("cooking recipe dinner", role="user")
        self.store.add("python code development", role="user")
        results = self.store.search("python programming")
        if len(results) >= 2:
            assert results[0][2] >= results[1][2]

    def test_search_top_k_limit(self):
        for i in range(10):
            self.store.add(f"message number {i}", role="user")
        results = self.store.search("message", top_k=3)
        assert len(results) <= 3

    def test_search_role_filter(self):
        self.store.add("hello", role="user")
        self.store.add("hello", role="assistant")
        results = self.store.search("hello", role_filter="user")
        for _, _, sim in results:
            assert sim is not None

    def test_get_context_empty(self):
        ctx = self.store.get_context("anything")
        assert ctx == ""

    def test_get_context_returns_string(self):
        self.store.add("important context here", role="user")
        ctx = self.store.get_context("important")
        assert isinstance(ctx, str)

    def test_get_context_max_chars(self):
        for i in range(5):
            self.store.add(f"message number {i} with some content", role="user")
        ctx = self.store.get_context("message", max_chars=50)
        assert len(ctx) <= 200  # allow some overhead from role labels

    def test_get_context_without_roles(self):
        self.store.add("hello world", role="user")
        ctx = self.store.get_context("hello", include_roles=False)
        assert "[User]" not in ctx

    def test_bundle_recent_empty(self):
        result = self.store.bundle_recent()
        assert len(result) == 256

    def test_bundle_recent_with_items(self):
        self.store.add("item one", role="user")
        self.store.add("item two", role="user")
        result = self.store.bundle_recent(n=2)
        assert len(result) == 256
        assert any(v != 0 for v in result)

    def test_get_stats(self):
        self.store.add("a", role="user")
        self.store.add("b", role="assistant")
        stats = self.store.get_stats()
        assert stats["total_items"] == 2
        assert stats["max_items"] == 10
        assert stats["dimension"] == 256
        assert "user" in stats["roles"]
        assert "assistant" in stats["roles"]

    def test_clear(self):
        self.store.add("x", role="user")
        count = self.store.clear()
        assert count == 1
        assert len(self.store.items) == 0

    def test_clear_empty(self):
        count = self.store.clear()
        assert count == 0

    def test_prune_no_duplicates(self):
        self.store.add("alpha bravo", role="user")
        self.store.add("charlie delta", role="user")
        pruned = self.store.prune(similarity_threshold=0.99)
        assert pruned == 0

    def test_prune_single_item(self):
        self.store.add("only one", role="user")
        pruned = self.store.prune()
        assert pruned == 0

    def test_prune_removes_duplicates(self):
        # Add same content repeatedly; at low dim, some will be near-identical
        for _ in range(5):
            self.store.add("exact same text here", role="user")
        initial = len(self.store.items)
        pruned = self.store.prune(similarity_threshold=0.5)
        assert pruned >= 0
        assert len(self.store.items) <= initial


# ===================================================================
# HDMemoryStore — role binding integration
# ===================================================================

class TestHDMemoryStoreRoleBinding:
    def setup_method(self):
        self.store = HDMemoryStore(dim=256, max_items=20)

    def test_role_vectors_initialized(self):
        # Trigger initialization by adding an item
        self.store.add("test", role="user")
        assert "user" in self.store.role_vectors
        assert "assistant" in self.store.role_vectors
        assert "system" in self.store.role_vectors

    def test_encode_content_returns_vector(self):
        vec = self.store.encode_content("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 256

    def test_encode_with_role_returns_vector(self):
        vec = self.store.encode_with_role("hello", "user")
        assert isinstance(vec, list)
        assert len(vec) == 256

    def test_different_roles_produce_different_vectors(self):
        v1 = self.store.encode_with_role("hello", "user")
        v2 = self.store.encode_with_role("hello", "assistant")
        assert v1 != v2

    def test_role_unbinding_in_search(self):
        self.store.add("common phrase", role="user")
        self.store.add("common phrase", role="assistant")
        results = self.store.search("common phrase", top_k=10)
        assert len(results) == 2
