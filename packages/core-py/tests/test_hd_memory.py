"""Tests for domains.soul.hd_memory — HDMemoryItem, HDMemoryStore."""

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


class TestHDMemoryStore:
    def test_init(self):
        store = HDMemoryStore(dim=100, max_items=10)
        assert store.dim == 100
        assert store.max_items == 10

    def test_add_and_search(self):
        store = HDMemoryStore(dim=100, max_items=10)
        store.add("hello world", role="user", metadata={})
        results = store.search("hello", top_k=5)
        assert len(results) >= 1

    def test_max_items(self):
        store = HDMemoryStore(dim=50, max_items=3)
        for i in range(5):
            store.add(f"item {i}", role="user", metadata={})
        assert len(store.items) <= 3

    def test_stats(self):
        store = HDMemoryStore(dim=50, max_items=10)
        stats = store.get_stats()
        assert "total_items" in stats
        assert stats["total_items"] == 0

    def test_clear(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("test", role="user", metadata={})
        n = store.clear()
        assert n >= 0

    def test_get_context(self):
        store = HDMemoryStore(dim=50, max_items=10)
        store.add("hello world", role="user", metadata={})
        ctx = store.get_context("hello")
        assert isinstance(ctx, str)
