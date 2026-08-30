"""Tests for domains.cognitive.reasoning.deep — WorkingMemory."""

from domains.cognitive.reasoning.deep import WorkingMemory


class TestWorkingMemory:
    def test_init(self):
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert wm.items == []

    def test_add(self):
        wm = WorkingMemory()
        wm.add("item1")
        assert "item1" in wm.items

    def test_add_multiple(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(f"item{i}")
        assert len(wm.items) == 5

    def test_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        assert len(wm.items) == 3
        assert "a" not in wm.items

    def test_lru_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.add("c")
        assert "b" not in wm.items
        assert "a" in wm.items

    def test_get_recent(self):
        wm = WorkingMemory()
        wm.add("x")
        wm.add("y")
        recent = wm.get_recent(1)
        assert len(recent) == 1

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.clear()
        assert wm.items == []

    def test_access_increments(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.access("a")
        wm.access("a")
        assert wm.access_count["a"] == 3

    def test_capacity_one_evicts_every_add(self):
        wm = WorkingMemory(capacity=1)
        wm.add("a")
        assert wm.items == ["a"]
        wm.add("b")
        assert wm.items == ["b"]
        wm.add("c")
        assert wm.items == ["c"]

    def test_capacity_one(self):
        wm = WorkingMemory(capacity=1)
        wm.add("a")
        assert wm.items == ["a"]
        wm.add("b")
        assert wm.items == ["b"]

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=5)
        for i in range(5):
            wm.add(f"i{i}")
        assert len(wm.items) == 5
        wm.add("overflow")
        assert len(wm.items) == 5
        assert "overflow" in wm.items

    def test_add_duplicate(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("a")
        assert wm.items.count("a") == 2
        assert wm.access_count["a"] == 1

    def test_eviction_removes_lru_not_recent(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.access("a")
        wm.add("d")
        assert "a" in wm.items
        assert "b" not in wm.items

    def test_get_recent_sorted_by_access_count(self):
        wm = WorkingMemory(capacity=5)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("c")
        wm.access("c")
        wm.access("c")
        wm.access("b")
        recent = wm.get_recent(3)
        assert recent[0] == "c"
        assert recent[1] == "b"

    def test_get_recent_n_greater_than_items(self):
        wm = WorkingMemory()
        wm.add("x")
        recent = wm.get_recent(10)
        assert recent == ["x"]

    def test_get_recent_empty(self):
        wm = WorkingMemory()
        assert wm.get_recent(5) == []

    def test_clear_resets_access_count(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.access("a")
        wm.clear()
        assert wm.access_count == {}
        assert wm.items == []

    def test_access_nonexistent_item(self):
        wm = WorkingMemory()
        wm.access("ghost")
        assert wm.access_count["ghost"] == 1

    def test_access_after_add_increments(self):
        wm = WorkingMemory()
        wm.add("x")
        wm.access("x")
        assert wm.access_count["x"] == 2

    def test_eviction_chain(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.add("c")
        assert "b" not in wm.items
        wm.access("c")
        wm.add("d")
        assert "a" not in wm.items

    def test_lru_eviction_with_equal_access_counts(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert len(wm.items) == 2

    def test_items_list_preserves_order(self):
        wm = WorkingMemory(capacity=5)
        wm.add("z")
        wm.add("y")
        wm.add("x")
        assert wm.items[-1] == "x"
        assert wm.items[0] == "z"

    def test_access_does_not_change_item_order(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        assert wm.items == ["a", "b"]

    def test_large_capacity(self):
        wm = WorkingMemory(capacity=1000)
        for i in range(1000):
            wm.add(f"i{i}")
        assert len(wm.items) == 1000

    def test_eviction_after_heavy_access(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        for _ in range(100):
            wm.access("a")
        wm.add("d")
        assert "a" in wm.items
        assert "b" not in wm.items

    def test_get_recent_n_zero(self):
        wm = WorkingMemory()
        wm.add("x")
        assert wm.get_recent(0) == []

    def test_multiple_clears(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.clear()
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_add_after_clear(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.clear()
        wm.add("b")
        assert wm.items == ["b"]

    def test_access_count_integrity(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.access("a")
        wm.access("a")
        wm.access("b")
        wm.add("d")
        assert wm.access_count.get("a") == 3
        assert wm.access_count.get("b") == 2
        assert wm.access_count.get("d") == 1

    def test_init_custom_capacity(self):
        wm = WorkingMemory(capacity=10)
        assert wm.capacity == 10
        assert wm.items == []
        assert wm.access_count == {}

    def test_add_and_get_recent_combined(self):
        wm = WorkingMemory(capacity=5)
        wm.add("alpha")
        wm.add("beta")
        wm.add("gamma")
        wm.access("gamma")
        wm.access("alpha")
        recent = wm.get_recent(2)
        assert "gamma" in recent
        assert "alpha" in recent
