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
