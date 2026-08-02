"""Tests for domains/soul/hd_memory.py."""

import sys

import pytest

from domains.soul.hd_memory import HDMemoryItem, HDMemoryStore


class FakeHD:
    def __init__(self, dim=10000):
        self.dim = dim
        self._counter = 0

    def _vec(self, seed):
        import hashlib
        h = hashlib.md5(str(seed).encode()).digest()
        vec = [((h[i % 16]) / 255.0) * 2 - 1 for i in range(self.dim)]
        return vec

    def encode(self, text):
        return self._vec(f"enc:{text}")

    def encode_text(self, text):
        return self._vec(f"text:{text}")

    def bind(self, a, b):
        return [(x * y) for x, y in zip(a, b)]

    def unbind(self, a, b):
        return [(x * y) for x, y in zip(a, b)]

    def bundle(self, vectors):
        if not vectors:
            return [0.0] * self.dim
        return [
            1.0 if sum(v[i] for v in vectors) > 0 else 0.0
            for i in range(self.dim)
        ]

    def similarity(self, a, b):
        import numpy as np
        va = np.array(a)
        vb = np.array(b)
        num = float(va @ vb)
        den = (float(np.linalg.norm(va)) * float(np.linalg.norm(vb))) or 1.0
        return num / den


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "domains.soul.quantum",
        type(
            "FakeQuantum",
            (),
            {"HyperdimensionalProcessor": lambda dim=10000: FakeHD(dim)},
        ),
    )
    s = HDMemoryStore(dim=64)
    return s


class TestHDMemoryItem:
    def test_fields(self):
        item = HDMemoryItem(
            id="1", content="hi", hypervector=[1.0], metadata={}, timestamp=1.0, role="user"
        )
        assert item.id == "1"
        assert item.content == "hi"
        assert item.role == "user"


class TestEncode:
    def test_encode_content_returns_vector(self, store):
        vec = store.encode_content("hello world")
        assert len(vec) == store.dim

    def test_encode_with_role(self, store):
        vec = store.encode_with_role("hello", "user")
        assert len(vec) == store.dim

    def test_unknown_role_falls_back(self, store):
        vec = store.encode_with_role("hello", "unknown_role")
        assert len(vec) == store.dim


class TestAdd:
    def test_add_returns_id(self, store):
        item_id = store.add("some content")
        assert item_id.startswith("mem_")
        assert item_id in [i.id for i in store.items]

    def test_add_stores_metadata(self, store):
        item_id = store.add("content", role="assistant", metadata={"k": "v"})
        item = next(i for i in store.items if i.id == item_id)
        assert item.metadata == {"k": "v"}
        assert item.role == "assistant"

    def test_add_evicts_over_capacity(self, store):
        store.max_items = 2
        store.add("a")
        store.add("b")
        store.add("c")
        assert len(store.items) == 2
        assert store.items[-1].content == "c"

    def test_initializes_role_vectors(self, store):
        store._get_hyperdim()
        assert set(store.role_vectors.keys()) == {"user", "assistant", "system"}


class TestSearch:
    def test_empty_store(self, store):
        assert store.search("query") == []

    def test_returns_results(self, store):
        store.add("one")
        store.add("two")
        results = store.search("one", top_k=5)
        assert len(results) == 2
        ids, contents, sims = zip(*results)
        assert all(isinstance(s, float) for s in sims)

    def test_role_filter(self, store):
        store.add("user msg", role="user")
        store.add("assistant msg", role="assistant")
        results = store.search("msg", role_filter="user")
        assert all(r[2] for r in results)
        assert all(store.items[0].role == "user" for _, _, _ in results)
        roles = [next(i.role for i in store.items if i.id == rid) for rid, _, _ in results]
        assert set(roles) == {"user"}


class TestGetContext:
    def test_empty_context(self, store):
        assert store.get_context("nothing") == ""

    def test_returns_context_with_roles(self, store):
        store.add("alpha content", role="user")
        store.add("beta content", role="assistant")
        ctx = store.get_context("alpha", include_roles=True)
        assert "[User]" in ctx or "[Assistant]" in ctx
        assert "content" in ctx

    def test_max_chars_respected(self, store):
        store.add("x" * 100, role="user")
        store.add("y" * 100, role="user")
        ctx = store.get_context("xy", max_chars=50)
        assert len(ctx) <= 60


class TestGetContextBranches:
    def test_get_context_breaks_at_max_chars(self, store):
        store.add("aaaa", role="user")
        store.search = lambda query, top_k=10, role_filter=None: [
            ("m1", "aaaa", 0.5),
            ("m2", "bbbb", 0.5),
        ]
        ctx = store.get_context("q", max_chars=5)
        assert ctx == "[User]: aaaa"

    def test_get_context_without_roles(self, store):
        store.add("alpha content", role="user")
        store.search = lambda query, top_k=10, role_filter=None: [
            ("m1", "alpha content", 0.9),
        ]
        ctx = store.get_context("q", include_roles=False)
        assert ctx == "alpha content"


class TestHyperdimFailure:
    def test_init_failure_raises(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "domains.soul.quantum", None)
        s = HDMemoryStore(dim=64)
        with pytest.raises(Exception):
            s._get_hyperdim()


class TestBundleRecent:
    def test_empty_bundle(self, store):
        vec = store.bundle_recent(5)
        assert len(vec) == store.dim
        assert all(v == 0 for v in vec)

    def test_bundle_nonempty(self, store):
        store.add("a")
        store.add("b")
        vec = store.bundle_recent(10)
        assert len(vec) == store.dim


class TestStatsAndClear:
    def test_stats(self, store):
        store.add("a", role="user")
        store.add("b", role="assistant")
        stats = store.get_stats()
        assert stats["total_items"] == 2
        assert stats["max_items"] == 1000
        assert stats["dimension"] == 64
        assert stats["roles"] == {"user": 1, "assistant": 1}

    def test_clear(self, store):
        store.add("a")
        assert store.clear() == 1
        assert store.items == []


class TestPrune:
    def test_prune_under_two_items(self, store):
        store.add("a")
        assert store.prune() == 0

    def test_prune_removes_duplicates(self, store):
        store.add("same text")
        store.add("same text")
        removed = store.prune(similarity_threshold=0.0)
        assert removed == 1
        assert len(store.items) == 1

    def test_prune_skips_already_removed(self, store):
        store.add("a")
        store.add("b")
        store.add("c")
        store.add("d")
        items = store.items
        markers = {0: [1.0], 1: [2.0], 2: [3.0], 3: [4.0]}
        for idx, it in enumerate(items):
            it.hypervector = markers[idx] * store.dim
        items[0].timestamp = 1000.0
        items[1].timestamp = 100.0
        items[2].timestamp = 500.0
        items[3].timestamp = 50.0

        def sim(a, b):
            return 0.99 if b[0] in (2.0, 4.0) else 0.1

        store._hyperdim.similarity = sim
        removed = store.prune(similarity_threshold=0.5)
        assert removed == 2
        assert [i.content for i in store.items] == ["a", "c"]
