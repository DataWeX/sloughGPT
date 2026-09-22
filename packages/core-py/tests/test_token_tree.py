"""Tests for training.token_tree — TokenTree class."""

from __future__ import annotations

import os
import tempfile

import pytest

from domain.training._internal.token_tree import TokenTree, TrieNode

# ── TrieNode ────────────────────────────────────────────────────────────────


class TestTrieNode:
    def test_default(self):
        node = TrieNode()
        assert node.children == {}
        assert node.token_id is None
        assert node.freq == 0
        assert node.left_id is None
        assert node.right_id is None

    def test_with_children(self):
        node = TrieNode(children={"a": TrieNode(), "b": TrieNode()})
        assert len(node.children) == 2

    def test_with_ids(self):
        node = TrieNode(token_id=5, left_id=1, right_id=2)
        assert node.token_id == 5
        assert node.left_id == 1
        assert node.right_id == 2


# ── TokenTree init ──────────────────────────────────────────────────────────


class TestTokenTreeInit:
    def test_default(self):
        tree = TokenTree()
        assert tree.vocab_size == 0
        assert tree.is_trained is False
        assert tree._pretokenizer == "gpt2"

    def test_whitespace(self):
        tree = TokenTree(pretokenizer="whitespace")
        assert tree._pretokenizer == "whitespace"

    def test_properties(self):
        tree = TokenTree()
        assert tree.vocab_size == 0
        assert tree.pad_id == 0
        assert tree.unk_id == 1
        assert tree.bos_id == 2
        assert tree.eos_id == 3


# ── TokenTree.train ────────────────────────────────────────────────────────


class TestTokenTreeTrain:
    def test_train(self):
        tree = TokenTree()
        tree.train(["hello world"], vocab_size=32)
        assert tree.is_trained is True
        assert tree.vocab_size > 0

    def test_train_empty(self):
        tree = TokenTree()
        with pytest.raises(ValueError, match="at least one text"):
            tree.train([])

    def test_encode_decode(self):
        tree = TokenTree()
        tree.train(["hello world", "hello there"], vocab_size=32)
        ids = tree.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        text = tree.decode(ids)
        assert isinstance(text, str)

    def test_encode_empty(self):
        tree = TokenTree()
        tree.train(["hello"], vocab_size=32)
        ids = tree.encode("")
        assert ids == []

    def test_batch_encode(self):
        tree = TokenTree()
        tree.train(["hello world", "foo bar"], vocab_size=32)
        batch = tree.encode_batch(["hello", "world"])
        assert len(batch) == 2
        for ids in batch:
            assert len(ids) > 0

    def test_embedding(self):
        tree = TokenTree()
        tree.train(["hello world"], vocab_size=32, embed_dim=8)
        ids = tree.encode("hello")
        vec = tree.embedding(ids[0])
        assert vec.shape == (8,)

    def test_decompose(self):
        tree = TokenTree()
        tree.train(["hello world"], vocab_size=32)
        ids = tree.encode("hello")
        pieces = tree.decompose(ids[0])
        assert isinstance(pieces, list)

    def test_train_multiple(self):
        tree = TokenTree()
        tree.train(["hello world", "foo bar baz", "test"], vocab_size=32)
        assert tree.is_trained is True


# ── TokenTree.save/load ────────────────────────────────────────────────────


class TestTokenTreeSaveLoad:
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = TokenTree()
            tree.train(["hello world"], vocab_size=32, embed_dim=8)
            path = os.path.join(tmpdir, "test_token_tree")
            tree.save(path)

            tree2 = TokenTree.load(path)
            assert tree2.vocab_size == tree.vocab_size
            assert tree2.is_trained is True

    def test_encode_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = TokenTree()
            tree.train(["hello world"], vocab_size=32)
            path = os.path.join(tmpdir, "test_token_tree")
            tree.save(path)

            tree2 = TokenTree.load(path)
            ids1 = tree.encode("hello")
            ids2 = tree2.encode("hello")
            assert ids1 == ids2


# ── TokenTree.merges ──────────────────────────────────────────────────────


class TestTokenTreeMerges:
    def test_merges(self):
        tree = TokenTree()
        tree.train(["hello world hello world"], vocab_size=32)
        # Merges may be empty if no pair meets min_frequency
        assert isinstance(tree.merges, list)

    def test_merge_pair(self):
        tree = TokenTree()
        tree.train(["hello world hello world"], vocab_size=32)
        for left, right in tree.merges:
            assert isinstance(left, str)
            assert isinstance(right, str)


# ── Aggregate caching ─────────────────────────────────────────────────────


class TestAggregateCaching:
    _CORPUS = ["the quick brown fox jumps over the lazy dog"] * 4

    @pytest.fixture(autouse=True)
    def _trained_tree(self):
        self.tree = TokenTree()
        self.tree.train(self._CORPUS, vocab_size=64, embed_dim=8)
        return self.tree

    def test_ranked_merges_is_memoized(self):
        first = self.tree._ranked_merges()
        assert first, "trained corpus should produce merges"
        second = self.tree._ranked_merges()
        assert second is first

    def test_top_merges_reuses_ranked_cache(self):
        self.tree.top_merges(top_n=3)
        cached = self.tree._ranked_merges_cache
        assert cached is not None
        assert self.tree.top_merges(top_n=5) == cached[:5]

    def test_search_merges_does_not_refresh_ranking(self):
        before = self.tree._ranked_merges()
        self.tree.search_merges("quick", limit=3)
        assert self.tree._ranked_merges_cache is before

    def test_library_stats_is_memoized(self):
        first = self.tree._library_stats()
        second = self.tree._library_stats()
        assert second is first
        assert self.tree.stats()["library"] is first

    def test_stale_cache_invalidated_on_retrain(self):
        self.tree.top_merges(top_n=5)
        old_ranked = self.tree._ranked_merges_cache
        self.tree.stats()
        old_stats = self.tree._library_stats_cache
        assert old_ranked is not None and old_stats is not None
        self.tree.train(["apple banana apple banana"], vocab_size=32, embed_dim=8)
        assert old_ranked is not self.tree._ranked_merges_cache
        assert old_stats is not self.tree._library_stats_cache
        assert self.tree._ranked_merges() != old_ranked

    def test_untrained_aggregates_match_baseline(self):
        bare = TokenTree()
        assert bare.top_merges(top_n=5) == []
        assert bare.search_merges("a", limit=5) == []
        bare.stats()
        assert bare._library_stats_cache is not None


class TestEmbeddingMatrixCaching:
    _CORPUS = ["the quick brown fox jumps over the lazy dog"] * 4

    @pytest.fixture(autouse=True)
    def _trained_tree(self):
        self.tree = TokenTree()
        self.tree.train(self._CORPUS, vocab_size=64, embed_dim=8)
        return self.tree

    def test_embedding_matrix_is_memoized(self):
        first = self.tree.embedding_matrix()
        assert first is not None
        assert self.tree.embedding_matrix() is first

    def test_matrix_stats_reuses_cached_matrix(self):
        self.tree.embedding_matrix()
        assert self._matrix_cached() is True
        self.tree.embedding_matrix_stats(top_n=4)
        assert self._matrix_cached() is True

    def test_similar_reuses_cached_matrix(self):
        self.tree.similar(self.tree.stoi[" the</w>"], top_k=3)
        assert self._matrix_cached() is True

    def test_cap_disables_cache_for_large_matrices(self, monkeypatch):
        import domain.training._internal.token_tree as tt

        monkeypatch.setattr(tt, "_EMBEDDING_MATRIX_CACHE_MAX_ELEMENTS", 4)
        mat = self.tree.embedding_matrix()
        assert mat is not None
        assert self._matrix_cached() is False

    def test_retrain_invalidates_cached_matrix(self):
        first = self.tree.embedding_matrix()
        assert self._matrix_cached() is True
        self.tree.train(["apple banana apple banana"], vocab_size=32, embed_dim=8)
        assert self._matrix_cached() is False
        second = self.tree.embedding_matrix()
        assert second is not first

    def test_disabled_embeddings_never_cache(self):
        bare = TokenTree()
        assert bare.embedding_matrix() is None
        assert bare._embedding_matrix_cached is False

    def _matrix_cached(self) -> bool:
        return self.tree._embedding_matrix_cached
