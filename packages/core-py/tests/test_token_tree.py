"""Tests for training.token_tree — TokenTree class."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
from domains.training.token_tree import TokenTree, TrieNode, SPECIAL_TOKENS


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
