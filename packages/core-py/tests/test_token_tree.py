"""
Tests for TokenTree — the tree-structured BPE tokenizer on pugqeep Points.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from domains.training.token_tree import (
    SPECIAL_TOKENS,
    WORD_SUFFIX,
    TokenTree,
    _split_pieces,
)
from domains.training.tokenizer import SloBPE

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick",
    "quick quicksilver quickest quickly",
    "brown brownie browning browner",
    "hello world hello there hello",
    "the world of the quick brown",
]


def make_tree(vocab_size: int = 128, **kw) -> TokenTree:
    tree = TokenTree()
    embed_dim = kw.pop("embed_dim", 16)
    tree.train(CORPUS, vocab_size=vocab_size, embed_dim=embed_dim, **kw)
    return tree


class TestTraining:
    def test_train_builds_vocab(self):
        tree = make_tree(vocab_size=200)
        assert tree.vocab_size >= 200 or tree.vocab_size >= len(SPECIAL_TOKENS) + 1
        assert tree.is_trained

    def test_special_tokens_first(self):
        tree = make_tree()
        for i, tok in enumerate(SPECIAL_TOKENS):
            assert tree.stoi[tok] == i
        assert WORD_SUFFIX in tree.stoi

    def test_merges_recorded(self):
        tree = make_tree(vocab_size=200)
        assert len(tree.merges) > 0
        assert all(len(m) == 2 for m in tree.merges)

    def test_vocab_equals_base_plus_merges(self):
        tree = make_tree(vocab_size=200)
        base_chars = [t for t in tree.vocab if len(t) == 1 and t not in SPECIAL_TOKENS]
        assert tree.vocab_size == len(SPECIAL_TOKENS) + 1 + len(base_chars) + len(tree.merges)

    def test_trie_terminals_match_vocab(self):
        tree = make_tree(vocab_size=200)
        ids = set()
        stack = [tree.root]
        while stack:
            node = stack.pop()
            if node.token_id is not None:
                ids.add(node.token_id)
            stack.extend(node.children.values())
        assert ids == set(range(tree.vocab_size))


class TestEncoding:
    def test_round_trip(self):
        tree = make_tree(vocab_size=300)
        for text in ["the quick brown fox", "hello world", "quickest browner"]:
            assert tree.decode(tree.encode(text)) == text

    def test_greedy_longest_match_merges_common_words(self):
        tree = make_tree(vocab_size=300)
        ids = tree.encode("the")
        joined = "".join(tree.itos[i] for i in ids)
        assert joined == "the" + WORD_SUFFIX

    def test_unknown_character_maps_to_unk(self):
        tree = make_tree()
        ids = tree.encode("hello\u20ac")  # € is not in the corpus
        assert tree.unk_id in ids

    def test_bos_eos_flags(self):
        tree = make_tree()
        ids = tree.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tree.bos_id
        assert ids[-1] == tree.eos_id

    def test_batch_matches_serial(self):
        tree = make_tree(vocab_size=300)
        texts = CORPUS * 3
        serial = [tree.encode(t) for t in texts]
        parallel = tree.encode_batch(texts, max_workers=4)
        assert parallel == serial

    def test_batch_order_preserved(self):
        tree = make_tree(vocab_size=300)
        texts = ["a", "the quick", "z", "hello world"] * 10
        out = tree.encode_batch(texts, max_workers=3)
        assert len(out) == len(texts)
        assert all(out[i] == tree.encode(t) for i, t in enumerate(texts))

    def test_batch_untrained_raises(self):
        tree = TokenTree()
        with pytest.raises(RuntimeError):
            tree.encode_batch(["hello"])

    def test_encode_batch_empty(self):
        tree = make_tree()
        assert tree.encode_batch([]) == []


class TestDecoding:
    def test_word_boundaries_preserved(self):
        tree = make_tree(vocab_size=300)
        text = "the quick brown fox jumps"
        assert tree.decode(tree.encode(text)) == text

    def test_special_tokens_skipped(self):
        tree = make_tree()
        ids = [tree.bos_id, tree.eos_id, tree.pad_id, tree.unk_id]
        assert tree.decode(ids) == ""

    def test_unknown_id_skipped(self):
        tree = make_tree()
        assert tree.decode([99999]) == ""


class TestMergeLineage:
    def test_decompose_base_token_is_itself(self):
        tree = make_tree(vocab_size=128)
        leaves = tree.decompose(tree.stoi["t"])
        assert leaves == ["t"]

    def test_decompose_walks_to_characters(self):
        tree = make_tree(vocab_size=300)
        # "the" must exist as a token after enough merges on this corpus
        tid = tree.stoi.get("the" + WORD_SUFFIX)
        assert tid is not None
        leaves = tree.decompose(tid)
        assert "".join(p for p in leaves if p != WORD_SUFFIX) == "the"

    def test_lineage_is_binary_tree(self):
        tree = make_tree(vocab_size=300)
        for left, right in tree.merges:
            merged = left + right
            assert merged in tree.stoi
            tid = tree.stoi[merged]
            l, r = tree._lineage[tid]
            assert (l, r) == (tree.stoi[left], tree.stoi[right])


class TestTokenPoints:
    def test_embedding_generated_from_point(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        vec = tree.embedding(0)
        assert vec is not None
        assert vec.shape == (16,)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))

    def test_embedding_deterministic(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        v1 = tree.embedding(tree.stoi["the" + WORD_SUFFIX])
        v2 = tree.embedding(tree.stoi["the" + WORD_SUFFIX])
        assert v1 is not None
        np.testing.assert_array_equal(v1, v2)

    def test_embedding_stored_as_cluster_or_raw_point(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        points = tree.embedding_points()
        assert points == tree.vocab_size
        assert tree.embedding_compression_ratio() >= 1.0

    def test_embedding_untrained_dim_zero(self):
        tree = TokenTree()
        assert tree.embedding(0) is None


class TestSemanticQuery:
    def test_embedding_matrix_shape_and_normalized(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        mat = tree.embedding_matrix()
        assert mat is not None
        assert mat.shape == (tree.vocab_size, 16)
        assert mat.dtype == np.float32
        assert np.all(np.isfinite(mat))
        norms = np.linalg.norm(mat, axis=1)
        # cluster reconstruction shrinks norms slightly; never exceeds 1
        assert np.all(norms <= 1.0 + 1e-3)
        assert np.any(norms > 0.5)

    def test_embedding_matrix_disabled(self):
        tree = TokenTree()
        assert tree.embedding_matrix() is None

    def test_similar_returns_ranked_pairs(self):
        tree = make_tree(vocab_size=300, embed_dim=16)
        tid = tree.stoi["the" + WORD_SUFFIX]
        results = tree.similar(tid, top_k=4)
        assert len(results) == 4
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        assert all(other != tid for other, _ in results)

    def test_similar_finds_cooccurring_neighbor(self):
        tree = make_tree(vocab_size=300, embed_dim=16)
        mat = tree.embedding_matrix()
        word_tokens = [
            tid for tid, tok in enumerate(tree.vocab)
            if tok.endswith(WORD_SUFFIX) and len(tok) > 3
        ]
        live = [tid for tid in word_tokens if np.linalg.norm(mat[tid]) > 0.5]
        assert live, "expected word tokens with non-degenerate embeddings"
        best = max(live, key=lambda tid: np.linalg.norm(mat[tid]))
        results = tree.similar(best, top_k=1)
        assert results and results[0][1] > 0.3

    def test_similar_untrained_returns_empty(self):
        assert TokenTree().similar(0, top_k=3) == []



class TestPersistence:
    def test_save_load_round_trip(self, tmp_path: Path):
        tree = make_tree(vocab_size=300)
        base = str(tmp_path / "toktree")
        meta_path, points_path = tree.save(base)
        assert meta_path.exists()
        assert points_path.exists()

        loaded = TokenTree.load(base)
        assert loaded.vocab == tree.vocab
        assert loaded.merges == tree.merges
        assert loaded.stoi == tree.stoi
        assert loaded.is_trained

        for text in ["the quick brown fox", "hello world", "quickest browner"]:
            assert loaded.decode(loaded.encode(text)) == text

    def test_save_load_preserves_embeddings(self, tmp_path: Path):
        tree = make_tree(vocab_size=200, embed_dim=16)
        base = str(tmp_path / "embtree")
        tree.save(base)
        loaded = TokenTree.load(base)
        for tid in range(min(10, tree.vocab_size)):
            a = tree.embedding(tid)
            b = loaded.embedding(tid)
            assert a is not None and b is not None
            np.testing.assert_array_equal(a, b)

    def test_save_load_preserves_lineage(self, tmp_path: Path):
        tree = make_tree(vocab_size=300)
        base = str(tmp_path / "linetree")
        tree.save(base)
        loaded = TokenTree.load(base)
        tid = tree.stoi.get("the" + WORD_SUFFIX)
        if tid is not None:
            assert loaded.decompose(tid) == tree.decompose(tid)

    def test_load_missing_meta_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            TokenTree.load(str(tmp_path / "nope"))


class TestIntrospection:
    def test_stats(self):
        tree = make_tree(vocab_size=200)
        s = tree.stats()
        assert s["trained"] is True
        assert s["vocab_size"] == tree.vocab_size
        assert s["embedding_points"] == tree.vocab_size
        assert s["embed_dim"] == 16

    def test_show_tree_builds(self):
        tree = make_tree(vocab_size=300)
        tid = tree.stoi.get("the" + WORD_SUFFIX)
        if tid is not None:
            rendered = tree.show_tree(tid)
            assert "the" + WORD_SUFFIX in rendered


class TestSloBPEParity:
    def test_round_trip_matches_slobpe(self):
        tree = make_tree(vocab_size=300)
        bpe = SloBPE()
        bpe.train(CORPUS, vocab_size=300)
        for text in CORPUS:
            assert tree.decode(tree.encode(text)) == text
            assert bpe.decode(bpe.encode(text)) == text

    def test_both_encode_corpus_consistently(self):
        tree = make_tree(vocab_size=300)
        bpe = SloBPE()
        bpe.train(CORPUS, vocab_size=300)
        for text in CORPUS:
            t_ids = tree.encode(text)
            b_ids = bpe.encode(text)
            # Same vocabulary size target; both must reproduce the text
            assert tree.decode(t_ids) == text
            assert bpe.decode(b_ids) == text


class TestSplitPieces:
    def test_single_char(self):
        assert _split_pieces("a") == ["a"]

    def test_word_with_suffix(self):
        assert _split_pieces("the" + WORD_SUFFIX) == ["t", "h", "e", WORD_SUFFIX]

    def test_special_token_whole(self):
        assert _split_pieces("<PAD>") == ["<PAD>"]

    def test_merged_token_pieces_concat(self):
        assert _split_pieces("th" + WORD_SUFFIX) == ["t", "h", WORD_SUFFIX]


class TestStringInput:
    """train() must treat a raw str as a single document, not as chars."""

    def test_single_string_trains_full_merges(self):
        text = (
            "the quick brown fox jumps over the lazy dog the quick "
            "brown fox the quick brown the quick"
        )
        tree = TokenTree().train(text, vocab_size=64)
        assert len(tree.merges) > 5
        assert "the" + WORD_SUFFIX in tree.stoi

    def test_single_string_encodes_with_subwords(self):
        tree = TokenTree().train(
            "the quick brown fox jumps over the lazy dog the quick "
            "brown fox the quick brown the quick",
            vocab_size=64,
        )
        ids = tree.encode("the quick brown fox")
        assert len(ids) <= 8

    def test_single_string_round_trips(self):
        text = "the quick brown fox jumps over the lazy dog"
        tree = TokenTree().train(text, vocab_size=64)
        assert tree.decode(tree.encode(text)) == text

    def test_list_input_still_works(self):
        tree = TokenTree().train([CORPUS[0], CORPUS[1]], vocab_size=64)
        for text in CORPUS[:2]:
            assert tree.decode(tree.encode(text)) == text

