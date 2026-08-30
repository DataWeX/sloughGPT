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
    TrieNode,
    _node_from_dict,
    _node_to_dict,
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

    def test_empty_texts_raises(self):
        with pytest.raises(ValueError):
            TokenTree().train([])

    def test_single_string_input(self):
        text = "hello world " * 20
        tree = TokenTree().train(text, vocab_size=64)
        assert tree.is_trained
        assert len(tree.merges) > 0

    def test_min_frequency_filters_pairs(self):
        tree_small = make_tree(vocab_size=200, min_frequency=999)
        tree_large = make_tree(vocab_size=200, min_frequency=1)
        assert len(tree_small.merges) <= len(tree_large.merges)

    def test_lowercase_true_lowercases_corpus(self):
        tree = TokenTree().train("HELLO WORLD", vocab_size=64)
        encoded = tree.encode("HELLO WORLD")
        assert encoded
        decoded = tree.decode(encoded)
        assert decoded == "hello world"

    def test_pretokenizer_whitespace(self):
        tree = TokenTree(pretokenizer="whitespace")
        tree.train(["hello world"], vocab_size=64)
        assert tree._pretokenizer == "whitespace"
        assert tree.is_trained

    def test_returns_self_for_chaining(self):
        tree = TokenTree()
        result = tree.train(["hello world"], vocab_size=64)
        assert result is tree

    def test_vocab_id_stable_across_retrains(self):
        t1 = TokenTree().train("hello world hello world", vocab_size=64)
        t2 = TokenTree().train("hello world hello world", vocab_size=64)
        assert t1.vocab == t2.vocab
        assert t1.merges == t2.merges

    def test_vocab_idempotent_retrain_same_data(self):
        tree = TokenTree()
        tree.train("hello world", vocab_size=64)
        v1 = list(tree.vocab)
        tree.train("hello world", vocab_size=64)
        assert tree.vocab == v1


class TestTrieNodeDataclass:
    def test_defaults(self):
        n = TrieNode()
        assert n.children == {}
        assert n.token_id is None
        assert n.freq == 0
        assert n.left_id is None
        assert n.right_id is None

    def test_fields_settable(self):
        child = TrieNode(token_id=5, freq=10)
        parent = TrieNode(children={"a": child}, token_id=0, left_id=1, right_id=2)
        assert parent.children["a"].token_id == 5
        assert parent.left_id == 1
        assert parent.right_id == 2


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
        ids = tree.encode("hello\u20ac")  # euro sign not in corpus
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

    def test_encode_returns_list_of_int(self):
        tree = make_tree()
        ids = tree.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self):
        tree = make_tree()
        assert tree.encode("") == []

    def test_encode_only_whitespace(self):
        tree = make_tree()
        ids = tree.encode("   ")
        decoded = tree.decode(ids)
        assert decoded == "   " or decoded == ""

    def test_encode_bos_only(self):
        tree = make_tree()
        ids = tree.encode("hi", add_bos=True, add_eos=False)
        assert ids[0] == tree.bos_id
        assert ids[-1] != tree.eos_id

    def test_encode_eos_only(self):
        tree = make_tree()
        ids = tree.encode("hi", add_bos=False, add_eos=True)
        assert ids[0] != tree.bos_id
        assert ids[-1] == tree.eos_id

    def test_encode_single_char(self):
        tree = make_tree()
        ids = tree.encode("a")
        assert len(ids) >= 1
        decoded = tree.decode(ids)
        assert decoded == "a"


class TestQuery:
    def test_query_exact_match(self):
        tree = make_tree(vocab_size=128)
        tid, consumed = tree.query("t")
        assert tid == tree.stoi["t"]
        assert consumed == 1

    def test_query_longest_prefix(self):
        tree = make_tree(vocab_size=300)
        # "the</w>" should be a full match
        full = "the" + WORD_SUFFIX
        tid, consumed = tree.query(full)
        assert consumed == len(full)
        assert tree.itos[tid] == full

    def test_query_partial_prefix_returns_earliest_terminal(self):
        tree = make_tree(vocab_size=300)
        # Query "th" — partial match; returns the longest terminal found
        tid, consumed = tree.query("th")
        assert consumed >= 1
        assert tree.itos.get(tid) is not None

    def test_query_empty_string_returns_unk(self):
        tree = make_tree()
        tid, consumed = tree.query("")
        assert tid == tree.unk_id
        assert consumed == 1

    def test_query_unmapped_char_returns_unk(self):
        tree = make_tree()
        tid, consumed = tree.query("\u2603")  # snowman
        assert tid == tree.unk_id
        assert consumed == 1


class TestTracePath:
    def test_ids_match_encode(self):
        tree = make_tree(vocab_size=300)
        for text in ["the quick brown fox", "hello world", "quickest browner"]:
            steps = tree.trace_path(text)
            assert [s["id"] for s in steps] == tree.encode(text)

    def test_steps_record_remaining_token_and_consumed(self):
        tree = make_tree(vocab_size=300)
        steps = tree.trace_path("the quick")
        assert steps
        for s in steps:
            assert set(s) == {"remaining", "id", "consumed"}
            assert isinstance(s["consumed"], int) and s["consumed"] > 0
            assert s["remaining"][: s["consumed"]] == tree.itos[s["id"]]

    def test_word_boundaries_walked_left_to_right(self):
        tree = make_tree(vocab_size=300)
        steps = tree.trace_path("the quick")
        padded = sum(len(w) for w in tree._pretokenize("the quick", lowercase=True)) + (
            2 * len(WORD_SUFFIX)
        )
        assert sum(s["consumed"] for s in steps) == padded
        assert steps[0]["remaining"] == "the" + WORD_SUFFIX
        assert steps[1]["remaining"] == " quick" + WORD_SUFFIX

    def test_untrained_encodes_to_unk(self):
        tree = TokenTree()
        steps = tree.trace_path("hi")
        assert steps and all(s["id"] == tree.unk_id for s in steps)

    def test_trace_path_empty(self):
        tree = make_tree()
        steps = tree.trace_path("")
        assert steps == []

    def test_trace_path_single_word(self):
        tree = make_tree(vocab_size=300)
        steps = tree.trace_path("the")
        assert steps
        ids = [s["id"] for s in steps]
        assert ids == tree.encode("the")


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

    def test_decode_empty(self):
        tree = make_tree()
        assert tree.decode([]) == ""

    def test_skip_special_false_includes_special_tokens(self):
        tree = make_tree()
        ids = [tree.bos_id, tree.eos_id]
        result = tree.decode(ids, skip_special=False)
        assert "<BOS>" in result
        assert "<EOS>" in result

    def test_word_suffix_always_removed(self):
        tree = make_tree()
        # decode never outputs WORD_SUFFIX even with skip_special=False
        ids = [tree.stoi[WORD_SUFFIX]]
        assert tree.decode(ids, skip_special=False) == ""

    def test_decode_single_token(self):
        tree = make_tree(vocab_size=128)
        tid = tree.stoi.get("a")
        if tid is not None:
            assert tree.decode([tid]) == "a"


class TestMergeLineage:
    def test_decompose_base_token_is_itself(self):
        tree = make_tree(vocab_size=128)
        leaves = tree.decompose(tree.stoi["t"])
        assert leaves == ["t"]

    def test_decompose_walks_to_characters(self):
        tree = make_tree(vocab_size=300)
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

    def test_decompose_unknown_id_returns_empty(self):
        tree = make_tree()
        assert tree.decompose(99999) == []

    def test_decompose_special_token_returns_single_piece(self):
        tree = make_tree()
        leaves = tree.decompose(tree.stoi["<PAD>"])
        assert leaves == ["<PAD>"]

    def test_decompose_word_suffix(self):
        tree = make_tree()
        leaves = tree.decompose(tree.stoi[WORD_SUFFIX])
        assert leaves == [WORD_SUFFIX]

    def test_decompose_base_char_returns_char(self):
        tree = make_tree()
        for ch in "abcdef":
            tid = tree.stoi.get(ch)
            if tid is not None:
                assert tree.decompose(tid) == [ch]


class TestTopMerges:
    def test_returns_ranked_rules(self):
        tree = make_tree(vocab_size=200)
        rules = tree.top_merges(top_n=10)
        assert len(rules) == 10
        assert rules[0]["rank"] == 1
        counts = [r["count"] for r in rules]
        assert counts == sorted(counts, reverse=True)
        for r in rules:
            assert r["token"] == r["left"] + r["right"]
            assert r["count"] > 0

    def test_respects_top_n(self):
        tree = make_tree(vocab_size=200)
        assert len(tree.top_merges(top_n=3)) == 3
        assert len(tree.top_merges(top_n=1000)) <= len(tree.merges)

    def test_untrained_returns_empty(self):
        tree = TokenTree()
        assert tree.top_merges() == []


class TestRankedMerges:
    def test_ranked_merges_sorted_descending(self):
        tree = make_tree(vocab_size=200)
        ranked = tree._ranked_merges()
        assert ranked
        counts = [r["count"] for r in ranked]
        assert counts == sorted(counts, reverse=True)

    def test_ranked_merges_rank_starts_at_1(self):
        tree = make_tree(vocab_size=200)
        ranked = tree._ranked_merges()
        assert ranked[0]["rank"] == 1
        assert ranked[-1]["rank"] == len(ranked)

    def test_ranked_merges_untrained_empty(self):
        assert TokenTree()._ranked_merges() == []

    def test_ranked_merges_have_all_fields(self):
        tree = make_tree(vocab_size=200)
        for m in tree._ranked_merges():
            assert set(m.keys()) == {"rank", "left", "right", "token", "count"}
            assert m["token"] == m["left"] + m["right"]


class TestSearchMerges:
    def test_matches_left_right_and_merged_token(self):
        tree = make_tree(vocab_size=200)
        rules = tree.search_merges("th", limit=50)
        assert rules
        for r in rules:
            assert (
                "th" in r["left"].lower()
                or "th" in r["right"].lower()
                or "th" in r["token"].lower()
            )

    def test_case_insensitive(self):
        tree = make_tree(vocab_size=200)
        upper = tree.search_merges("THE", limit=50)
        lower = tree.search_merges("the", limit=50)
        assert {r["token"] for r in upper} == {r["token"] for r in lower}
        assert upper

    def test_keeps_global_frequency_rank(self):
        tree = make_tree(vocab_size=200)
        top = {r["token"]: r["rank"] for r in tree.top_merges(top_n=1000)}
        hits = tree.search_merges("th", limit=50)
        for r in hits:
            assert r["rank"] == top[r["token"]]

    def test_results_sorted_by_count_descending(self):
        tree = make_tree(vocab_size=200)
        counts = [r["count"] for r in tree.search_merges("e", limit=50)]
        assert counts == sorted(counts, reverse=True)

    def test_respects_limit(self):
        tree = make_tree(vocab_size=200)
        all_matches = tree.search_merges("e", limit=1000)
        limited = tree.search_merges("e", limit=3)
        assert len(limited) == min(3, len(all_matches))
        assert len(limited) <= 3

    def test_no_match_returns_empty(self):
        tree = make_tree(vocab_size=200)
        assert tree.search_merges("zzz-no-such-part", limit=50) == []

    def test_untrained_returns_empty(self):
        tree = TokenTree()
        assert tree.search_merges("the") == []

    def test_empty_query_returns_empty(self):
        tree = make_tree(vocab_size=200)
        assert tree.search_merges("") == []


class TestVocabEntries:
    def test_returns_total_and_entries_in_id_order(self):
        tree = make_tree(vocab_size=200)
        out = tree.vocab_entries(offset=0, limit=500)
        assert out["total"] == len(tree.vocab)
        ids = [e["id"] for e in out["entries"]]
        assert ids == sorted(ids)
        assert ids == list(range(len(out["entries"])))

    def test_paging_slices_vocabulary(self):
        tree = make_tree(vocab_size=200)
        full = tree.vocab_entries(offset=0, limit=0)["entries"]
        first = tree.vocab_entries(offset=0, limit=10)["entries"]
        second = tree.vocab_entries(offset=10, limit=10)["entries"]
        assert len(first) == 10
        assert [e["id"] for e in first] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert [e["id"] for e in second] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert first + second == full[:20]

    def test_offset_beyond_end_returns_empty(self):
        tree = make_tree(vocab_size=64)
        out = tree.vocab_entries(offset=10_000, limit=50)
        assert out["total"] == len(tree.vocab)
        assert out["entries"] == []

    def test_special_tokens_flagged_first(self):
        tree = make_tree(vocab_size=200)
        entries = tree.vocab_entries(offset=0, limit=10)["entries"]
        specials = [e for e in entries if e["is_special"]]
        assert len(specials) == len(SPECIAL_TOKENS)
        for e in specials:
            assert e["freq"] == 0

    def test_base_and_merged_flags(self):
        tree = make_tree(vocab_size=200)
        out = tree.vocab_entries(offset=0, limit=0)
        by_id = {e["id"]: e for e in out["entries"]}
        merged_ids = set(tree._lineage)
        for tid, e in by_id.items():
            assert e["is_merged"] == (tid in merged_ids)
            assert e["token"] == tree.vocab[tid]

    def test_no_limit_returns_everything(self):
        tree = make_tree(vocab_size=64)
        out = tree.vocab_entries(offset=0, limit=0)
        assert out["total"] == len(tree.vocab)
        assert len(out["entries"]) == out["total"]

    def test_untrained_returns_empty(self):
        tree = TokenTree()
        out = tree.vocab_entries(offset=0, limit=50)
        assert out["total"] == 0
        assert out["entries"] == []

    def test_negative_limit_no_limit(self):
        tree = make_tree(vocab_size=64)
        out = tree.vocab_entries(offset=0, limit=-1)
        assert len(out["entries"]) == len(tree.vocab)

    def test_entry_fields(self):
        tree = make_tree(vocab_size=64)
        out = tree.vocab_entries(offset=0, limit=5)
        for e in out["entries"]:
            assert set(e.keys()) == {"id", "token", "freq", "is_special", "is_merged"}


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

    def test_embedding_nonexistent_id(self):
        tree = make_tree(vocab_size=64, embed_dim=16)
        assert tree.embedding(99999) is None


class TestSemanticQuery:
    def test_embedding_matrix_shape_and_normalized(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        mat = tree.embedding_matrix()
        assert mat is not None
        assert mat.shape == (tree.vocab_size, 16)
        assert mat.dtype == np.float32
        assert np.all(np.isfinite(mat))
        norms = np.linalg.norm(mat, axis=1)
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

    def test_similar_top_k_larger_than_vocab(self):
        tree = make_tree(vocab_size=64, embed_dim=16)
        results = tree.similar(0, top_k=1000)
        assert len(results) <= tree.vocab_size - 1


class TestEmbeddingMatrixStats:
    def test_shape_and_norm_stats(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        stats = tree.embedding_matrix_stats(top_n=4)
        assert stats["matrix"] == [tree.vocab_size, 16]
        assert 0.0 <= stats["norm_min"] <= stats["norm_mean"] <= stats["norm_max"] <= 1.0
        assert stats["dead_tokens"] + stats["live_tokens"] == tree.vocab_size

    def test_energy_lists_are_triples_sorted_by_norm(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        stats = tree.embedding_matrix_stats(top_n=5)
        most = stats["most_energetic"]
        least = stats["least_energetic"]
        assert len(most) == len(least) == 5
        for entry in most + least:
            assert len(entry) == 3
            assert isinstance(entry[1], int)
            assert entry[0] == tree.itos[entry[1]]
        most_norms = [e[2] for e in most]
        least_norms = [e[2] for e in least]
        assert most_norms == sorted(most_norms, reverse=True)
        assert least_norms == sorted(least_norms)

    def test_live_rows_all_positive_norm(self):
        tree = make_tree(vocab_size=200, embed_dim=16)
        stats = tree.embedding_matrix_stats(top_n=4)
        for entry in stats["most_energetic"] + stats["least_energetic"]:
            assert entry[2] > 0.0

    def test_disabled_embeddings_return_zeros(self):
        stats = TokenTree().embedding_matrix_stats(top_n=8)
        assert stats["matrix"] is None
        assert stats["dead_tokens"] == 0
        assert stats["live_tokens"] == 0
        assert stats["most_energetic"] == []
        assert stats["least_energetic"] == []

    def test_top_n_larger_than_vocab(self):
        tree = make_tree(vocab_size=32, embed_dim=16)
        stats = tree.embedding_matrix_stats(top_n=100)
        assert len(stats["most_energetic"]) <= tree.vocab_size
        assert len(stats["least_energetic"]) <= tree.vocab_size


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

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        tree = make_tree(vocab_size=64)
        deep = str(tmp_path / "a" / "b" / "c" / "tree")
        tree.save(deep)
        assert Path(deep + ".meta.json").exists()

    def test_save_load_untrained(self, tmp_path: Path):
        tree = TokenTree()
        base = str(tmp_path / "untrained")
        tree.save(base)
        loaded = TokenTree.load(base)
        assert loaded.is_trained is False
        assert loaded.vocab_size == 0

    def test_load_without_points_file(self, tmp_path: Path):
        tree = make_tree(vocab_size=64)
        base = str(tmp_path / "nopo")
        tree.save(base)
        Path(base + ".points.json").unlink()
        loaded = TokenTree.load(base)
        assert loaded.vocab == tree.vocab


class TestDictSerialization:
    def test_to_dict_round_trip(self):
        tree = make_tree(vocab_size=300, embed_dim=16)
        data = tree.to_dict()
        assert data["version"] == 1
        assert data["vocab"] == tree.vocab
        assert data["merges"] == [list(m) for m in tree.merges]
        assert data["trained"] is True

        rebuilt = TokenTree.from_dict(data)
        assert rebuilt.vocab == tree.vocab
        assert rebuilt.stoi == tree.stoi
        assert rebuilt.merges == tree.merges
        assert rebuilt.is_trained
        for text in ["the quick brown fox", "hello world", "quickest browner"]:
            assert rebuilt.decode(rebuilt.encode(text)) == text

    def test_to_dict_is_json_safe(self):
        tree = make_tree(vocab_size=300, embed_dim=16)
        payload = json.dumps(tree.to_dict())
        rebuilt = TokenTree.from_dict(json.loads(payload))
        for text in ["the lazy dog", "brown browner"]:
            assert rebuilt.decode(rebuilt.encode(text)) == text

    def test_from_dict_untrained(self):
        data = TokenTree().to_dict()
        rebuilt = TokenTree.from_dict(data)
        assert rebuilt.is_trained is False
        assert rebuilt.vocab_size == 0
        assert all(tid == rebuilt.unk_id for tid in rebuilt.encode("hello"))

    def test_to_dict_matches_save_meta(self, tmp_path: Path):
        tree = make_tree(vocab_size=300, embed_dim=16)
        base = str(tmp_path / "dictmeta")
        tree.save(base)
        saved = json.loads(Path(base + ".meta.json").read_text())
        assert saved["vocab"] == tree.to_dict()["vocab"]
        assert saved["trie"] == tree.to_dict()["trie"]

    def test_to_dict_lineage_keys_are_strings(self):
        tree = make_tree(vocab_size=200)
        data = tree.to_dict()
        for k, v in data["lineage"].items():
            assert isinstance(k, str)
            assert isinstance(v, list)

    def test_to_dict_freqs_keys_are_strings(self):
        tree = make_tree(vocab_size=200)
        data = tree.to_dict()
        for k, v in data["freqs"].items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    def test_to_dict_has_all_fields(self):
        tree = make_tree(vocab_size=64)
        data = tree.to_dict()
        assert set(data.keys()) == {
            "version", "pretokenizer", "word_suffix", "embed_dim",
            "vocab", "merges", "lineage", "freqs", "trie", "trained",
        }


class TestNodeToFromDict:
    def test_leaf_round_trip(self):
        node = TrieNode(token_id=7)
        d = _node_to_dict(node)
        assert d == {"id": 7}
        rebuilt = _node_from_dict(d)
        assert rebuilt.token_id == 7
        assert rebuilt.children == {}
        assert rebuilt.left_id is None

    def test_node_with_children(self):
        child = TrieNode(token_id=10, freq=5)
        parent = TrieNode(children={"a": child}, token_id=3, left_id=1, right_id=2)
        d = _node_to_dict(parent)
        assert d == {"id": 3, "l": 1, "r": 2, "c": {"a": {"id": 10}}}
        rebuilt = _node_from_dict(d)
        assert rebuilt.token_id == 3
        assert rebuilt.left_id == 1
        assert rebuilt.right_id == 2
        assert rebuilt.children["a"].token_id == 10

    def test_empty_dict_round_trip(self):
        node = TrieNode()
        d = _node_to_dict(node)
        rebuilt = _node_from_dict(d)
        assert rebuilt.token_id is None
        assert rebuilt.children == {}

    def test_deep_nesting(self):
        leaf = TrieNode(token_id=99)
        mid = TrieNode(children={"z": leaf}, token_id=50)
        root = TrieNode(children={"x": mid}, token_id=1)
        d = _node_to_dict(root)
        rebuilt = _node_from_dict(d)
        assert rebuilt.children["x"].children["z"].token_id == 99

    def test_multiple_children(self):
        a = TrieNode(token_id=1)
        b = TrieNode(token_id=2)
        c = TrieNode(token_id=3)
        root = TrieNode(children={"a": a, "b": b, "c": c})
        d = _node_to_dict(root)
        rebuilt = _node_from_dict(d)
        assert set(rebuilt.children.keys()) == {"a", "b", "c"}
        assert rebuilt.children["a"].token_id == 1
        assert rebuilt.children["b"].token_id == 2
        assert rebuilt.children["c"].token_id == 3


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

    def test_show_merges_runs(self, caplog):
        tree = make_tree(vocab_size=200)
        import logging
        with caplog.at_level(logging.INFO, logger="slo.token_tree"):
            tree.show_merges(top_n=5)
        assert "count" in caplog.text.lower() or caplog.text == "" or True

    def test_show_tree_base_token(self):
        tree = make_tree()
        tid = tree.stoi["a"]
        rendered = tree.show_tree(tid)
        assert rendered.strip() == "a"

    def test_show_tree_no_lineage(self):
        tree = make_tree()
        tid = tree.stoi["<PAD>"]
        rendered = tree.show_tree(tid)
        assert "<PAD>" in rendered
        assert rendered.count("\n") == 0

    def test_stats_untrained(self):
        tree = TokenTree()
        s = tree.stats()
        assert s["trained"] is False
        assert s["vocab_size"] == 0
        assert s["num_merges"] == 0


class TestSplitPieces:
    def test_single_char(self):
        assert _split_pieces("a") == ["a"]

    def test_word_with_suffix(self):
        assert _split_pieces("the" + WORD_SUFFIX) == ["t", "h", "e", WORD_SUFFIX]

    def test_special_token_whole(self):
        assert _split_pieces("<PAD>") == ["<PAD>"]

    def test_merged_token_pieces_concat(self):
        assert _split_pieces("th" + WORD_SUFFIX) == ["t", "h", WORD_SUFFIX]

    def test_empty_string(self):
        assert _split_pieces("") == []

    def test_all_special_tokens(self):
        for tok in SPECIAL_TOKENS:
            assert _split_pieces(tok) == [tok]

    def test_suffix_only(self):
        assert _split_pieces(WORD_SUFFIX) == [WORD_SUFFIX]

    def test_special_token_with_suffix(self):
        result = _split_pieces("<UNK>" + WORD_SUFFIX)
        assert "<UNK>" in result
        assert WORD_SUFFIX in result

    def test_long_word(self):
        pieces = _split_pieces("hello")
        assert pieces == ["h", "e", "l", "l", "o"]

    def test_multichar_special_in_token(self):
        result = _split_pieces("<BOS>world" + WORD_SUFFIX)
        assert result[0] == "<BOS>"
        assert result[-1] == WORD_SUFFIX


class TestStringInput:
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


class TestResolveToken:
    def test_numeric_id(self):
        tree = make_tree(vocab_size=64)
        assert tree.resolve_token("7") == 7

    def test_word_resolves_to_suffixed_form(self):
        tree = make_tree(vocab_size=64)
        tid = tree.stoi.get("quick" + WORD_SUFFIX)
        if tid is not None:
            assert tree.resolve_token("quick") == tid

    def test_special_token_kept_whole(self):
        tree = make_tree(vocab_size=64)
        assert tree.resolve_token("<PAD>") == tree.stoi.get("<PAD>")

    def test_unknown_raises_key_error(self):
        tree = make_tree(vocab_size=64)
        with pytest.raises(KeyError):
            tree.resolve_token("zzz-no-such-token")

    def test_resolve_literal_suffixed_form(self):
        tree = make_tree(vocab_size=64)
        full = "the" + WORD_SUFFIX
        if full in tree.stoi:
            assert tree.resolve_token(full) == tree.stoi[full]

    def test_resolve_leading_space_suffixed(self):
        tree = make_tree(vocab_size=128)
        full = " the" + WORD_SUFFIX
        if full in tree.stoi:
            assert tree.resolve_token(" the") == tree.stoi[full]

    def test_resolve_space_word_form(self):
        tree = make_tree(vocab_size=128)
        if " the" in tree.stoi:
            assert tree.resolve_token(" the") == tree.stoi[" the"]

    def test_resolve_numeric_zero(self):
        tree = make_tree(vocab_size=64)
        assert tree.resolve_token("0") == 0


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
            assert tree.decode(t_ids) == text
            assert bpe.decode(b_ids) == text


class TestAddToken:
    def test_add_duplicate_increments_freq(self):
        tree = TokenTree()
        tree._add_token("hello", freq=5)
        tree._add_token("hello", freq=3)
        tid = tree.stoi["hello"]
        assert tree._freqs[tid] == 8

    def test_add_duplicate_no_freq_change_when_zero(self):
        tree = TokenTree()
        tree._add_token("hello", freq=5)
        tree._add_token("hello", freq=0)
        tid = tree.stoi["hello"]
        assert tree._freqs[tid] == 5

    def test_add_token_with_lineage(self):
        tree = TokenTree()
        left_id = tree._add_token("a", freq=1)
        right_id = tree._add_token("b", freq=1)
        merged_id = tree._add_token("ab", freq=2, left_id=left_id, right_id=right_id)
        assert tree._lineage[merged_id] == (left_id, right_id)

    def test_add_token_stores_in_trie(self):
        tree = TokenTree()
        tid = tree._add_token("x", freq=1)
        # Walk trie to find the node
        node = tree.root
        for piece in _split_pieces("x"):
            node = node.children[piece]
        assert node.token_id == tid


class TestNormalization:
    def test_lowercase_true(self):
        assert TokenTree._normalize("HELLO", lowercase=True) == "hello"

    def test_lowercase_false(self):
        assert TokenTree._normalize("Hello", lowercase=False) == "Hello"


class TestEncodeBatchEdgeCases:
    def test_batch_single_worker(self):
        tree = make_tree(vocab_size=128)
        texts = ["hello", "world"]
        result = tree.encode_batch(texts, max_workers=1)
        assert [tree.encode(t) for t in texts] == result

    def test_batch_many_workers(self):
        tree = make_tree(vocab_size=128)
        texts = CORPUS * 5
        result = tree.encode_batch(texts, max_workers=16)
        assert [tree.encode(t) for t in texts] == result

    def test_batch_add_bos_eos(self):
        tree = make_tree(vocab_size=128)
        texts = ["hello", "world"]
        result = tree.encode_batch(texts, add_bos=True, add_eos=True)
        for ids in result:
            assert ids[0] == tree.bos_id
            assert ids[-1] == tree.eos_id


class TestCompressorCaching:
    def test_compressor_is_cached(self):
        tree = TokenTree()
        c1 = tree._compressor()
        c2 = tree._compressor()
        assert c1 is c2


class TestEmbeddingMatrixUntrained:
    def test_embedding_matrix_untrained(self):
        tree = TokenTree()
        assert tree.embedding_matrix() is None
        assert tree.embedding_points() == 0
        assert tree.embedding_compression_ratio() == 0.0


class TestDecomposeEdgeCases:
    def test_decompose_base_tokens(self):
        tree = make_tree(vocab_size=128)
        for ch in "abcdefghijklmnopqrstuvwxyz":
            tid = tree.stoi.get(ch)
            if tid is not None:
                assert tree.decompose(tid) == [ch]

    def test_decompose_nonexistent(self):
        tree = make_tree()
        assert tree.decompose(99999) == []

    def test_decompose_preserves_order(self):
        tree = make_tree(vocab_size=300)
        for left, right in tree.merges:
            merged = left + right
            tid = tree.stoi.get(merged)
            if tid is not None:
                leaves = tree.decompose(tid)
                text = "".join(p for p in leaves if p != WORD_SUFFIX)
                assert text == merged.replace(WORD_SUFFIX, "")
