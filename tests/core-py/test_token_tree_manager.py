"""Tests for TokenTreeManager — lazy default training, train, and queries."""

import pytest

import domain.training._internal.token_tree_manager as token_tree_manager_module
from domain.training._internal.token_tree_manager import (
    DEFAULT_CORPUS,
    TokenTree,
    TokenTreeManager,
    get_token_tree_manager,
)


@pytest.fixture(autouse=True)
def _reset_manager():
    """Give every test a fresh singleton state."""
    old = TokenTreeManager._instance
    TokenTreeManager._instance = None
    yield
    TokenTreeManager._instance = old


class TestLazyTraining:
    def test_get_tree_trains_default(self):
        mgr = TokenTreeManager.get_instance()
        assert mgr.is_trained() is False
        tree = mgr.get_tree(vocab_size=64, embed_dim=8)
        assert mgr.is_trained() is True
        assert tree.is_trained
        assert " quick" + "</w>" in tree.stoi

    def test_default_corpus_is_builtin(self):
        assert len(DEFAULT_CORPUS) >= 5
        assert any("quick" in d for d in DEFAULT_CORPUS)

    def test_get_tree_is_cached(self):
        mgr = TokenTreeManager.get_instance()
        a = mgr.get_tree(vocab_size=64)
        b = mgr.get_tree(vocab_size=64)
        assert a is b

    def test_singleton_shared(self):
        assert get_token_tree_manager() is TokenTreeManager.get_instance()


class TestDefaultCache:
    """Persisted default tree: cached on first train, loaded (not retrained) later."""

    def _purge_cache(self):
        cache_dir = token_tree_manager_module._SAVE_DIR / "_cache"
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)

    def test_default_tree_is_persisted_to_cache(self):
        self._purge_cache()
        first = TokenTreeManager.get_instance()
        tree = first.get_tree(vocab_size=64, embed_dim=8)
        assert tree.is_trained
        cache_path = first._default_cache_path(64, 8)
        assert (str(cache_path) + ".meta.json").endswith("_default_v64_e8.meta.json")
        # The cache lives under _cache/ and must never surface as a saved tree.
        saved = first.list_saved()
        assert not any(
            entry["name"].startswith("_cache_") or "_default_" in entry["name"] for entry in saved
        )

    def test_cold_start_reuses_cache(self):
        self._purge_cache()
        first = TokenTreeManager.get_instance()
        expected = first.get_tree(vocab_size=64, embed_dim=8)
        assert expected.is_trained

        # Simulate a fresh process: new singleton with no in-memory tree
        # must reconstruct from the persisted cache instead of retraining.
        TokenTreeManager._instance = None
        second = TokenTreeManager.get_instance()
        tree = second.get_tree(vocab_size=64, embed_dim=8)
        assert tree is not expected
        assert tree.encode("the quick brown fox") == expected.encode("the quick brown fox")
        assert second._load_cached_default(64, 8) is not None

    def test_cache_is_param_keyed(self):
        self._purge_cache()
        first = TokenTreeManager.get_instance()
        a = first.get_tree(vocab_size=64, embed_dim=8)

        TokenTreeManager._instance = None
        second = TokenTreeManager.get_instance()
        b = second.get_tree(vocab_size=32, embed_dim=8)
        assert a.vocab_size != b.vocab_size
        assert second._default_cache_path(64, 8) != second._default_cache_path(32, 8)


class TestExplicitTrain:
    def test_train_replaces_tree(self):
        mgr = TokenTreeManager.get_instance()
        first = mgr.get_tree(vocab_size=64)
        second = mgr.train(["apple banana apple banana"], vocab_size=32)
        assert mgr.get_tree() is second
        assert second is not first
        assert "apple" + "</w>" in second.stoi

    def test_stats_reflects_trained_tree(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(["foo bar foo baz foo"], vocab_size=32)
        stats = mgr.stats()
        assert stats["trained"] is True
        assert stats["vocab_size"] > 0
        assert "vocab_size" in stats


class TestAdopt:
    def test_adopt_replaces_current_tree(self):
        from domain.training._internal.token_tree import TokenTree

        mgr = TokenTreeManager.get_instance()
        before = mgr.get_tree(vocab_size=32)
        external = TokenTree().train(["zzz zzz qux qux"], vocab_size=16, min_frequency=1)
        adopted = mgr.adopt(external)
        assert adopted is external
        assert mgr.get_tree() is external
        assert mgr.get_tree() is not before

    def test_adopt_tree_saves_and_queries(self, tmp_path, monkeypatch):
        from domain.training._internal.token_tree import TokenTree

        monkeypatch.setattr("domain.training._internal.token_tree_manager._SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        external = TokenTree().train(["the quick brown fox"], vocab_size=32, min_frequency=1)
        mgr.adopt(external)
        info = mgr.save("adopted")
        assert info["name"] == "adopted"
        assert mgr.stats()["trained"] is True


class TestQueries:
    def test_similar_returns_ranked_neighbors(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.similar("quick")
        assert out["neighbors"]
        scores = [n["score"] for n in out["neighbors"]]
        assert scores == sorted(scores, reverse=True)
        assert all(n["token"] for n in out["neighbors"])

    def test_similar_numeric_id(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        tree = mgr.get_tree()
        out = mgr.similar(str(tree.stoi["the</w>"]))
        assert out["neighbors"]

    def test_similar_unknown_raises(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        with pytest.raises(KeyError):
            mgr.similar("zzz-no-such-token")

    def test_encode_and_decode_round_trip(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        enc = mgr.encode("the quick brown fox")
        assert enc["ids"]
        assert len(enc["tokens"]) == len(enc["ids"])
        assert mgr.decode(enc["ids"])["text"] == "the quick brown fox"

    def test_path_matches_encode_ids(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.path("the quick brown fox")
        enc = mgr.encode("the quick brown fox")
        assert out["ids"] == enc["ids"]
        assert len(out["steps"]) == len(out["ids"])

    def test_path_reports_remaining_suffix_and_consumed(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.path("quick")
        first = out["steps"][0]
        assert first["remaining"].startswith("quick")
        assert first["consumed"] >= 1
        assert first["id"] == mgr.get_tree().stoi[first["remaining"][: first["consumed"]]]
        assert all(s["consumed"] >= 1 for s in out["steps"])

    def test_path_steps_consume_input_left_to_right(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.path("the quick")
        # two pretokenized words, each padded with the 4-char word suffix
        total = sum(s["consumed"] for s in out["steps"])
        assert total == len("the quick") + 8
        assert out["steps"][0]["remaining"] == "the</w>"

    def test_lineage_decomposes(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.lineage("quick")
        assert out["leaves"]
        assert out["tree"]
        assert "".join(p for p in out["leaves"] if p != "</w>").strip() == "quick"

    def test_lineage_unknown_raises(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        with pytest.raises(KeyError):
            mgr.lineage("zzz-no-such-token")


class TestEmbeddingInfo:
    def test_embedding_info_literal_token(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.embedding_info("quick")
        assert out["id"] == mgr.get_tree().stoi[" quick</w>"]
        assert out["dim"] == 8
        assert out["norm"] > 0
        assert len(out["top"]) == 8
        for dim, value in out["top"]:
            assert 0 <= dim < 8
            assert isinstance(value, float)
        assert out["embedding_points"] >= 1

    def test_embedding_info_numeric_id(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        tree = mgr.get_tree()
        tid = tree.stoi["the</w>"]
        out = mgr.embedding_info(str(tid))
        assert out["id"] == tid

    def test_embedding_info_top_k_caps_dimensions(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.embedding_info("quick", top_k=3)
        assert len(out["top"]) == 3

    def test_embedding_info_compression_ratio_present(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.embedding_info("quick")
        assert out["compression_ratio"] > 0
        assert out["token"]

    def test_embedding_info_unknown_raises(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        with pytest.raises(KeyError):
            mgr.embedding_info("zzz-no-such-token")


class TestMatrixSummary:
    def test_matrix_summary_shape_and_totals(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.matrix_summary(top_k=4)
        assert out["matrix"] == [mgr.get_tree().vocab_size, 8]
        assert out["live_tokens"] + out["dead_tokens"] == mgr.get_tree().vocab_size
        assert out["norm_min"] <= out["norm_max"] <= 1.0

    def test_matrix_summary_energy_entries(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.matrix_summary(top_k=3)
        assert len(out["most_energetic"]) == 3
        assert len(out["least_energetic"]) == 3
        for entry in out["most_energetic"]:
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], int)
            assert entry[2] > 0

    def test_matrix_summary_disabled_embeddings(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=0)
        out = mgr.matrix_summary(top_k=4)
        assert out["matrix"] is None
        assert out["most_energetic"] == []


class TestManagerResultCache:
    def test_similar_is_memoized(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        first = mgr.similar("quick")
        assert mgr.similar("quick") is first

    def test_embedding_info_is_memoized(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        first = mgr.embedding_info("quick")
        assert mgr.embedding_info("quick") is first

    def test_matrix_summary_is_memoized(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        first = mgr.matrix_summary(top_k=4)
        assert mgr.matrix_summary(top_k=4) is first

    def test_top_k_is_part_of_the_key(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        assert mgr.matrix_summary(top_k=2) is not mgr.matrix_summary(top_k=4)

    def test_training_a_new_tree_drops_cache(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        first = mgr.similar("quick")
        mgr.train(["apple banana apple banana"], vocab_size=32, embed_dim=8)
        second = mgr.similar("apple")
        assert first is not second
        assert mgr._result_cache_tree_id == id(mgr.get_tree())

    def test_unknown_token_is_not_cached(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        with pytest.raises(KeyError):
            mgr.similar("zzz-no-such-token")
        with pytest.raises(KeyError):
            mgr.similar("zzz-no-such-token")
        assert all("zzz-no-such-token" not in k for k in mgr._result_cache)

    def test_lru_evicts_beyond_cap(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        mgr._result_cache_max = 2
        mgr.similar("quick")
        mgr.similar("quick", top_k=3)
        mgr.similar("quick", top_k=5)
        similar_keys = [k for k in mgr._result_cache if k.startswith("similar:")]
        assert len(similar_keys) == 2


class TestTopMerges:
    def test_top_merges_ranked(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        rules = mgr.top_merges(top_n=5)
        assert len(rules) == 5
        counts = [r["count"] for r in rules]
        assert counts == sorted(counts, reverse=True)
        assert all(r["token"] == r["left"] + r["right"] for r in rules)

    def test_top_merges_defaults_to_twenty(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        assert len(mgr.top_merges()) == 20


class TestSearchMerges:
    def test_search_filters_rules(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        rules = mgr.search_merges(query="qu", limit=10)
        assert rules
        assert len(rules) <= 10
        for r in rules:
            assert (
                "qu" in r["left"].lower()
                or "qu" in r["right"].lower()
                or "qu" in r["token"].lower()
            )

    def test_search_keeps_global_rank(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        top = {r["token"]: r["rank"] for r in mgr.top_merges(top_n=100)}
        for r in mgr.search_merges(query="e", limit=100):
            assert r["rank"] == top[r["token"]]

    def test_search_no_match_empty(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        assert mgr.search_merges(query="zzz-no-such-part", limit=10) == []


class TestVocabEntries:
    def test_paged_entries_match_tree(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        out = mgr.vocab_entries(offset=0, limit=50)
        tree = mgr.get_tree()
        assert out["total"] == len(tree.vocab)
        assert [e["token"] for e in out["entries"]] == tree.vocab[: len(out["entries"])]
        assert [e["id"] for e in out["entries"]] == list(range(len(out["entries"])))

    def test_second_page(self):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        first = mgr.vocab_entries(offset=0, limit=10)["entries"]
        second = mgr.vocab_entries(offset=10, limit=10)["entries"]
        assert [e["id"] for e in first] == list(range(10))
        assert [e["id"] for e in second] == list(range(10, 20))

    def test_lazy_default_tree_returns_entries(self):
        mgr = TokenTreeManager.get_instance()
        out = mgr.vocab_entries(offset=0, limit=50)
        assert out["total"] == len(mgr.get_tree().vocab)
        assert len(out["entries"]) == min(50, out["total"])


class TestPersistence:
    def test_save_writes_sidecars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        info = mgr.save("my-tree")
        assert info["name"] == "my-tree"
        assert info["vocab_size"] > 0
        assert (tmp_path / "my-tree.meta.json").exists()
        assert (tmp_path / "my-tree.points.json").exists()

    def test_load_replaces_current_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        saved = mgr.save("saved")
        mgr.train(["tiny corpus one two three"], vocab_size=32, min_frequency=1)
        assert mgr.stats()["vocab_size"] != saved["vocab_size"]
        loaded = mgr.load("saved")
        assert loaded["name"] == "saved"
        assert loaded["vocab_size"] == saved["vocab_size"]
        assert mgr.stats()["vocab_size"] == saved["vocab_size"]

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            TokenTreeManager.get_instance().load("missing")

    def test_list_saved_returns_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        mgr.save("alpha")
        mgr.save("beta")
        names = [t["name"] for t in mgr.list_saved()]
        assert set(names) == {"alpha", "beta"}
        assert all(t["vocab_size"] > 0 for t in mgr.list_saved())

    def test_list_saved_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path / "no-such-dir")
        assert TokenTreeManager.get_instance().list_saved() == []

    def test_delete_saved_removes_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        mgr.save("doomed")
        assert mgr.delete_saved("doomed") is True
        assert not (tmp_path / "doomed.meta.json").exists()
        assert not (tmp_path / "doomed.points.json").exists()
        assert mgr.delete_saved("doomed") is False

    def test_sanitize_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        for bad in ("", "   ", "../escape", "a/b", "a\\b", ".hidden"):
            with pytest.raises(ValueError):
                mgr.save(bad)
        assert not tmp_path.exists() or not any(tmp_path.iterdir())

    def test_sanitize_accepts_plain_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        info = mgr.save("shakespeare.v2")
        assert info["name"] == "shakespeare.v2"
        assert (tmp_path / "shakespeare.v2.meta.json").exists()


class TestCompare:
    def _save_two(self, tmp_path, monkeypatch):
        """Save two distinct trees and return the manager."""
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        mgr.train(["alpha alpha beta gamma gamma delta"], vocab_size=32, min_frequency=1)
        mgr.save("tree-a")
        mgr.train(["beta beta epsilon zeta zeta"], vocab_size=32, min_frequency=1)
        mgr.save("tree-b")
        return mgr

    def test_compare_reports_counts(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        out = mgr.compare("tree-a", "tree-b")
        assert out["a"]["name"] == "tree-a"
        assert out["b"]["name"] == "tree-b"
        assert out["shared_tokens"] > 0
        assert out["shared_merges"] >= 0
        assert out["a"]["stats"]["vocab_size"] > 0
        assert out["b"]["stats"]["vocab_size"] > 0

    def test_compare_does_not_touch_current_tree(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        mgr.train(["different corpus entirely"], vocab_size=16, min_frequency=1)
        current = mgr.stats()["vocab_size"]
        out = mgr.compare("tree-a", "tree-b")
        assert out["a"]["stats"]["vocab_size"] != current
        assert mgr.stats()["vocab_size"] == current

    def test_compare_identical_trees(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        mgr.load("tree-a")
        mgr.save("tree-a-copy")
        out = mgr.compare("tree-a", "tree-a-copy")
        vsize = out["a"]["stats"]["vocab_size"]
        assert out["shared_tokens"] == vsize
        assert out["only_a_tokens"] == 0
        assert out["only_b_tokens"] == 0
        assert out["shared_merges"] == out["a"]["stats"]["num_merges"]
        assert out["only_a_merges"] == 0
        assert out["only_b_merges"] == 0
        assert len(out["shared_examples"]) == min(10, vsize)

    def test_compare_self_raises(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            mgr.compare("tree-a", "tree-a")

    def test_compare_missing_raises(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError):
            mgr.compare("tree-a", "ghost")
        with pytest.raises(FileNotFoundError):
            mgr.compare("ghost", "tree-a")

    def test_compare_examples_ranked_by_freq(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        out = mgr.compare("tree-a", "tree-b")
        for side in ("only_a_examples", "only_b_examples", "shared_examples"):
            ranks = out[side]
            assert isinstance(ranks, list)
            freqs = [item[1] for item in ranks]
            assert freqs == sorted(freqs, reverse=True)
            for item in ranks:
                assert isinstance(item[0], str)
                assert isinstance(item[1], int)

    def test_compare_examples_cap_top_n(self, tmp_path, monkeypatch):
        mgr = self._save_two(tmp_path, monkeypatch)
        out = mgr.compare("tree-a", "tree-b", top_n=3)
        assert len(out["only_a_examples"]) <= 3
        assert len(out["only_b_examples"]) <= 3
        assert len(out["shared_examples"]) <= 3

    def test_compare_sanitizes_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(token_tree_manager_module, "_SAVE_DIR", tmp_path)
        mgr = TokenTreeManager.get_instance()
        mgr.train(["corpus one"], vocab_size=16, min_frequency=1)
        mgr.save("plain")
        for bad in ("../escape", "a/b"):
            with pytest.raises(ValueError):
                mgr.compare(bad, "plain")
            with pytest.raises(ValueError):
                mgr.compare("plain", bad)


class TestServerResponseCache:
    """stats/top_merges/search_merges/vocab_entries must be memoized per tree
    (like similar/embedding_info/matrix_summary already are). These four are the
    heavy paths behind the WRN slow-request logs on /token-tree/stats (~3.1s)
    and /token-tree/merges (~1.3s): they recomputed full-tree math every
    request instead of routing through the manager's per-tree LRU.
    """

    def _wrap_tree(self, tree) -> None:
        """Install recompute counters on a tree instance.

        All spies share one ``self._calls`` dict so tests can wrap a post-adopt
        tree and still reason about cumulative recomputes.
        """
        if not hasattr(self, "_calls"):
            self._calls = {"stats": 0, "top_merges": 0, "search_merges": 0, "vocab_entries": 0}

        for name in self._calls:
            orig = getattr(tree, name)

            def spy(*args, _n=name, _o=orig, **kwargs):
                self._calls[_n] += 1
                return _o(*args, **kwargs)

            setattr(tree, name, spy)

    def _fresh(self, vocab_size: int = 102):
        mgr = TokenTreeManager.get_instance()
        mgr.train(list(DEFAULT_CORPUS), vocab_size=vocab_size, embed_dim=8)
        self._wrap_tree(mgr.get_tree())
        return mgr

    def test_stats_cached_across_requests(self):
        mgr = self._fresh()
        a = mgr.stats()
        b = mgr.stats()
        assert a is b
        assert self._calls["stats"] == 1

    def test_top_merges_cached_across_requests(self):
        mgr = self._fresh()
        a = mgr.top_merges(top_n=10)
        b = mgr.top_merges(top_n=10)
        assert a is b
        assert self._calls["top_merges"] == 1

    def test_search_merges_cached_per_query(self):
        mgr = self._fresh()
        a = mgr.search_merges("quick", limit=10)
        b = mgr.search_merges("quick", limit=10)
        assert a is b
        c = mgr.search_merges("fox", limit=8)
        assert c is not a
        assert self._calls["search_merges"] == 2

    def test_vocab_entries_cached_per_page(self):
        mgr = self._fresh()
        a = mgr.vocab_entries(offset=0, limit=50)
        b = mgr.vocab_entries(offset=0, limit=50)
        assert a is b
        c = mgr.vocab_entries(offset=50, limit=50)
        assert c is not a
        assert self._calls["vocab_entries"] == 2

    def test_cache_invalidated_on_adopt(self):
        mgr = self._fresh()
        first = mgr.stats()
        other = TokenTree().train(list(DEFAULT_CORPUS), vocab_size=64, embed_dim=8)
        mgr.adopt(other)
        self._wrap_tree(mgr.get_tree())  # spy the *new* tree post-adopt
        second = mgr.stats()
        assert second is not first  # identity proves cache was dropped
        assert self._calls["stats"] == 2  # ...and the new tree actually recomputed
