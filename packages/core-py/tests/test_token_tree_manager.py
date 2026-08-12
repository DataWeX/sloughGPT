"""Tests for TokenTreeManager — lazy default training, train, and queries."""

import pytest

from domains.training.token_tree_manager import (
    DEFAULT_CORPUS,
    TokenTreeManager,
    get_token_tree_manager,
)
import domains.training.token_tree_manager as token_tree_manager_module


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
