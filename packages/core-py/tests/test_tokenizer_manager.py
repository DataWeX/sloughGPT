"""Tests for domains/training/tokenizer_manager.py TokenizerManager."""

import json
from types import SimpleNamespace

import pytest

from domains.training.tokenizer_manager import (
    _TOKENIZER_ALGO_KEY,
    TokenizerManager,
    get_tokenizer_manager,
)


CORPUS = [
    "hello world this is a test corpus",
    "the quick brown fox jumps over the lazy dog",
    "tokenization is the process of splitting text",
    "bpe merges the most frequent pairs of symbols",
    "unigram finds the most probable segmentation",
]


class TestSingleton:
    def test_get_instance_returns_singleton(self):
        TokenizerManager._instance = None
        a = TokenizerManager.get_instance()
        b = TokenizerManager.get_instance()
        assert a is b

    def test_get_tokenizer_manager_shortcut(self):
        TokenizerManager._instance = None
        assert get_tokenizer_manager() is TokenizerManager.get_instance()

    def test_fresh_manager_untrained(self):
        mgr = TokenizerManager()
        assert mgr._tokenizer is None
        assert mgr.tokenizer_type == "bpe"
        assert mgr.is_trained() is False
        assert mgr.vocab_size == 0


class TestLifecycle:
    def test_get_tokenizer_creates_bpe(self):
        mgr = TokenizerManager()
        from domains.training.tokenizer import SloBPE
        assert isinstance(mgr.get_tokenizer(), SloBPE)

    def test_train_bpe(self):
        mgr = TokenizerManager()
        stats = mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        assert mgr.tokenizer_type == "bpe"
        assert mgr.is_trained() is True
        assert mgr.vocab_size > 0
        assert stats["vocab_size"] > 0

    def test_train_unigram(self):
        mgr = TokenizerManager()
        stats = mgr.train(CORPUS, vocab_size=128, algo="unigram", seed_max_len=4)
        assert mgr.tokenizer_type == "unigram"
        assert stats["vocab_size"] > 0

    def test_train_passes_algo_kwargs(self):
        mgr = TokenizerManager()
        stats = mgr.train(CORPUS, vocab_size=128, algo="unigram",
                          seed_max_len=3, pruning_ratio=0.3)
        assert mgr.tokenizer_type == "unigram"
        assert stats["vocab_size"] > 0

    def test_tokenize_detokenize_roundtrip(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        ids = mgr.tokenize("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert mgr.detokenize(ids) == "hello world"

    def test_tokenizer_type_updates(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=128, algo="unigram", seed_max_len=4)
        assert mgr.tokenizer_type == "unigram"
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        assert mgr.tokenizer_type == "bpe"

    def test_pretokenizer_property(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1, pretokenizer="whitespace")
        assert mgr.pretokenizer == "whitespace"

    def test_pretokenizer_fallback_without_attr(self):
        mgr = TokenizerManager()
        mgr.set_tokenizer(SimpleNamespace(vocab_size=10))
        assert mgr.pretokenizer == "whitespace"

    def test_reset_returns_to_fresh_bpe(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        mgr.reset()
        assert mgr.tokenizer_type == "bpe"
        assert mgr.is_trained() is False
        from domains.training.tokenizer import SloBPE
        assert isinstance(mgr._tokenizer, SloBPE)


class TestStats:
    def test_stats_untrained(self):
        mgr = TokenizerManager()
        s = mgr.stats()
        assert s["vocab_size"] == 0
        assert s["trained"] is False

    def test_stats_trained(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1, algo="bpe")
        s = mgr.stats()
        assert s["trained"] is True
        assert s["algo"] == "bpe"
        assert s["vocab_size"] > 0

    def test_vocab_size_property(self):
        mgr = TokenizerManager()
        assert mgr.vocab_size == 0
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        assert mgr.vocab_size > 0


class TestIntrospection:
    def test_analyze_corpus_delegates(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        result = mgr.analyze_corpus(["hello world"])
        assert isinstance(result, dict)
        assert "error" not in result

    def test_analyze_corpus_fallback(self):
        mgr = TokenizerManager()
        mgr.set_tokenizer(SimpleNamespace(vocab_size=10))
        assert mgr.analyze_corpus(["x"]) == {
            "error": "analyze_corpus not available for this tokenizer type"}

    def test_show_pretokenization_delegates(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        result = mgr.show_pretokenization("hello world")
        assert "error" not in result

    def test_show_pretokenization_fallback(self):
        mgr = TokenizerManager()
        mgr.set_tokenizer(SimpleNamespace(vocab_size=10))
        result = mgr.show_pretokenization("hi")
        assert result["error"].startswith("show_pretokenization not available")
        assert result["pretokens"] == []
        assert result["count"] == 0

    def test_decompose_token_delegates(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        result = mgr.decompose_token("hello")
        assert isinstance(result, dict)

    def test_decompose_token_missing_raises(self):
        mgr = TokenizerManager()
        mgr.set_tokenizer(SimpleNamespace(vocab_size=10))
        with pytest.raises(ValueError):
            mgr.decompose_token("hello")


class TestDirectory:
    def test_train_from_directory_recursive(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.txt").write_text(CORPUS[0], encoding="utf-8")
        (tmp_path / "two.txt").write_text(CORPUS[1], encoding="utf-8")
        mgr = TokenizerManager()
        stats = mgr.train_from_directory(str(tmp_path), pattern="*.txt", vocab_size=256,
                                         min_frequency=1)
        assert mgr.is_trained() is True
        assert stats["vocab_size"] > 0

    def test_train_from_directory_non_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.txt").write_text(CORPUS[2], encoding="utf-8")
        (tmp_path / "top.txt").write_text(CORPUS[3], encoding="utf-8")
        mgr = TokenizerManager()
        mgr.train_from_directory(str(tmp_path), pattern="*.txt", vocab_size=256,
                                 min_frequency=1, recursive=False)
        assert mgr.is_trained() is True
        assert "nested" not in mgr.tokenize("nested word")

    def test_train_from_directory_empty(self, tmp_path):
        mgr = TokenizerManager()
        with pytest.raises(ValueError):
            mgr.train_from_directory(str(tmp_path), pattern="*.txt")


class TestPersistence:
    def test_to_dict_includes_algo(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        data = mgr.to_dict()
        assert data[_TOKENIZER_ALGO_KEY] == "bpe"
        assert "vocab" in data

    def test_from_dict_bpe_roundtrip(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        data = mgr.to_dict()
        mgr2 = TokenizerManager()
        mgr2.from_dict(data)
        assert mgr2.tokenizer_type == "bpe"
        assert mgr2.is_trained() is True
        assert mgr2.tokenize("hello world") == mgr.tokenize("hello world")

    def test_from_dict_unigram_roundtrip(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=128, algo="unigram", seed_max_len=4)
        data = mgr.to_dict()
        mgr2 = TokenizerManager()
        mgr2.from_dict(data)
        assert mgr2.tokenizer_type == "unigram"
        assert mgr2.tokenize("hello world") == mgr.tokenize("hello world")

    def test_save_load_roundtrip(self, tmp_path):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        path = str(tmp_path / "tok.json")
        mgr.save(path)
        mgr2 = TokenizerManager()
        mgr2.load(path)
        assert mgr2.tokenizer_type == "bpe"
        assert mgr2.is_trained() is True
        assert mgr2.tokenize("hello world") == mgr.tokenize("hello world")

    def test_saved_file_is_json(self, tmp_path):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        path = str(tmp_path / "tok.json")
        mgr.save(path)
        data = json.loads(open(path, encoding="utf-8").read())
        assert _TOKENIZER_ALGO_KEY in data


class TestIntegration:
    def test_borrow_from_autotrain_when_trained(self):
        mgr = TokenizerManager()
        mgr.train(CORPUS, vocab_size=256, min_frequency=1)
        assert mgr.borrow_from_autotrain() is True

    def test_borrow_from_autotrain_unavailable(self):
        mgr = TokenizerManager()
        assert mgr.borrow_from_autotrain() is False

    def test_borrow_from_autotrain_adopts(self, monkeypatch):
        import types as _types
        import sys
        fake_tok = SimpleNamespace(vocab_size=100)
        fake_mod = _types.ModuleType("routers.auto_train")
        fake_mod.state = SimpleNamespace(student_tokenizer=fake_tok)
        monkeypatch.setitem(sys.modules, "routers", _types.ModuleType("routers"))
        monkeypatch.setitem(sys.modules, "routers.auto_train", fake_mod)
        mgr = TokenizerManager()
        assert mgr.borrow_from_autotrain() is True
        assert mgr._tokenizer is fake_tok

    def test_borrow_from_autotrain_ignores_small(self, monkeypatch):
        import types as _types
        import sys
        fake_mod = _types.ModuleType("routers.auto_train")
        fake_mod.state = SimpleNamespace(student_tokenizer=SimpleNamespace(vocab_size=3))
        monkeypatch.setitem(sys.modules, "routers", _types.ModuleType("routers"))
        monkeypatch.setitem(sys.modules, "routers.auto_train", fake_mod)
        mgr = TokenizerManager()
        assert mgr.borrow_from_autotrain() is False

    def test_adopt_valid_tokenizer(self):
        mgr = TokenizerManager()
        fake = SimpleNamespace(vocab_size=5, encode=lambda t: [1], decode=lambda i: "x")
        mgr.adopt(fake)
        assert mgr._tokenizer is fake

    def test_adopt_invalid_tokenizer(self):
        mgr = TokenizerManager()
        mgr.adopt(SimpleNamespace(vocab_size=0))
        assert mgr._tokenizer is None

    def test_set_tokenizer(self):
        mgr = TokenizerManager()
        fake = SimpleNamespace(vocab_size=3)
        mgr.set_tokenizer(fake)
        assert mgr.get_tokenizer() is fake
