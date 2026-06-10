"""Tests for SloBPE, SloUnigram, and TokenizerManager integration."""

import json
import tempfile
from pathlib import Path

import pytest

from domains.training.tokenizer import SloBPE, SloUnigram
from domains.training.tokenizer_manager import get_tokenizer_manager


CORPUS = ["hello world", "hello there", "hello hello", "world peace", "machine learning"]


def _reset():
    mgr = get_tokenizer_manager()
    mgr.reset()
    return mgr


# ── SloBPE ──

class TestSloBPE:

    def test_train_creates_vocab(self):
        tok = SloBPE()
        tok.train(CORPUS, vocab_size=64)
        assert tok.vocab_size >= 20

    def test_encode_decode_roundtrip(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        ids = tok.encode("hello world")
        assert tok.decode(ids) == "hello world"

    def test_special_tokens_present(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert sp in tok.stoi

    def test_add_special_tokens(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        n = tok.add_special_tokens(["<CUSTOM>"])
        assert n == 1
        assert "<CUSTOM>" in tok.stoi
        assert tok.is_special("<CUSTOM>")

    def test_is_special(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.is_special("<PAD>")
        assert not tok.is_special("hello")

    def test_special_ids(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        ids = tok.special_ids
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert tok.stoi[sp] in ids

    def test_serialization(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        tok2 = SloBPE.from_dict(tok.to_dict())
        assert tok2.decode(tok2.encode("hello world")) == "hello world"
        assert tok2.vocab_size == tok.vocab_size

    def test_empty_text(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_lowercase(self):
        tok = SloBPE(); tok.train(["HELLO WORLD"], vocab_size=32, lowercase=True)
        assert tok.encode("hello world") == tok.encode("HELLO WORLD")

    def test_pretokenizer_gpt2(self):
        tok = SloBPE(pretokenizer="gpt2"); tok.train(CORPUS, vocab_size=64)
        assert len(tok.encode("don't")) >= 1

    def test_show_pretokenization(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        r = tok.show_pretokenization("hello world")
        assert r["count"] >= 1

    def test_decompose_base_char(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        r = tok.decompose_token("a")
        assert r["type"] == "base_char"

    def test_decompose_special(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        r = tok.decompose_token("<PAD>")
        assert r["type"] == "special"

    def test_analyze_corpus(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        r = tok.analyze_corpus(CORPUS)
        assert r["total_chars"] > 0 and r["compression_ratio"] > 0

    def test_vocab_stats(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        s = tok.vocab_stats()
        assert "total_merges_learned" in s


# ── SloUnigram ──

class TestSloUnigram:

    def test_train_creates_vocab(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.vocab_size >= 16

    def test_encode_decode_roundtrip(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        ids = tok.encode("hello world")
        assert tok.decode(ids) == "hello world"

    def test_special_tokens_present(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert sp in tok.stoi

    def test_serialization(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        tok2 = SloUnigram.from_dict(tok.to_dict())
        assert tok2.decode(tok2.encode("hello world")) == "hello world"

    def test_empty_text(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_deterministic_encode(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.encode("hello") == tok.encode("hello")

    def test_show_pretokenization(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        r = tok.show_pretokenization("hello world")
        assert r["count"] >= 1

    def test_decompose_base_char(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        r = tok.decompose_token("a")
        assert r["type"] == "base_char"
        assert "score" in r

    def test_decompose_special(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        r = tok.decompose_token("<PAD>")
        assert r["type"] == "special"

    def test_analyze_corpus(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        r = tok.analyze_corpus(CORPUS)
        assert r["total_chars"] > 0 and r["compression_ratio"] > 0

    def test_vocab_stats_type(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.vocab_stats()["type"] == "unigram"

    def test_lowercase(self):
        tok = SloUnigram(); tok.train(["HELLO WORLD"], vocab_size=32, lowercase=True)
        assert tok.encode("hello world") == tok.encode("HELLO WORLD")


# ── TokenizerManager Integration ──

class TestManager:

    def teardown_method(self):
        _reset()

    def test_default_is_bpe(self):
        assert _reset().tokenizer_type == "bpe"

    def test_train_bpe(self):
        mgr = _reset()
        s = mgr.train(CORPUS, vocab_size=64, algo="bpe")
        assert s["vocab_size"] >= 20

    def test_train_unigram(self):
        mgr = _reset()
        s = mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert s["vocab_size"] >= 10

    def test_encode_decode_bpe(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="bpe")
        assert mgr.detokenize(mgr.tokenize("hello world")) == "hello world"

    def test_encode_decode_unigram(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert mgr.detokenize(mgr.tokenize("hello world")) == "hello world"

    def test_stats_algo(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert mgr.stats()["algo"] == "unigram"

    def test_is_trained(self):
        mgr = _reset()
        assert not mgr.is_trained()
        mgr.train(CORPUS, vocab_size=64)
        assert mgr.is_trained()

    def test_serialization_bpe(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="bpe")
        data = mgr.to_dict()  # capture before reset
        mgr.reset()
        mgr.from_dict(data)
        assert mgr.tokenizer_type == "bpe"
        assert mgr.detokenize(mgr.tokenize("hello world")) == "hello world"

    def test_serialization_unigram(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        data = mgr.to_dict()  # capture before reset
        mgr.reset()
        mgr.from_dict(data)
        assert mgr.tokenizer_type == "unigram"
        assert mgr.detokenize(mgr.tokenize("hello world")) == "hello world"

    def test_reset_clears_algo(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert mgr.tokenizer_type == "unigram"
        mgr.reset()
        assert mgr.tokenizer_type == "bpe" and not mgr.is_trained()

    def test_algo_kwargs_passthrough(self):
        mgr = _reset()
        mgr.train(["a b c d e f g"], vocab_size=16, algo="unigram",
                   seed_max_len=4, pruning_ratio=0.3, em_iters=2)
        assert mgr.is_trained()

    def test_analyze_corpus_bpe(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="bpe")
        assert mgr.analyze_corpus(CORPUS)["total_chars"] > 0

    def test_analyze_corpus_unigram(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert mgr.analyze_corpus(CORPUS)["total_chars"] > 0

    def test_show_pretokenization_bpe(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="bpe")
        assert mgr.show_pretokenization("hello world")["count"] >= 1

    def test_show_pretokenization_unigram(self):
        mgr = _reset(); mgr.train(CORPUS, vocab_size=64, algo="unigram")
        assert mgr.show_pretokenization("hello world")["count"] >= 1


# ── SloEngine integration ──

class TestSloEngineLearn:

    def test_learn_bpe(self):
        from domains.core.soul import SloEngine
        engine = SloEngine()
        result = engine.learn(["hello world", "hello there"], epochs=1, vocab_size=64, algo="bpe")
        assert result["success"]
        assert result["soul_name"] == "assistant"
        assert result["vocab_size"] >= 15
        assert result["model_type"] == "slonet-lstm"
        assert "steps" in result
        assert "loss" in result
        tok = engine._tokenizer
        assert tok is not None
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_learn_unigram(self):
        from domains.core.soul import SloEngine
        engine = SloEngine()
        result = engine.learn(["hello world", "hello there"], epochs=1, vocab_size=64, algo="unigram")
        assert result["success"]
        assert result["soul_name"] == "assistant"
        assert result["vocab_size"] >= 20
        assert "steps" in result
        assert "loss" in result
        tok = engine._tokenizer
        assert tok is not None
        ids = tok.encode("hello world")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert "hello world" in decoded or decoded == "hello world"

    def test_learn_empty_texts(self):
        from domains.core.soul import SloEngine
        engine = SloEngine()
        result = engine.learn([], epochs=1, algo="bpe")
        assert not result["success"]
        assert "error" in result

    def test_learn_preserves_soul_name(self):
        from domains.core.soul import SloEngine
        engine = SloEngine()
        result = engine.learn(["hello world"], epochs=1, vocab_size=64, algo="unigram", soul_name="test-soul")
        assert result["success"]
        assert result["soul_name"] == "test-soul"
