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


# ── Coverage: SloBPE edge paths ──

class TestBPEHelpers:

    def test_default_pretokenize(self):
        from domains.training.tokenizer import default_pretokenize, gpt2_pretokenize
        assert default_pretokenize("hello world") == ["hello", "world"]
        assert gpt2_pretokenize("don't") == ["don", "'t"]

    def test_bos_eos_ids(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.bos_id == tok.stoi["<BOS>"]
        assert tok.eos_id == tok.stoi["<EOS>"]
        assert tok.pad_id == tok.stoi["<PAD>"]
        assert tok.unk_id == tok.stoi["<UNK>"]

    def test_train_empty_texts_raises(self):
        with pytest.raises(ValueError):
            SloBPE().train([])

    def test_train_no_pairs_breaks(self):
        tok = SloBPE(); tok.train([""], vocab_size=64)
        assert tok.vocab_size == 5

    def test_train_verbose_logs(self):
        tok = SloBPE()
        tok.train(["abc abc abc"], vocab_size=10, verbose=True)
        assert tok.vocab_size == 10

    def test_train_reaches_target(self):
        tok = SloBPE(pretokenizer="whitespace")
        tok.train(["abc abc abc"], vocab_size=10)
        assert tok.vocab_size == 10
        assert tok.encode("abc") == tok.encode("abc")

    def test_encode_bos_eos(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_encode_batch_padded(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        batch = tok.encode_batch(["hello world", "hi"], add_bos=True, add_eos=True,
                                 max_length=6, pad=True)
        assert len(batch) == 2
        assert all(len(ids) == 6 for ids in batch)
        assert batch[1][-1] == tok.pad_id

    def test_encode_batch_plain(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        batch = tok.encode_batch(["hello world", "hi"])
        assert len(batch) == 2
        assert all(isinstance(ids, list) for ids in batch)

    def test_decode_out_of_range(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.decode([10 ** 9]) == "?"
        assert tok.decode([-5]) == "?"

    def test_decode_skips_special(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.decode([tok.bos_id, tok.eos_id]) == ""

    def test_decode_unk_not_skipped(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.decode([tok.unk_id], skip_special=False) == "?"

    def test_decode_whitespace_pretokenizer(self):
        tok = SloBPE(pretokenizer="whitespace"); tok.train(CORPUS, vocab_size=64)
        assert tok.decode(tok.encode("hello world")) == "hello world"

    def test_from_dict_v1_defaults_whitespace(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        data = tok.to_dict(); data.pop("version")
        tok2 = SloBPE.from_dict(data)
        assert tok2._pretokenizer == "whitespace"

    def test_from_dict_adds_missing_specials(self):
        data = {
            "version": 2,
            "vocab": ["a", "b", "c"],
            "merges": [],
            "stoi": {"a": 0, "b": 1, "c": 2},
            "itos": {"0": "a", "1": "b", "2": "c"},
            "pretokenizer": "gpt2",
        }
        tok = SloBPE.from_dict(data)
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert sp in tok.stoi
        assert tok.vocab_size == 7

    def test_save_load_roundtrip(self, tmp_path):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        p = str(tmp_path / "bpe.json")
        tok.save(p)
        tok2 = SloBPE.load(p)
        assert tok2.decode(tok2.encode("hello world")) == "hello world"

    def test_export_to_checkpoint(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        cp = tok.export_to_checkpoint()
        assert cp["vocab_size"] == tok.vocab_size
        assert cp["tokenizer_type"] == "slonet_bpe"
        assert set(cp["stoi"]) == set(tok.stoi)

    def test_from_checkpoint_full(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        tok2 = SloBPE.from_checkpoint(tok.export_to_checkpoint())
        assert tok2.stoi == tok.stoi
        assert tok2.vocab_size == tok.vocab_size

    def test_from_checkpoint_no_stoi(self):
        tok = SloBPE.from_checkpoint({"chars": list("abc"), "itos": {}})
        assert "a" in tok.stoi
        assert "b" in tok.stoi

    def test_from_checkpoint_bad_itos_keys(self):
        cp = {"itos": {"x": "a", "y": "b"}, "stoi": {"a": 0, "b": 1, "<PAD>": 2}}
        tok = SloBPE.from_checkpoint(cp)
        assert tok.itos[0] == "a"
        assert tok.itos[1] == "b"
        for sp in ["<UNK>", "<BOS>", "<EOS>"]:
            assert sp in tok.stoi

    def test_show_merges(self):
        tok = SloBPE(pretokenizer="whitespace"); tok.train(["abc abc abc"], vocab_size=10)
        tok.show_merges(top_n=1)

    def test_show_vocab(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        tok.show_vocab(top_n=5)

    def test_add_special_existing_not_special(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        n = tok.add_special_tokens(["hello"])
        assert n == 0
        assert tok.is_special("hello")

    def test_is_special_int(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        assert tok.is_special(tok.stoi["<PAD>"])
        assert not tok.is_special(tok.stoi["e"])

    def test_decompose_missing_raises(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        with pytest.raises(ValueError):
            tok.decompose_token("zzz-not-in-vocab")

    def test_decompose_bad_id_raises(self):
        tok = SloBPE(); tok.train(CORPUS, vocab_size=64)
        with pytest.raises(ValueError):
            tok.decompose_token(999)

    def test_decompose_merged_subword(self):
        tok = SloBPE(pretokenizer="whitespace"); tok.train(["abc abc abc"], vocab_size=10)
        r = tok.decompose_token("ab")
        assert r["type"] == "merged_subword"
        assert r["merge_path"]
        assert r["depth"] > 0
        assert r["base_chars"] == ["a", "b"]

    def test_decompose_token_no_merge_record(self):
        tok = SloBPE()
        tok.vocab = ["a", "b", "ab"]
        tok.stoi = {"a": 0, "b": 1, "ab": 2}
        tok.itos = {0: "a", 1: "b", 2: "ab"}
        tok._special_set = set(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        r = tok.decompose_token("ab")
        assert r["type"] == "merged_subword"
        assert r["base_chars"] == ["a", "b"]

    def test_train_from_directory(self, tmp_path):
        d = tmp_path / "corpus"; d.mkdir()
        (d / "a.txt").write_text("hello world", encoding="utf-8")
        sub = d / "sub"; sub.mkdir()
        (sub / "b.txt").write_text("hello there", encoding="utf-8")
        (d / "empty.txt").write_text("", encoding="utf-8")
        tok = SloBPE.train_from_directory(str(d), vocab_size=64)
        assert tok.vocab_size >= 13
        tok_nr = SloBPE.train_from_directory(str(d), vocab_size=64, recursive=False)
        assert tok_nr.vocab_size >= 10

    def test_train_from_directory_not_dir(self):
        with pytest.raises(ValueError):
            SloBPE.train_from_directory("/nonexistent/dir")

    def test_train_from_directory_no_files(self, tmp_path):
        d = tmp_path / "empty_dir"; d.mkdir()
        with pytest.raises(ValueError):
            SloBPE.train_from_directory(str(d))

    def test_train_from_directory_skips_unreadable(self, tmp_path):
        import os
        d = tmp_path / "corpus2"; d.mkdir()
        (d / "ok.txt").write_text("hello world", encoding="utf-8")
        bad = d / "bad.txt"; bad.write_text("secret", encoding="utf-8")
        os.chmod(bad, 0)
        try:
            tok = SloBPE.train_from_directory(str(d), vocab_size=32)
            assert tok.vocab_size >= 5
        finally:
            os.chmod(bad, 0o644)


# ── Coverage: SloUnigram edge paths ──

class TestUnigramEdgePaths:

    def test_bos_eos_ids(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.bos_id == tok.stoi["<BOS>"]
        assert tok.eos_id == tok.stoi["<EOS>"]
        assert tok.pad_id == tok.stoi["<PAD>"]
        assert tok.unk_id == tok.stoi["<UNK>"]

    def test_train_empty_raises(self):
        with pytest.raises(ValueError):
            SloUnigram().train([])

    def test_train_verbose_prunes(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=16, verbose=True)
        assert tok.vocab_size < 269

    def test_train_reaches_target(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=128)
        assert tok.vocab_size <= 128

    def test_train_total_count_zero_breaks(self):
        tok = SloUnigram()
        tok.train(["   "], vocab_size=20)
        assert tok.vocab_size >= 4

    def test_encode_bos_eos(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_encode_with_scores(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        out = tok.encode_with_scores("hello world", nbest=4, alpha=0.5)
        assert out
        for ids, score in out:
            assert isinstance(ids, list)
            assert isinstance(score, float)

    def test_encode_with_scores_sorted(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        out = tok.encode_with_scores("hello", nbest=3)
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_viterbi_empty(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok._viterbi("") == []

    def test_viterbi_fallback_char_level(self):
        tok = SloUnigram(pretokenizer="whitespace"); tok.train(["abc"], vocab_size=64)
        ids = tok.encode("\U0001f600x")
        assert len(ids) == 2
        assert ids[0] == tok.unk_id

    def test_all_segmentations_empty(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok._all_segmentations("") == []

    def test_decode_out_of_range(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.decode([10 ** 9]) == "?"
        assert tok.decode([-1]) == "?"

    def test_decode_skips_special(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        assert tok.decode([tok.bos_id, tok.eos_id, tok.pad_id]) == ""

    def test_from_dict_missing_specials(self):
        data = {
            "vocab": ["a", "b"],
            "stoi": {"a": 0, "b": 1},
            "itos": {"0": "a", "1": "b"},
            "scores": {},
            "pretokenizer": "gpt2",
        }
        tok = SloUnigram.from_dict(data)
        for sp in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            assert sp in tok.stoi
        assert tok.vocab_size == 6

    def test_save_load_roundtrip(self, tmp_path):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        p = str(tmp_path / "unigram.json")
        tok.save(p)
        tok2 = SloUnigram.load(p)
        assert tok2.decode(tok2.encode("hello world")) == "hello world"

    def test_decompose_missing_raises(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        with pytest.raises(ValueError):
            tok.decompose_token("zzz-not-in-vocab")

    def test_decompose_bad_id_raises(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        with pytest.raises(ValueError):
            tok.decompose_token(999)

    def test_decompose_subword(self):
        tok = SloUnigram(pretokenizer="whitespace"); tok.train(["abc abc abc"], vocab_size=130)
        sub = [t for t in tok.vocab if len(t) > 1 and t not in tok.SPECIAL_TOKENS]
        assert sub
        r = tok.decompose_token(sub[0])
        assert r["type"] == "subword"
        assert "score" in r

    def test_train_from_directory(self, tmp_path):
        d = tmp_path / "uni"; d.mkdir()
        (d / "a.txt").write_text("hello world", encoding="utf-8")
        sub = d / "sub"; sub.mkdir()
        (sub / "b.txt").write_text("hello there", encoding="utf-8")
        tok = SloUnigram.train_from_directory(str(d), vocab_size=64,
                                               seed_max_len=4, pruning_ratio=0.3, em_iters=2)
        assert tok.vocab_size >= 4
        assert tok.decode(tok.encode("hello")) == "hello"
        tok2 = SloUnigram.train_from_directory(str(d), vocab_size=32, recursive=False)
        assert tok2.vocab_size >= 4

    def test_show_vocab(self):
        tok = SloUnigram(); tok.train(CORPUS, vocab_size=64)
        tok.show_vocab(top_n=5)

    def test_pretokenize_whitespace(self):
        tok = SloUnigram(pretokenizer="whitespace"); tok.train(CORPUS, vocab_size=64)
        assert tok.encode("hello world") == tok.encode("hello") + tok.encode("world")
