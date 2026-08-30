"""Tests for tokenizer.py — pure logic, no mocks."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from domains.training.tokenizer import (
    SloBPE,
    SloUnigram,
    gpt2_pretokenize,
    default_pretokenize,
)


# ── gpt2_pretokenize ──────────────────────────────────────────────


class TestGPT2Pretokenize:
    def test_simple_words(self):
        result = gpt2_pretokenize("hello world")
        assert "hello" in result
        assert " world" in result

    def test_leading_space(self):
        result = gpt2_pretokenize("hello world")
        assert any(" " in t for t in result)

    def test_contraction(self):
        result = gpt2_pretokenize("it's a dog")
        tokens = gpt2_pretokenize("it's a dog")
        assert "'s" in tokens

    def test_digits(self):
        result = gpt2_pretokenize("42 numbers")
        assert any("42" in t for t in result)

    def test_punctuation(self):
        result = gpt2_pretokenize("hello!")
        assert "!" in result or "hello!" in result


# ── default_pretokenize ───────────────────────────────────────────


class TestDefaultPretokenize:
    def test_whitespace_split(self):
        assert default_pretokenize("hello world") == ["hello", "world"]

    def test_empty_string(self):
        assert default_pretokenize("") == []

    def test_multiple_spaces(self):
        assert default_pretokenize("a  b") == ["a", "b"]


# ── SloBPE ────────────────────────────────────────────────────────


class TestSloBPEInit:
    def test_default_vocab_size(self):
        tok = SloBPE()
        assert tok.vocab_size == 0

    def test_pad_id(self):
        tok = SloBPE()
        assert tok.pad_id == 0

    def test_unk_id(self):
        tok = SloBPE()
        assert tok.unk_id == 1

    def test_bos_id(self):
        tok = SloBPE()
        assert tok.bos_id == 2

    def test_eos_id(self):
        tok = SloBPE()
        assert tok.eos_id == 3


class TestSloBPETraining:
    def test_train_basic(self):
        tok = SloBPE()
        # Use enough variety so BPE can reach 64 vocab
        words = [chr(i) for i in range(ord('a'), ord('z')+1)] + [str(i) for i in range(10)]
        texts = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
        tok.train(texts, vocab_size=64)
        assert tok.vocab_size >= 40  # BPE may stop early if pairs run out
        assert len(tok.merges) > 0

    def test_train_empty_raises(self):
        tok = SloBPE()
        with pytest.raises(ValueError, match="at least one text"):
            tok.train([])

    def test_train_builds_stoi_itos(self):
        tok = SloBPE()
        tok.train(["ab ab", "cd cd"], vocab_size=32)
        assert len(tok.stoi) == tok.vocab_size
        assert len(tok.itos) == tok.vocab_size
        for i in range(tok.vocab_size):
            assert tok.itos[i] in tok.stoi
            assert tok.stoi[tok.itos[i]] == i

    def test_train_special_tokens_present(self):
        tok = SloBPE()
        tok.train(["hello world"], vocab_size=16)
        for sp in SloBPE.SPECIAL_TOKENS:
            assert sp in tok.stoi

    def test_train_returns_self(self):
        tok = SloBPE()
        result = tok.train(["hello world"], vocab_size=16)
        assert result is tok

    def test_train_min_frequency(self):
        tok = SloBPE()
        tok.train(["ab ab", "cd cd"], vocab_size=32, min_frequency=10)
        # With high min_frequency, fewer merges
        tok2 = SloBPE()
        tok2.train(["ab ab", "cd cd"], vocab_size=32, min_frequency=1)
        assert len(tok.merges) <= len(tok2.merges)


class TestSloBPEEncode:
    @pytest.fixture
    def trained_tok(self):
        tok = SloBPE()
        texts = ["hello world", "hello there", "world of code"] * 10
        tok.train(texts, vocab_size=128)
        return tok

    def test_encode_returns_list(self, trained_tok):
        ids = trained_tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self, trained_tok):
        ids = trained_tok.encode("")
        assert ids == []

    def test_encode_with_bos(self, trained_tok):
        ids = trained_tok.encode("hello", add_bos=True)
        assert ids[0] == trained_tok.bos_id

    def test_encode_with_eos(self, trained_tok):
        ids = trained_tok.encode("hello", add_eos=True)
        assert ids[-1] == trained_tok.eos_id

    def test_encode_bos_eos(self, trained_tok):
        ids = trained_tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == trained_tok.bos_id
        assert ids[-1] == trained_tok.eos_id

    def test_encode_deterministic(self, trained_tok):
        assert trained_tok.encode("hello") == trained_tok.encode("hello")

    def test_encode_unknown_chars(self, trained_tok):
        ids = trained_tok.encode("zzz")
        assert len(ids) > 0


class TestSloBPEEncodeBatch:
    @pytest.fixture
    def trained_tok(self):
        tok = SloBPE()
        tok.train(["hello world", "foo bar"] * 10, vocab_size=64)
        return tok

    def test_batch_encoding(self, trained_tok):
        batch = trained_tok.encode_batch(["hello", "world"])
        assert len(batch) == 2
        assert isinstance(batch[0], list)

    def test_batch_with_max_length(self, trained_tok):
        batch = trained_tok.encode_batch(["hello world", "hi"], max_length=5)
        for ids in batch:
            assert len(ids) <= 5

    def test_batch_with_pad(self, trained_tok):
        batch = trained_tok.encode_batch(["hello world", "hi"], max_length=10, pad=True)
        assert all(len(ids) == 10 for ids in batch)
        # Padded tokens should be pad_id
        for ids in batch:
            assert ids[-1] == trained_tok.pad_id

    def test_batch_pad_without_max_length(self, trained_tok):
        batch = trained_tok.encode_batch(["hello", "world", "hi"], pad=True)
        max_len = max(len(ids) for ids in batch)
        assert all(len(ids) == max_len for ids in batch)


class TestSloBPEDecode:
    @pytest.fixture
    def trained_tok(self):
        tok = SloBPE()
        tok.train(["hello world", "foo bar"] * 10, vocab_size=64)
        return tok

    def test_decode_reconstruction(self, trained_tok):
        text = "hello"
        ids = trained_tok.encode(text)
        decoded = trained_tok.decode(ids)
        assert "hello" in decoded

    def test_decode_skip_special(self, trained_tok):
        ids = [trained_tok.bos_id, trained_tok.pad_id, trained_tok.eos_id]
        decoded = trained_tok.decode(ids, skip_special=True)
        assert decoded == ""

    def test_decode_no_skip_special(self, trained_tok):
        ids = [trained_tok.bos_id]
        decoded = trained_tok.decode(ids, skip_special=False)
        assert decoded != ""

    def test_decode_out_of_range(self, trained_tok):
        ids = [9999]
        decoded = trained_tok.decode(ids)
        assert "?" in decoded

    def test_decode_negative_id(self, trained_tok):
        ids = [-1]
        decoded = trained_tok.decode(ids)
        assert "?" in decoded


class TestSloBPESerialization:
    def test_to_dict_roundtrip(self):
        tok = SloBPE()
        tok.train(["hello world", "foo bar"] * 5, vocab_size=32)
        d = tok.to_dict()
        tok2 = SloBPE.from_dict(d)
        assert tok2.vocab == tok.vocab
        assert tok2.merges == tok.merges
        assert tok2.stoi == tok.stoi
        assert tok2.vocab_size == tok.vocab_size

    def test_save_load_roundtrip(self, tmp_path):
        tok = SloBPE()
        tok.train(["hello world"] * 5, vocab_size=32)
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = SloBPE.load(path)
        assert tok2.vocab == tok.vocab
        assert tok2.merges == tok.merges

    def test_to_dict_version(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        d = tok.to_dict()
        assert d["version"] == 2

    def test_from_dict_backward_compat(self):
        data = {
            "version": 1,
            "vocab": ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "</w>", "a", "b"],
            "merges": [],
            "stoi": {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "</w>": 4, "a": 5, "b": 6},
            "itos": {"0": "<PAD>", "1": "<UNK>", "2": "<BOS>", "3": "<EOS>", "4": "</w>", "5": "a", "6": "b"},
        }
        tok = SloBPE.from_dict(data)
        assert tok._pretokenizer == "whitespace"


class TestSloBPEVocabStats:
    def test_vocab_stats(self):
        tok = SloBPE()
        texts = ["hello world", "foo bar baz"] * 20
        tok.train(texts, vocab_size=32)
        stats = tok.vocab_stats()
        assert stats["vocab_size"] >= 32
        assert stats["base_chars"] > 0
        assert stats["special_tokens"] == 4
        assert stats["total_merges_learned"] == len(tok.merges)


class TestSloBPEDecomposeToken:
    def test_decompose_merged_token(self):
        tok = SloBPE()
        tok.train(["hello world", "hello there"] * 10, vocab_size=128)
        # Find a token created by a merge (not special tokens or </w>)
        merged = [
            t for t in tok.vocab
            if len(t) > 1 and t not in SloBPE.SPECIAL_TOKENS and t != SloBPE.WORD_SUFFIX
            and t in {l + r for l, r in tok.merges}
        ]
        if merged:
            result = tok.decompose_token(merged[0])
            assert result["token"] == merged[0]
            assert result["depth"] >= 1
            assert len(result["base_chars"]) > 0

    def test_decompose_special_token(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        result = tok.decompose_token("<PAD>")
        assert result["type"] == "special"
        assert result["depth"] == 0

    def test_decompose_base_char(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        result = tok.decompose_token("h")
        assert result["type"] == "base_char"
        assert result["depth"] == 0

    def test_decompose_unknown_raises(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        with pytest.raises(ValueError, match="not found"):
            tok.decompose_token("NONEXISTENT")


class TestSloBPEAnalyzeCorpus:
    def test_analyze_corpus(self):
        tok = SloBPE()
        tok.train(["hello world", "foo bar"] * 5, vocab_size=64)
        stats = tok.analyze_corpus(["hello world", "hello there"])
        assert stats["total_chars"] > 0
        assert stats["total_tokens"] > 0
        assert stats["compression_ratio"] > 0
        assert stats["unique_tokens"] > 0
        assert stats["vocab_utilization"] > 0
        assert len(stats["top_tokens"]) > 0


class TestSloBPEAddSpecialTokens:
    def test_add_new_special_token(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        added = tok.add_special_tokens(["<|user|>", "<|assistant|>"])
        assert added == 2
        assert tok.is_special("<|user|>")
        assert tok.is_special("<|assistant|>")

    def test_add_existing_noop(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        added = tok.add_special_tokens(["<PAD>"])
        assert added == 0

    def test_special_ids(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        ids = tok.special_ids
        assert len(ids) == len(SloBPE.SPECIAL_TOKENS)


class TestSloBPEIsSpecial:
    def test_is_special_string(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        assert tok.is_special("<PAD>") is True
        assert tok.is_special("hello") is False

    def test_is_special_by_id(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=16)
        assert tok.is_special(0) is True  # PAD ID


class TestSloBPENormalize:
    def test_lowercasing(self):
        result = SloBPE._normalize("Hello WORLD")
        assert result == "hello world"

    def test_no_lowercasing(self):
        result = SloBPE._normalize("Hello WORLD", lowercase=False)
        assert result == "Hello WORLD"

    def test_whitespace_collapse(self):
        result = SloBPE._normalize("hello    world")
        assert result == "hello world"

    def test_stripping(self):
        result = SloBPE._normalize("  hello  ")
        assert result == "hello"


class TestSloBPEPretokenize:
    def test_gpt2_mode(self):
        tok = SloBPE(pretokenizer="gpt2")
        result = tok._pretokenize("hello world")
        assert "hello" in result

    def test_whitespace_mode(self):
        tok = SloBPE(pretokenizer="whitespace")
        result = tok._pretokenize("hello world")
        assert result == ["hello", "world"]


class TestSloBPEShowPretokenization:
    def test_show_pretokenization(self):
        tok = SloBPE()
        result = tok.show_pretokenization("hello world")
        assert "pretokens" in result
        assert "segments" in result
        assert "count" in result
        assert result["count"] == len(result["pretokens"])
        for seg in result["segments"]:
            assert "text" in seg
            assert "char_count" in seg
            assert "pct" in seg


# ── SloUnigram ────────────────────────────────────────────────────


class TestSloUnigramInit:
    def test_default_vocab_size(self):
        tok = SloUnigram()
        assert tok.vocab_size == 0

    def test_special_token_ids(self):
        tok = SloUnigram()
        assert tok.pad_id == 0
        assert tok.bos_id == 0
        assert tok.eos_id == 0


class TestSloUnigramTraining:
    def test_train_basic(self):
        tok = SloUnigram()
        texts = ["hello world", "hello there", "foo bar baz"] * 20
        tok.train(texts, vocab_size=64)
        assert tok.vocab_size >= 64
        assert len(tok._scores) > 0

    def test_train_empty_raises(self):
        tok = SloUnigram()
        with pytest.raises(ValueError, match="at least one text"):
            tok.train([])

    def test_train_returns_self(self):
        tok = SloUnigram()
        result = tok.train(["hello world"], vocab_size=32)
        assert result is tok

    def test_train_builds_stoi_itos(self):
        tok = SloUnigram()
        tok.train(["hello world"], vocab_size=32)
        assert len(tok.stoi) == tok.vocab_size
        assert len(tok.itos) == tok.vocab_size

    def test_train_special_tokens_present(self):
        tok = SloUnigram()
        tok.train(["hello world"], vocab_size=32)
        for sp in SloUnigram.SPECIAL_TOKENS:
            assert sp in tok.stoi


class TestSloUnigramEncode:
    @pytest.fixture
    def trained_tok(self):
        tok = SloUnigram()
        texts = ["hello world", "hello there", "world of code"] * 10
        tok.train(texts, vocab_size=128)
        return tok

    def test_encode_returns_list(self, trained_tok):
        ids = trained_tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self, trained_tok):
        ids = trained_tok.encode("")
        assert ids == []

    def test_encode_with_bos(self, trained_tok):
        ids = trained_tok.encode("hello", add_bos=True)
        assert ids[0] == trained_tok.bos_id

    def test_encode_with_eos(self, trained_tok):
        ids = trained_tok.encode("hello", add_eos=True)
        assert ids[-1] == trained_tok.eos_id

    def test_encode_deterministic(self, trained_tok):
        assert trained_tok.encode("hello") == trained_tok.encode("hello")


class TestSloUnigramEncodeWithScores:
    def test_encode_with_scores(self):
        tok = SloUnigram()
        tok.train(["hello world", "hello there"] * 5, vocab_size=64)
        results = tok.encode_with_scores("hello world", nbest=4)
        assert isinstance(results, list)
        assert len(results) > 0
        for ids, score in results:
            assert isinstance(ids, list)
            assert isinstance(score, float)


class TestSloUnigramDecode:
    @pytest.fixture
    def trained_tok(self):
        tok = SloUnigram()
        tok.train(["hello world", "foo bar"] * 10, vocab_size=128)
        return tok

    def test_decode_basic(self, trained_tok):
        ids = trained_tok.encode("hello")
        decoded = trained_tok.decode(ids)
        assert "hello" in decoded

    def test_decode_skip_special(self, trained_tok):
        ids = [trained_tok.bos_id, trained_tok.eos_id]
        decoded = trained_tok.decode(ids, skip_special=True)
        assert decoded == ""

    def test_decode_no_skip_special(self, trained_tok):
        ids = [trained_tok.bos_id]
        decoded = trained_tok.decode(ids, skip_special=False)
        assert decoded != ""

    def test_decode_out_of_range(self, trained_tok):
        ids = [9999]
        decoded = trained_tok.decode(ids)
        assert "?" in decoded


class TestSloUnigramSerialization:
    def test_to_dict_roundtrip(self):
        tok = SloUnigram()
        tok.train(["hello world"] * 5, vocab_size=32)
        d = tok.to_dict()
        tok2 = SloUnigram.from_dict(d)
        assert tok2.vocab == tok.vocab
        assert tok2.vocab_size == tok.vocab_size
        assert tok2._pretokenizer == tok._pretokenizer

    def test_save_load_roundtrip(self, tmp_path):
        tok = SloUnigram()
        tok.train(["hello world"] * 5, vocab_size=32)
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = SloUnigram.load(path)
        assert tok2.vocab == tok.vocab
        assert len(tok2._scores) == len(tok._scores)

    def test_to_dict_type(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        d = tok.to_dict()
        assert d["type"] == "unigram"
        assert d["version"] == 1


class TestSloUnigramVocabStats:
    def test_vocab_stats(self):
        tok = SloUnigram()
        texts = ["hello world", "foo bar baz"] * 20
        tok.train(texts, vocab_size=32)
        stats = tok.vocab_stats()
        assert stats["vocab_size"] >= 32
        assert stats["type"] == "unigram"
        assert stats["special_tokens"] == 4


class TestSloUnigramDecomposeToken:
    def test_decompose_base_char(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        result = tok.decompose_token("h")
        assert result["type"] == "base_char"
        assert result["depth"] == 0

    def test_decompose_special(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        result = tok.decompose_token("<PAD>")
        assert result["type"] == "special"

    def test_decompose_subword(self):
        tok = SloUnigram()
        tok.train(["hello world"] * 5, vocab_size=128)
        subwords = [t for t in tok.vocab if len(t) > 1 and t not in SloUnigram.SPECIAL_TOKENS]
        if subwords:
            result = tok.decompose_token(subwords[0])
            assert result["type"] == "subword"
            assert "score" in result

    def test_decompose_unknown_raises(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        with pytest.raises(ValueError, match="not found"):
            tok.decompose_token("NONEXISTENT")


class TestSloUnigramAnalyzeCorpus:
    def test_analyze_corpus(self):
        tok = SloUnigram()
        tok.train(["hello world", "foo bar"] * 5, vocab_size=64)
        stats = tok.analyze_corpus(["hello world", "hello there"])
        assert stats["total_chars"] > 0
        assert stats["total_tokens"] > 0
        assert stats["compression_ratio"] > 0
        assert stats["unique_tokens"] > 0
        assert stats["vocab_utilization"] > 0
        assert len(stats["top_tokens"]) > 0


class TestSloUnigramShowPretokenization:
    def test_show(self):
        tok = SloUnigram()
        result = tok.show_pretokenization("hello world")
        assert "pretokens" in result
        assert result["count"] == len(result["pretokens"])


class TestSloUnigramNormalize:
    def test_lowercasing(self):
        result = SloUnigram._normalize("Hello WORLD")
        assert result == "hello world"

    def test_whitespace_collapse(self):
        result = SloUnigram._normalize("a   b")
        assert result == "a b"


class TestSloUnigramViterbi:
    def test_viterbi_returns_tokens(self):
        tok = SloUnigram()
        tok.train(["hello world"] * 5, vocab_size=128)
        tokens = tok._viterbi("hello")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_viterbi_empty_string(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        assert tok._viterbi("") == []


class TestSloUnigramAllSegmentations:
    def test_all_segmentations(self):
        tok = SloUnigram()
        tok.train(["hello world"] * 5, vocab_size=128)
        segs = tok._all_segmentations("hello")
        assert isinstance(segs, list)
        assert len(segs) > 0
        for seg in segs:
            assert "".join(seg) == "hello"

    def test_empty_string(self):
        tok = SloUnigram()
        tok.train(["hello"], vocab_size=16)
        assert tok._all_segmentations("") == []
