"""Meaningful tests for CharTokenizer and BPETokenizer — vocab building, encode/decode, save/load."""

import pytest
from pathlib import Path
from domains.multimodal.char_tokenizer import CharTokenizer
from domains.multimodal.bpe_tokenizer import BPETokenizer


# ── CharTokenizer ──────────────────────────────────────────────────────

class TestCharTokenizer:
    def test_build_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        assert tok.vocab_size > 0
        assert tok._built is True

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        ids = tok.encode("hello")
        text = tok.decode(ids)
        assert text == "hello"

    def test_encode_adds_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("a")
        assert ids[0] == 0  # BOS
        assert ids[-1] == 1  # EOS
        assert len(ids) == 3  # BOS + a + EOS

    def test_encode_unknown_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("xyz")
        # x, y, z should map to UNK (3) since they're not in training text
        # but they ARE in ASCII fallback, so they should have valid IDs
        assert len(ids) == 5  # BOS + 3 chars + EOS

    def test_encode_empty(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("")
        assert ids == [0, 1]  # BOS + EOS

    def test_decode_strips_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("abc")
        # Should not contain BOS/EOS in output
        decoded = tok.decode(ids)
        assert decoded == "abc"

    def test_vocab_includes_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        assert "<BOS>" in tok.vocab
        assert "<EOS>" in tok.vocab
        assert "<PAD>" in tok.vocab
        assert "<UNK>" in tok.vocab

    def test_pad_to(self):
        tok = CharTokenizer(pad_to=32)
        assert tok.pad_to == 32

    def test_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        path = str(tmp_path / "tokenizer.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2._built is True
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("hello") == tok.encode("hello")

    def test_load_nonexistent(self):
        tok = CharTokenizer()
        assert tok.load("/nonexistent/path.json") is False

    def test_encode_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="not trained"):
            tok.encode("hello")

    def test_vocab_size_includes_special_and_training_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["abcdef"])
        # Should have 4 special tokens + printable ASCII chars
        assert tok.vocab_size >= 4 + 10


# ── BPETokenizer ───────────────────────────────────────────────────────

class TestBPETokenizer:
    def test_train(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["hello world", "hello there"])
        assert tok._built is True
        assert len(tok.merges) > 0

    def test_encode_decode_roundtrip(self):
        tok = BPETokenizer(vocab_size=200)
        tok.train(["the cat sat on the mat"] * 10)
        ids = tok.encode("the cat")
        text = tok.decode(ids)
        assert "cat" in text
        assert "the" in text

    def test_encode_empty(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["hello world"])
        ids = tok.encode("")
        assert ids == []

    def test_encode_before_train_raises(self):
        tok = BPETokenizer()
        with pytest.raises(RuntimeError, match="not trained"):
            tok.encode("hello")

    def test_vocab_includes_special_tokens(self):
        tok = BPETokenizer()
        tok.train(["test"])
        assert "<BOS>" in tok.vocab
        assert "<EOS>" in tok.vocab
        assert "<UNK>" in tok.vocab

    def test_merges_learned(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["ab ab ab ab ab ab ab ab"])
        assert len(tok.merges) > 0

    def test_save_load_roundtrip(self, tmp_path):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["hello world hello world"])
        path = str(tmp_path / "bpe.json")
        tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        assert tok2._built is True
        assert tok2.vocab_size == tok.vocab_size

    def test_load_nonexistent(self):
        tok = BPETokenizer()
        assert tok.load("/nonexistent/path.json") is False

    def test_preprocess(self):
        tok = BPETokenizer()
        tokens = tok._preprocess("hello")
        assert len(tokens) > 0
        assert tokens[-1].endswith("</w>")

    def test_vocab_size_cap(self):
        tok = BPETokenizer(vocab_size=20)
        tok.train(["a b c d e f g"])
        assert len(tok.vocab) <= 20 + len(tok.special_tokens)

    def test_decode_cleans_word_boundaries(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["hello world"] * 5)
        ids = tok.encode("hello world")
        decoded = tok.decode(ids)
        assert "</w>" not in decoded
