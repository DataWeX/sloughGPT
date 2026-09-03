"""Tests for CharTokenizer — character-level tokenizer."""
from __future__ import annotations

import json

from domains.multimodal.char_tokenizer import CharTokenizer


class TestCharTokenizer:
    def test_build_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        assert tok.vocab_size > 4  # at least special tokens + chars

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("abc")
        text = tok.decode(ids)
        assert text == "abc"

    def test_encode_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["x"])
        ids = tok.encode("x")
        assert ids[0] == 0  # BOS
        assert ids[-1] == 1  # EOS

    def test_unknown_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["ab"])
        ids = tok.encode("z")  # z not in training text but in ASCII fallback
        text = tok.decode(ids)
        # z is in ASCII fallback, so it should be decoded
        assert text == "z"

    def test_special_tokens_stripped(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("a")
        decoded = tok.decode(ids)
        assert "<BOS>" not in decoded
        assert "<EOS>" not in decoded

    def test_save_load(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode("hi") == tok.encode("hi")

    def test_load_nonexistent(self):
        tok = CharTokenizer()
        assert tok.load("/nonexistent/path.json") is False

    def test_encode_before_build_raises(self):
        tok = CharTokenizer()
        try:
            tok.encode("test")
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_vocab_includes_ascii(self):
        tok = CharTokenizer()
        tok.build_vocab([])
        assert " " in tok.vocab
        assert "a" in tok.vocab
        assert "z" in tok.vocab
