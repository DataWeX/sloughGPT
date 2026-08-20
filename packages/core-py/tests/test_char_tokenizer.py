"""Tests for domains.multimodal.char_tokenizer — CharTokenizer.

Covers: vocab building, encode/decode round-trip, special tokens, save/load,
pad_to, unknown character handling, vocab_size property.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.multimodal.char_tokenizer import CharTokenizer


class TestCharTokenizer:
    def test_build_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello", "world"])
        assert tok.vocab_size > 0
        assert tok._built is True

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        ids = tok.encode("hello")
        text = tok.decode(ids)
        assert text == "hello"

    def test_encode_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("ab")
        assert ids[0] == tok.vocab["<BOS>"]
        assert ids[-1] == tok.vocab["<EOS>"]
        assert len(ids) == 4  # BOS + a + b + EOS

    def test_encode_without_build(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError):
            tok.encode("test")

    def test_decode_strips_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [tok.vocab["<BOS>"], tok.vocab["a"], tok.vocab["b"], tok.vocab["<EOS>"]]
        assert tok.decode(ids) == "ab"

    def test_unknown_character(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("xyz")
        # x, y, z might be in ASCII fallback or mapped to UNK
        assert len(ids) > 2

    def test_vocab_size(self):
        tok = CharTokenizer()
        tok.build_vocab(["a"])
        assert tok.vocab_size > 4  # 4 special + at least 'a' + ASCII

    def test_save_and_load(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["test data"])
        path = str(tmp_path / "tokenizer.json")
        tok.save(path)

        tok2 = CharTokenizer()
        assert tok2.load(path) is True
        assert tok2.vocab_size == tok.vocab_size

    def test_load_nonexistent(self):
        tok = CharTokenizer()
        assert tok.load("/nonexistent/path.json") is False

    def test_pad_to(self):
        tok = CharTokenizer(pad_to=20)
        assert tok.pad_to == 20

    def test_ensure_ascii(self):
        tok = CharTokenizer()
        ascii_chars = tok._ensure_ascii()
        assert "a" in ascii_chars
        assert "z" in ascii_chars
        assert "A" in ascii_chars

    def test_encode_empty(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("")
        assert ids == [tok.vocab["<BOS>"], tok.vocab["<EOS>"]]

    def test_special_tokens(self):
        assert CharTokenizer.SPECIAL_TOKENS == ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]
