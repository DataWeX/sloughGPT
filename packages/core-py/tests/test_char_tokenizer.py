"""Tests for CharTokenizer — character-level tokenization, vocab, encode/decode, save/load.

Covers:
  - build_vocab from texts
  - encode produces BOS + char_ids + EOS
  - decode strips special tokens
  - Unknown characters map to <UNK>
  - vocab_size property
  - save/load round-trip
  - pad_to parameter
"""

import json
import pytest
from domains.multimodal.char_tokenizer import CharTokenizer


class TestCharTokenizer:
    def test_init(self):
        tok = CharTokenizer()
        assert tok.vocab_size == 0
        assert tok._built is False

    def test_build_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello", "world"])
        assert tok._built is True
        assert tok.vocab_size > 4  # special tokens + chars

    def test_special_tokens_first(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        assert tok.vocab["<BOS>"] == 0
        assert tok.vocab["<EOS>"] == 1
        assert tok.vocab["<PAD>"] == 2
        assert tok.vocab["<UNK>"] == 3

    def test_encode_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        ids = tok.encode("hi")
        assert ids[0] == 0  # BOS
        assert ids[-1] == 1  # EOS
        assert len(ids) == 4  # BOS + h + i + EOS

    def test_encode_single_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["a"])
        ids = tok.encode("a")
        assert ids[0] == 0  # BOS
        assert ids[-1] == 1  # EOS
        assert len(ids) == 3  # BOS + a + EOS

    def test_encode_empty(self):
        tok = CharTokenizer()
        tok.build_vocab(["x"])
        ids = tok.encode("")
        assert ids == [0, 1]  # BOS + EOS only

    def test_decode_strips_special(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        ids = tok.encode("hi")
        decoded = tok.decode(ids)
        assert decoded == "hi"

    def test_decode_strips_unk(self):
        tok = CharTokenizer()
        tok.build_vocab(["ab"])
        # 'z' not in training text but in ASCII fallback
        ids = tok.encode("z")
        decoded = tok.decode(ids)
        assert decoded == "z"

    def test_encode_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="not trained"):
            tok.encode("hello")

    def test_decode_before_build(self):
        tok = CharTokenizer()
        # decode doesn't require build_vocab (uses itos which is empty)
        result = tok.decode([0, 1])
        assert result == ""

    def test_vocab_includes_ascii(self):
        tok = CharTokenizer()
        tok.build_vocab([""])
        # All printable ASCII should be in vocab
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert ch in tok.vocab

    def test_pad_to(self):
        tok = CharTokenizer(pad_to=32)
        assert tok.pad_to == 32

    def test_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer(pad_to=16)
        tok.build_vocab(["hello world"])
        path = str(tmp_path / "tok.json")
        tok.save(path)

        tok2 = CharTokenizer()
        assert tok2.load(path) is True
        assert tok2.vocab == tok.vocab
        assert tok2.itos == tok.itos
        assert tok2.pad_to == 16
        assert tok2._built is True

    def test_load_nonexistent(self):
        tok = CharTokenizer()
        assert tok.load("/nonexistent/path.json") is False

    def test_save_creates_parent_dirs(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        path = str(tmp_path / "subdir" / "tok.json")
        tok.save(path)
        assert (tmp_path / "subdir" / "tok.json").exists()

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["the quick brown fox"])
        text = "hello"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_encode_decode_unicode(self):
        tok = CharTokenizer()
        tok.build_vocab(["café"])
        text = "café"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_multiple_encode_calls(self):
        tok = CharTokenizer()
        tok.build_vocab(["ab", "cd"])
        ids1 = tok.encode("ab")
        ids2 = tok.encode("cd")
        assert ids1 != ids2
        assert ids1[0] == ids2[0] == 0  # both start with BOS
        assert ids1[-1] == ids2[-1] == 1  # both end with EOS
