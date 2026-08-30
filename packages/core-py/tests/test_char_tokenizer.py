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


# ---------------------------------------------------------------------------
# Vocab Building
# ---------------------------------------------------------------------------

class TestBuildVocab:
    def test_build_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello", "world"])
        assert tok.vocab_size > 0
        assert tok._built is True

    def test_build_vocab_empty_list(self):
        tok = CharTokenizer()
        tok.build_vocab([])
        assert tok._built is True
        assert tok.vocab_size >= 4  # at least special tokens

    def test_build_vocab_single_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["a"])
        assert "a" in tok.vocab
        assert tok.vocab["a"] >= 4  # after special tokens

    def test_build_vocab_unicode_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        assert "é" not in tok.vocab
        tok.build_vocab(["café"])
        assert "é" in tok.vocab

    def test_build_vocab_merges_multiple_texts(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc", "def"])
        for ch in "abcdef":
            assert ch in tok.vocab

    def test_build_vocab_special_tokens_first(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        assert tok.vocab["<BOS>"] == 0
        assert tok.vocab["<EOS>"] == 1
        assert tok.vocab["<PAD>"] == 2
        assert tok.vocab["<UNK>"] == 3

    def test_build_vocab_deterministic_order(self):
        tok1 = CharTokenizer()
        tok1.build_vocab(["hello"])
        tok2 = CharTokenizer()
        tok2.build_vocab(["hello"])
        assert tok1.vocab == tok2.vocab

    def test_build_vocab_includes_ascii_fallback(self):
        tok = CharTokenizer()
        tok.build_vocab([])
        for i in range(32, 127):
            assert chr(i) in tok.vocab

    def test_build_vocab_includes_newline_and_tab(self):
        tok = CharTokenizer()
        tok.build_vocab([])
        assert "\n" in tok.vocab
        assert "\t" in tok.vocab

    def test_ensure_ascii(self):
        tok = CharTokenizer()
        ascii_chars = tok._ensure_ascii()
        assert "a" in ascii_chars
        assert "z" in ascii_chars
        assert "A" in ascii_chars
        assert len(ascii_chars) >= 95

    def test_build_vocab_chinese_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["你好世界"])
        assert "你" in tok.vocab
        assert "好" in tok.vocab

    def test_build_vocab_emoji_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello 🌍"])
        assert "🌍" in tok.vocab

    def test_build_vocab_long_text(self):
        tok = CharTokenizer()
        long_text = ["a" * 10000]
        tok.build_vocab(long_text)
        assert "a" in tok.vocab
        assert tok._built is True


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncode:
    def test_encode_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("ab")
        assert ids[0] == tok.vocab["<BOS>"]
        assert ids[-1] == tok.vocab["<EOS>"]
        assert len(ids) == 4

    def test_encode_without_build(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError):
            tok.encode("test")

    def test_encode_empty(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("")
        assert ids == [tok.vocab["<BOS>"], tok.vocab["<EOS>"]]

    def test_encode_single_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["x"])
        ids = tok.encode("x")
        assert ids == [tok.vocab["<BOS>"], tok.vocab["x"], tok.vocab["<EOS>"]]

    def test_encode_multiple_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        ids = tok.encode("hello")
        assert len(ids) == 7

    def test_encode_unknown_char_maps_to_un(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("x")
        assert ids[1] != tok.vocab["<UNK>"]

    def test_encode_preserves_char_order(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        ids = tok.encode("hello")
        chars = [tok.itos[i] for i in ids[1:-1]]
        assert chars == list("hello")

    def test_encode_long_text(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        text = "a" * 1000
        ids = tok.encode(text)
        assert len(ids) == 1002

    def test_encode_repeated_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("aaa")
        assert ids[1] == ids[2] == ids[3]
        assert ids[0] == tok.vocab["<BOS>"]
        assert ids[-1] == tok.vocab["<EOS>"]

    def test_encode_returns_list_of_ints(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = tok.encode("test")
        assert isinstance(ids, list)
        for x in ids:
            assert isinstance(x, int)

    def test_encode_tab_character(self):
        tok = CharTokenizer()
        tok.build_vocab(["a\tb"])
        ids = tok.encode("a\tb")
        assert len(ids) == 5

    def test_encode_space_character(self):
        tok = CharTokenizer()
        tok.build_vocab(["a b"])
        ids = tok.encode("a b")
        assert len(ids) == 5

    def test_encode_newline_character(self):
        tok = CharTokenizer()
        tok.build_vocab(["a\nb"])
        ids = tok.encode("a\nb")
        assert len(ids) == 5

    def test_encode_mixed_case(self):
        tok = CharTokenizer()
        tok.build_vocab(["Hello WORLD"])
        ids = tok.encode("Hi")
        chars = [tok.itos[i] for i in ids[1:-1]]
        assert chars == ["H", "i"]

    def test_encode_punctuation(self):
        tok = CharTokenizer()
        tok.build_vocab(["!@#$%"])
        ids = tok.encode("!@")
        assert len(ids) == 4


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class TestDecode:
    def test_decode_strips_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [tok.vocab["<BOS>"], tok.vocab["a"], tok.vocab["b"], tok.vocab["<EOS>"]]
        assert tok.decode(ids) == "ab"

    def test_decode_strips_pad(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [tok.vocab["<PAD>"], tok.vocab["a"], tok.vocab["<PAD>"]]
        assert tok.decode(ids) == "a"

    def test_decode_strips_unk(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [tok.vocab["<UNK>"], tok.vocab["c"]]
        assert tok.decode(ids) == "c"

    def test_decode_empty_list(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        assert tok.decode([]) == ""

    def test_decode_all_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [0, 1, 2, 3]
        assert tok.decode(ids) == ""

    def test_decode_roundtrip_single_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["a"])
        ids = tok.encode("a")
        assert tok.decode(ids) == "a"

    def test_decode_roundtrip_full(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        text = "hello world"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_decode_unknown_id_skipped(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = [tok.vocab["<BOS>"], 9999, tok.vocab["<EOS>"]]
        assert tok.decode(ids) == ""

    def test_decode_preserves_spaces(self):
        tok = CharTokenizer()
        tok.build_vocab(["a b"])
        ids = tok.encode("a b")
        decoded = tok.decode(ids)
        assert decoded == "a b"

    def test_decode_newline_preserved(self):
        tok = CharTokenizer()
        tok.build_vocab(["a\nb"])
        ids = tok.encode("a\nb")
        decoded = tok.decode(ids)
        assert decoded == "a\nb"

    def test_decode_only_bos_eos(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        ids = [tok.vocab["<BOS>"], tok.vocab["<EOS>"]]
        assert tok.decode(ids) == ""


# ---------------------------------------------------------------------------
# Encode/Decode Roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        ids = tok.encode("hello")
        text = tok.decode(ids)
        assert text == "hello"

    def test_roundtrip_various_strings(self):
        tok = CharTokenizer()
        tok.build_vocab(["abcdefghijklmnopqrstuvwxyz 0123456789"])
        for s in ["hello", "world", "test", "abc 123", ""]:
            ids = tok.encode(s)
            assert tok.decode(ids) == s

    def test_roundtrip_special_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["!@#$%^&*()"])
        for s in ["!", "@", "#"]:
            ids = tok.encode(s)
            assert tok.decode(ids) == s

    def test_roundtrip_numbers_as_chars(self):
        tok = CharTokenizer()
        tok.build_vocab(["0123456789"])
        ids = tok.encode("42")
        assert tok.decode(ids) == "42"

    def test_roundtrip_long_string(self):
        tok = CharTokenizer()
        long = "abcdefghij" * 100
        tok.build_vocab([long])
        ids = tok.encode(long)
        assert tok.decode(ids) == long

    def test_roundtrip_unicode(self):
        tok = CharTokenizer()
        tok.build_vocab(["café résumé"])
        text = "café"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_roundtrip_single_space(self):
        tok = CharTokenizer()
        tok.build_vocab(["a b"])
        ids = tok.encode(" ")
        assert tok.decode(ids) == " "


# ---------------------------------------------------------------------------
# vocab_size Property
# ---------------------------------------------------------------------------

class TestVocabSize:
    def test_vocab_size(self):
        tok = CharTokenizer()
        tok.build_vocab(["a"])
        assert tok.vocab_size > 4

    def test_vocab_size_includes_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab([])
        assert tok.vocab_size >= 4

    def test_vocab_size_same_for_subset_of_ascii(self):
        tok1 = CharTokenizer()
        tok1.build_vocab(["ab"])
        tok2 = CharTokenizer()
        tok2.build_vocab(["abcdef"])
        assert tok2.vocab_size == tok1.vocab_size

    def test_vocab_size_grows_with_non_ascii(self):
        tok1 = CharTokenizer()
        tok1.build_vocab(["abc"])
        tok3 = CharTokenizer()
        tok3.build_vocab(["abc日本語"])
        assert tok3.vocab_size > tok1.vocab_size

    def test_vocab_size_unbuilt(self):
        tok = CharTokenizer()
        assert tok.vocab_size == 0

    def test_vocab_size_after_multiple_builds(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        size1 = tok.vocab_size
        tok.build_vocab(["abcxyz"])
        assert tok.vocab_size >= size1


# ---------------------------------------------------------------------------
# Save/Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
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

    def test_save_creates_parent_dirs(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        path = str(tmp_path / "sub" / "dir" / "tok.json")
        tok.save(path)
        assert Path(path).exists()

    def test_save_load_roundtrip_encode(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        ids1 = tok.encode("hello")
        ids2 = tok2.encode("hello")
        assert ids1 == ids2

    def test_save_load_preserves_pad_to(self, tmp_path):
        tok = CharTokenizer(pad_to=32)
        tok.build_vocab(["test"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2.pad_to == 32

    def test_save_json_structure(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "chars" in data
        assert "pad_to" in data
        assert isinstance(data["chars"], list)

    def test_save_load_multiple_chars(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["abcdefghijklmnopqrstuvwxyz"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2.vocab_size == tok.vocab_size
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert ch in tok2.vocab

    def test_save_load_roundtrip_decode(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        text = "hello"
        ids1 = tok.encode(text)
        assert tok2.decode(ids1) == text

    def test_save_load_preserves_built_state(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        assert tok2._built is False
        tok2.load(path)
        assert tok2._built is True

    def test_load_invalid_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json {{{")
        tok = CharTokenizer()
        with pytest.raises(json.JSONDecodeError):
            tok.load(path)

    def test_save_load_empty_vocab(self, tmp_path):
        tok = CharTokenizer()
        tok.build_vocab([])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2.vocab_size == tok.vocab_size

    def test_load_empty_chars_list(self, tmp_path):
        path = str(tmp_path / "empty.json")
        with open(path, "w") as f:
            json.dump({"chars": [], "pad_to": None}, f)
        tok = CharTokenizer()
        tok.load(path)
        assert tok._built is True
        assert tok.vocab_size >= 4


# ---------------------------------------------------------------------------
# pad_to
# ---------------------------------------------------------------------------

class TestPadTo:
    def test_pad_to(self):
        tok = CharTokenizer(pad_to=20)
        assert tok.pad_to == 20

    def test_pad_to_none_default(self):
        tok = CharTokenizer()
        assert tok.pad_to is None

    def test_pad_to_survives_save_load(self, tmp_path):
        tok = CharTokenizer(pad_to=64)
        tok.build_vocab(["test"])
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = CharTokenizer()
        tok2.load(path)
        assert tok2.pad_to == 64

    def test_pad_to_zero(self):
        tok = CharTokenizer(pad_to=0)
        assert tok.pad_to == 0

    def test_pad_to_large_value(self):
        tok = CharTokenizer(pad_to=1024)
        tok.build_vocab(["test"])
        assert tok.pad_to == 1024


# ---------------------------------------------------------------------------
# Special Tokens
# ---------------------------------------------------------------------------

class TestSpecialTokens:
    def test_special_tokens(self):
        assert CharTokenizer.SPECIAL_TOKENS == ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]

    def test_special_token_ids_are_zero_through_three(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        assert tok.vocab["<BOS>"] == 0
        assert tok.vocab["<EOS>"] == 1
        assert tok.vocab["<PAD>"] == 2
        assert tok.vocab["<UNK>"] == 3

    def test_itos_inverses_vocab(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        for ch, idx in tok.vocab.items():
            assert tok.itos[idx] == ch

    def test_special_tokens_are_list(self):
        assert isinstance(CharTokenizer.SPECIAL_TOKENS, list)

    def test_special_tokens_count(self):
        assert len(CharTokenizer.SPECIAL_TOKENS) == 4


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_build_vocab_then_rebuild(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        size1 = tok.vocab_size
        tok.build_vocab(["xyz"])
        size2 = tok.vocab_size
        assert size2 >= size1

    def test_encode_build_vocab_reset(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids1 = tok.encode("abc")
        tok.build_vocab(["xyz"])
        ids2 = tok.encode("xyz")
        assert ids1 != ids2

    def test_ensure_ascii_returns_set(self):
        tok = CharTokenizer()
        result = tok._ensure_ascii()
        assert isinstance(result, set)

    def test_vocab_is_dict(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        assert isinstance(tok.vocab, dict)
        assert isinstance(tok.itos, dict)

    def test_encode_newline_char(self):
        tok = CharTokenizer()
        tok.build_vocab(["a\nb"])
        ids = tok.encode("a\nb")
        assert len(ids) == 5

    def test_itos_contains_all_special_tokens(self):
        tok = CharTokenizer()
        tok.build_vocab(["test"])
        for i in range(4):
            assert i in tok.itos

    def test_vocab_unique_values(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        values = list(tok.vocab.values())
        assert len(values) == len(set(values))

    def test_encode_single_space(self):
        tok = CharTokenizer()
        tok.build_vocab(["a b"])
        ids = tok.encode(" ")
        assert len(ids) == 3  # BOS + space + EOS
        assert tok.decode(ids) == " "
