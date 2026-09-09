"""Tests for multimodal/italian_phoneme_encoder.py — ItalianPhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.italian_phoneme_encoder import (
    ItalianPhonemeEncoder, italian_text_to_phonemes,
    ITALIAN_ID_TO_PHONEME, BOS, EOS, PAD, SPACE, NUM_ITALIAN_PHONEMES,
)


class TestItalianPhonemeEncoderEncode:
    def test_encode_ciao(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("ciao")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["CH", "IY", "AW"]

    def test_encode_casa(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("casa")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["K", "AH", "S", "AH"]

    def test_encode_grazie(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("grazie")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["G", "R", "AH", "TS", "IY", "EH"]

    def test_encode_returns_numpy(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = ItalianPhonemeEncoder()
        assert enc.vocab_size == NUM_ITALIAN_PHONEMES


class TestItalianPhonemeEncoderDecode:
    def test_decode_ciao(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("ciao")
        text = enc.decode(ids)
        assert text == "chiau"

    def test_decode_casa(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("casa")
        text = enc.decode(ids)
        assert text == "cusu"

    def test_decode_returns_string(self):
        enc = ItalianPhonemeEncoder()
        ids = enc.encode("test")
        text = enc.decode(ids)
        assert isinstance(text, str)


class TestItalianPhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = ItalianPhonemeEncoder()
        result = enc.visualize("ciao")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = ItalianPhonemeEncoder()
        result = enc.visualize("ciao")
        assert "ciao" in result

    def test_visualize_contains_phonemes(self):
        enc = ItalianPhonemeEncoder()
        result = enc.visualize("ciao")
        assert "CH" in result
        assert "AW" in result


class TestItalianPhonemeEncoderRoundtrip:
    def test_roundtrip_common_words(self):
        enc = ItalianPhonemeEncoder()
        words = ["ciao", "si", "no", "bene", "grazie"]
        for word in words:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_roundtrip_sentences(self):
        enc = ItalianPhonemeEncoder()
        sentences = ["ciao", "grazie", "buongiorno"]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0


class TestItalianTextToPhonemes:
    def test_returns_list(self):
        result = italian_text_to_phonemes("ciao")
        assert isinstance(result, list)

    def test_starts_with_bos(self):
        result = italian_text_to_phonemes("ciao")
        assert result[0] == ITALIAN_ID_TO_PHONEME[BOS]

    def test_ends_with_eos(self):
        result = italian_text_to_phonemes("ciao")
        assert result[-1] == ITALIAN_ID_TO_PHONEME[EOS]

    def test_dict_lookup(self):
        result = italian_text_to_phonemes("ciao")
        # "ciao" is in the dictionary
        assert "CH" in result
        assert "IY" in result
        assert "AW" in result

    def test_rule_fallback(self):
        result = italian_text_to_phonemes("xyz")
        # "xyz" is not in the dictionary, should use rules
        assert isinstance(result, list)
        assert len(result) > 2  # BOS + phonemes + EOS
