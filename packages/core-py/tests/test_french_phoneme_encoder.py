"""Tests for multimodal/french_phoneme_encoder.py — FrenchPhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.french_phoneme_encoder import (
    FrenchPhonemeEncoder, french_text_to_phonemes,
    FRENCH_ID_TO_PHONEME, BOS, EOS, PAD, SPACE, NUM_FRENCH_PHONEMES,
)


class TestFrenchPhonemeEncoderEncode:
    def test_encode_je(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("je")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["ZH", "UH"]

    def test_encode_bonjour(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("bonjour")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["B", "OW", "ZH", "UW", "R"]

    def test_encode_returns_numpy(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = FrenchPhonemeEncoder()
        assert enc.vocab_size == NUM_FRENCH_PHONEMES


class TestFrenchPhonemeEncoderDecode:
    def test_decode_je(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("je")
        text = enc.decode(ids)
        assert text == "jou"

    def test_decode_bonjour(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("bonjour")
        text = enc.decode(ids)
        assert text == "beaujour"

    def test_decode_returns_string(self):
        enc = FrenchPhonemeEncoder()
        ids = enc.encode("test")
        text = enc.decode(ids)
        assert isinstance(text, str)


class TestFrenchPhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = FrenchPhonemeEncoder()
        result = enc.visualize("je")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = FrenchPhonemeEncoder()
        result = enc.visualize("je")
        assert "je" in result

    def test_visualize_contains_phonemes(self):
        enc = FrenchPhonemeEncoder()
        result = enc.visualize("je")
        assert "ZH" in result
        assert "UH" in result


class TestFrenchPhonemeEncoderRoundtrip:
    def test_roundtrip_common_words(self):
        enc = FrenchPhonemeEncoder()
        words = ["je", "tu", "bon", "merci", "bonjour"]
        for word in words:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_roundtrip_sentences(self):
        enc = FrenchPhonemeEncoder()
        sentences = ["bonjour", "merci", "je suis"]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0
