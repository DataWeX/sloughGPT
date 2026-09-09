"""Tests for multimodal/spanish_phoneme_encoder.py — SpanishPhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.spanish_phoneme_encoder import (
    SpanishPhonemeEncoder, spanish_text_to_phonemes,
    SPANISH_ID_TO_PHONEME, BOS, EOS, PAD, SPACE, NUM_SPANISH_PHONEMES,
)


class TestSpanishPhonemeEncoderEncode:
    def test_encode_hola(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("hola")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["OW", "L", "AH"]

    def test_encode_gracias(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("gracias")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["G", "R", "AH", "S", "AH", "S"]

    def test_encode_returns_numpy(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = SpanishPhonemeEncoder()
        assert enc.vocab_size == NUM_SPANISH_PHONEMES


class TestSpanishPhonemeEncoderDecode:
    def test_decode_hola(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("hola")
        text = enc.decode(ids)
        assert text == "olu"

    def test_decode_gracias(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("gracias")
        text = enc.decode(ids)
        assert text == "grusus"

    def test_decode_returns_string(self):
        enc = SpanishPhonemeEncoder()
        ids = enc.encode("test")
        text = enc.decode(ids)
        assert isinstance(text, str)


class TestSpanishPhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = SpanishPhonemeEncoder()
        result = enc.visualize("hola")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = SpanishPhonemeEncoder()
        result = enc.visualize("hola")
        assert "hola" in result

    def test_visualize_contains_phonemes(self):
        enc = SpanishPhonemeEncoder()
        result = enc.visualize("hola")
        assert "OW" in result
        assert "L" in result


class TestSpanishPhonemeEncoderRoundtrip:
    def test_roundtrip_common_words(self):
        enc = SpanishPhonemeEncoder()
        words = ["hola", "si", "no", "bueno", "gracias"]
        for word in words:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_roundtrip_sentences(self):
        enc = SpanishPhonemeEncoder()
        sentences = ["hola", "gracias", "buenos dias"]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0
