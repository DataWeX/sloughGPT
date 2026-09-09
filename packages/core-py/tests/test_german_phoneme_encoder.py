"""Tests for multimodal/german_phoneme_encoder.py — GermanPhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.german_phoneme_encoder import (
    GermanPhonemeEncoder, german_text_to_phonemes,
    GERMAN_ID_TO_PHONEME, BOS, EOS, PAD, SPACE, NUM_GERMAN_PHONEMES,
)


class TestGermanPhonemeEncoderEncode:
    def test_encode_ich(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("ich")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["IH", "CH"]

    def test_encode_guten_morgen(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("guten morgen")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["G", "UW", "T", "AH", "N", "M", "AO", "R", "G", "AH", "N"]

    def test_encode_returns_numpy(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = GermanPhonemeEncoder()
        assert enc.vocab_size == NUM_GERMAN_PHONEMES


class TestGermanPhonemeEncoderDecode:
    def test_decode_ich(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("ich")
        text = enc.decode(ids)
        assert text == "ich"

    def test_decode_guten_morgen(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("guten morgen")
        text = enc.decode(ids)
        assert text == "gutunmorgun"

    def test_decode_returns_string(self):
        enc = GermanPhonemeEncoder()
        ids = enc.encode("test")
        text = enc.decode(ids)
        assert isinstance(text, str)


class TestGermanPhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = GermanPhonemeEncoder()
        result = enc.visualize("ich")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = GermanPhonemeEncoder()
        result = enc.visualize("ich")
        assert "ich" in result

    def test_visualize_contains_phonemes(self):
        enc = GermanPhonemeEncoder()
        result = enc.visualize("ich")
        assert "IH" in result
        assert "CH" in result


class TestGermanPhonemeEncoderRoundtrip:
    def test_roundtrip_common_words(self):
        enc = GermanPhonemeEncoder()
        words = ["ich", "du", "gut", "hallo", "welt"]
        for word in words:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            # Check that decoding produces something
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_roundtrip_sentences(self):
        enc = GermanPhonemeEncoder()
        sentences = ["guten morgen", "danke bitte", "ich bin"]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0


class TestGermanPhonemeEncoderScore:
    def test_score_perfect(self):
        enc = GermanPhonemeEncoder()
        result = enc.score_pronunciation("hallo", "hallo")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_imperfect(self):
        enc = GermanPhonemeEncoder()
        result = enc.score_pronunciation("hallo", "helo")
        assert 0.0 <= result["score"] <= 1.0
        assert result["score"] < 1.0

    def test_score_empty_spoken(self):
        enc = GermanPhonemeEncoder()
        result = enc.score_pronunciation("hallo", "")
        assert result["score"] == 0.0

    def test_score_returns_dict(self):
        enc = GermanPhonemeEncoder()
        result = enc.score_pronunciation("test", "test")
        assert isinstance(result, dict)
        assert "score" in result
        assert "precision" in result
        assert "recall" in result
        assert "target_phonemes" in result
        assert "spoken_phonemes" in result

    def test_score_common_words(self):
        enc = GermanPhonemeEncoder()
        words = ["ich", "du", "gut", "welt"]
        for word in words:
            result = enc.score_pronunciation(word, word)
            assert result["score"] == 1.0
