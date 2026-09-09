"""Tests for multimodal/portuguese_phoneme_encoder.py — PortuguesePhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.portuguese_phoneme_encoder import (
    PortuguesePhonemeEncoder, portuguese_text_to_phonemes,
    PORTUGUESE_ID_TO_PHONEME, BOS, EOS, PAD, SPACE, NUM_PORTUGUESE_PHONEMES,
)


class TestPortuguesePhonemeEncoderEncode:
    def test_encode_ola(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("ola")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["OW", "L", "AH"]

    def test_encode_casa(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("casa")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["K", "AH", "Z", "AH"]

    def test_encode_obrigado(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("obrigado")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["OW", "B", "R", "IY", "G", "AH", "D", "OW"]

    def test_encode_returns_numpy(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = PortuguesePhonemeEncoder()
        assert enc.vocab_size == NUM_PORTUGUESE_PHONEMES


class TestPortuguesePhonemeEncoderDecode:
    def test_decode_ola(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("ola")
        text = enc.decode(ids)
        assert text == "olu"

    def test_decode_casa(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("casa")
        text = enc.decode(ids)
        assert text == "cuzu"

    def test_decode_returns_string(self):
        enc = PortuguesePhonemeEncoder()
        ids = enc.encode("test")
        text = enc.decode(ids)
        assert isinstance(text, str)


class TestPortuguesePhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = PortuguesePhonemeEncoder()
        result = enc.visualize("ola")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = PortuguesePhonemeEncoder()
        result = enc.visualize("ola")
        assert "ola" in result

    def test_visualize_contains_phonemes(self):
        enc = PortuguesePhonemeEncoder()
        result = enc.visualize("ola")
        assert "OW" in result
        assert "L" in result


class TestPortuguesePhonemeEncoderRoundtrip:
    def test_roundtrip_common_words(self):
        enc = PortuguesePhonemeEncoder()
        words = ["ola", "sim", "nao", "bem", "obrigado"]
        for word in words:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_roundtrip_sentences(self):
        enc = PortuguesePhonemeEncoder()
        sentences = ["ola", "obrigado", "bom dia"]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0


class TestPortugueseTextToPhonemes:
    def test_returns_list(self):
        result = portuguese_text_to_phonemes("ola")
        assert isinstance(result, list)

    def test_starts_with_bos(self):
        result = portuguese_text_to_phonemes("ola")
        assert result[0] == PORTUGUESE_ID_TO_PHONEME[BOS]

    def test_ends_with_eos(self):
        result = portuguese_text_to_phonemes("ola")
        assert result[-1] == PORTUGUESE_ID_TO_PHONEME[EOS]

    def test_dict_lookup(self):
        result = portuguese_text_to_phonemes("ola")
        # "ola" is in the dictionary
        assert "OW" in result
        assert "L" in result
        assert "AH" in result

    def test_rule_fallback(self):
        result = portuguese_text_to_phonemes("xyz")
        # "xyz" is not in the dictionary, should use rules
        assert isinstance(result, list)
        assert len(result) > 2  # BOS + phonemes + EOS
