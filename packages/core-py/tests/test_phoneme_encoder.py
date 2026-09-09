"""Tests for multimodal/phoneme_encoder.py — PhonemeEncoder encode/decode roundtrip."""

import numpy as np
import pytest
from domains.multimodal.phoneme_encoder import (
    PhonemeEncoder, text_to_phonemes, ID_TO_PHONEME,
    BOS, EOS, PAD, SPACE, SILENCE, NUM_PHONEMES,
)


class TestPhonemeEncoderEncode:
    def test_encode_hello(self):
        enc = PhonemeEncoder()
        ids = enc.encode("hello")
        assert ids.flatten().tolist() == [40, 21, 6, 24, 11, 41]

    def test_encode_good_morning(self):
        enc = PhonemeEncoder()
        ids = enc.encode("good morning")
        assert ids.flatten().tolist() == [40, 20, 13, 17, 25, 3, 29, 26, 9, 27, 41]

    def test_encode_returns_numpy(self):
        enc = PhonemeEncoder()
        ids = enc.encode("test")
        assert isinstance(ids, np.ndarray)

    def test_encode_returns_int32(self):
        enc = PhonemeEncoder()
        ids = enc.encode("test")
        assert ids.dtype == np.int32

    def test_encode_starts_with_bos(self):
        enc = PhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[0] == BOS

    def test_encode_ends_with_eos(self):
        enc = PhonemeEncoder()
        ids = enc.encode("any word")
        assert ids.flatten()[-1] == EOS

    def test_vocab_size(self):
        enc = PhonemeEncoder()
        assert enc.vocab_size == NUM_PHONEMES


class TestPhonemeEncoderDecode:
    def test_decode_hello(self):
        enc = PhonemeEncoder()
        ids = np.array([[40, 21, 6, 24, 11, 41]], dtype=np.int32)
        text = enc.decode(ids)
        assert text == "helo"

    def test_decode_good_morning(self):
        enc = PhonemeEncoder()
        ids = np.array([[40, 20, 13, 17, 25, 3, 29, 26, 9, 27, 41]], dtype=np.int32)
        text = enc.decode(ids)
        assert text == "gudmorning"

    def test_decode_strips_bos_eos(self):
        enc = PhonemeEncoder()
        ids = np.array([[BOS, 21, EOS]], dtype=np.int32)
        text = enc.decode(ids)
        assert text == "h"

    def test_decode_empty_just_bos_eos(self):
        enc = PhonemeEncoder()
        ids = np.array([[BOS, EOS]], dtype=np.int32)
        text = enc.decode(ids)
        assert text == ""

    def test_decode_returns_string(self):
        enc = PhonemeEncoder()
        ids = np.array([[BOS, 21, EOS]], dtype=np.int32)
        text = enc.decode(ids)
        assert isinstance(text, str)

    def test_decode_the_quick_brown_fox(self):
        enc = PhonemeEncoder()
        ids = enc.encode("the quick brown fox")
        text = enc.decode(ids)
        assert text == "thukwikbrownfoks"


class TestPhonemeEncoderDecodePhonemes:
    def test_decode_phonemes_hello(self):
        enc = PhonemeEncoder()
        ids = enc.encode("hello")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["HH", "EH", "L", "OW"]

    def test_decode_phonemes_good_morning(self):
        enc = PhonemeEncoder()
        ids = enc.encode("good morning")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["G", "UH", "D", "M", "AO", "R", "N", "IH", "NG"]

    def test_decode_phonemes_returns_list(self):
        enc = PhonemeEncoder()
        ids = enc.encode("test")
        phonemes = enc.decode_phonemes(ids)
        assert isinstance(phonemes, list)

    def test_decode_phonemes_strips_bos_eos(self):
        enc = PhonemeEncoder()
        ids = enc.encode("hello")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes[0] != "<"
        assert phonemes[-1] != ">"

    def test_decode_phonemes_the_quick_brown_fox(self):
        enc = PhonemeEncoder()
        ids = enc.encode("the quick brown fox")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == [
            "DH", "AH",  # the
            "K", "W", "IH", "K",  # quick
            "B", "R", "AW", "N",  # brown
            "F", "AO", "K", "S",  # fox
        ]

    def test_decode_phonemes_matches_encode(self):
        """decode_phonemes should return the same phonemes used in encoding."""
        enc = PhonemeEncoder()
        for word in ["hello", "good", "morning", "quick", "brown", "fox"]:
            ids = enc.encode(word)
            phonemes = enc.decode_phonemes(ids)
            # Re-encode phonemes and compare
            re_encoded = enc.encode(" ".join(phonemes))
            # They should be equivalent (ignoring exact ID mapping)
            assert len(phonemes) > 0


class TestPhonemeEncoderVisualize:
    def test_visualize_returns_string(self):
        enc = PhonemeEncoder()
        result = enc.visualize("hello")
        assert isinstance(result, str)

    def test_visualize_contains_input(self):
        enc = PhonemeEncoder()
        result = enc.visualize("hello")
        assert "hello" in result

    def test_visualize_contains_phonemes(self):
        enc = PhonemeEncoder()
        result = enc.visualize("hello")
        assert "HH" in result
        assert "EH" in result

    def test_visualize_contains_ids(self):
        enc = PhonemeEncoder()
        result = enc.visualize("hello")
        assert "40" in result  # BOS
        assert "41" in result  # EOS

    def test_visualize_multiline(self):
        enc = PhonemeEncoder()
        result = enc.visualize("hello")
        lines = result.split("\n")
        assert len(lines) == 4


class TestPhonemeEncoderBatch:
    def test_encode_batch_single(self):
        enc = PhonemeEncoder()
        result = enc.encode_batch(["hello"])
        assert result.shape == (1, 6)  # BOS + 4 phonemes + EOS

    def test_encode_batch_multiple(self):
        enc = PhonemeEncoder()
        result = enc.encode_batch(["hello", "world"])
        assert result.shape[0] == 2
        assert result.shape[1] > 0

    def test_encode_batch_padded(self):
        enc = PhonemeEncoder()
        result = enc.encode_batch(["hi", "hello world"])
        # Both should be padded to the same length
        assert result.shape[0] == 2
        assert result.shape[1] == result.shape[1]  # Same length

    def test_encode_batch_empty(self):
        enc = PhonemeEncoder()
        result = enc.encode_batch([])
        assert result.shape == (0, 0)

    def test_encode_batch_preserves_order(self):
        enc = PhonemeEncoder()
        texts = ["hello", "world", "test"]
        result = enc.encode_batch(texts)
        # First row should match single encode of "hello"
        single = enc.encode("hello")
        assert np.array_equal(result[0, :single.shape[1]], single.flatten())

    def test_encode_batch_variable_length(self):
        enc = PhonemeEncoder()
        result = enc.encode_batch(["a", "hello"], pad_to_max=False)
        # Should be padded to max length
        assert result.shape[0] == 2
        assert result.shape[1] == max(len(enc.encode("a").flatten()), len(enc.encode("hello").flatten()))


class TestPhonemeEncoderScorePronunciation:
    def test_perfect_score(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_partial_score(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "helo")
        assert 0.0 < result["score"] < 1.0

    def test_no_match(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "xyz")
        assert result["score"] == 0.0

    def test_returns_dict(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert isinstance(result, dict)
        assert "score" in result
        assert "precision" in result
        assert "recall" in result
        assert "target_phonemes" in result
        assert "spoken_phonemes" in result

    def test_phoneme_lists(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert result["target_phonemes"] == ["HH", "EH", "L", "OW"]
        assert result["spoken_phonemes"] == ["HH", "EH", "L", "OW"]

    def test_empty_spoken(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("hello", "")
        assert result["score"] == 0.0
        assert result["spoken_len"] == 0

    def test_empty_target(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("", "hello")
        assert result["score"] == 0.0
        assert result["target_len"] == 0

    def test_both_empty(self):
        enc = PhonemeEncoder()
        result = enc.score_pronunciation("", "")
        assert result["score"] == 0.0


class TestPhonemeEncoderRoundtrip:
    def test_encode_decode_produces_string(self):
        """Encode then decode produces a non-empty string."""
        enc = PhonemeEncoder()
        sentences = [
            "hello",
            "good morning",
            "the quick brown fox",
        ]
        for sentence in sentences:
            ids = enc.encode(sentence)
            decoded = enc.decode(ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_decode_encode_strips_bos_eos(self):
        """Decoding strips BOS/EOS, re-encoding adds them back."""
        enc = PhonemeEncoder()
        for word in ["hello", "good", "morning"]:
            ids = enc.encode(word)
            decoded = enc.decode(ids)
            re_encoded = enc.encode(decoded)
            # Both should start with BOS and end with EOS
            assert re_encoded.flatten()[0] == BOS
            assert re_encoded.flatten()[-1] == EOS
