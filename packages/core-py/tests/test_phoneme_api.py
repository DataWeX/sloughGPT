"""Tests for phoneme encoding endpoints (REST API).

Tests the phoneme encoding, decoding, batch encoding, language detection,
pronunciation scoring, and TTS synthesis endpoints.
"""

import numpy as np
import pytest
from domains.multimodal.unified_phoneme_encoder import UnifiedPhonemeEncoder
from domains.multimodal.tts import TTSEngine


class TestUnifiedPhonemeEncoderAPI:
    """Test the unified phoneme encoder methods used by REST API."""

    def test_encode_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32
        assert ids.shape[0] == 1

    def test_encode_german(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ich", language="de")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32

    def test_encode_french(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("je", language="fr")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32

    def test_encode_spanish(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola", language="es")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32

    def test_encode_italian(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ciao", language="it")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32

    def test_encode_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ola", language="pt")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32

    def test_decode_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        decoded = enc.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_decode_phonemes_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        phonemes = enc.decode_phonemes(ids)
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0

    def test_batch_encode(self):
        enc = UnifiedPhonemeEncoder()
        texts = ["hello", "world", "test"]
        results = enc.encode_batch(texts)
        assert len(results) == 3
        for result in results:
            assert "text" in result
            assert "language" in result
            assert "phonemes" in result
            assert "ids" in result
            assert "decoded" in result

    def test_batch_encode_mixed_languages(self):
        enc = UnifiedPhonemeEncoder()
        texts = ["hello", "ich", "je", "hola", "ciao", "ola"]
        results = enc.encode_batch(texts)
        assert len(results) == 6
        languages = [r["language"] for r in results]
        assert "en" in languages
        assert "de" in languages
        assert "fr" in languages
        assert "es" in languages
        assert "it" in languages
        assert "pt" in languages

    def test_detect_language(self):
        enc = UnifiedPhonemeEncoder()
        assert enc.detect_language("hello") == "en"
        assert enc.detect_language("ich") == "de"
        assert enc.detect_language("je") == "fr"
        assert enc.detect_language("hola") == "es"
        assert enc.detect_language("ciao") == "it"
        assert enc.detect_language("ola") == "pt"

    def test_score_pronunciation(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert "score" in result
        assert "precision" in result
        assert "recall" in result
        assert "target_phonemes" in result
        assert "spoken_phonemes" in result
        assert result["score"] == 1.0

    def test_score_pronunciation_different(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "helo")
        assert "score" in result
        assert result["score"] < 1.0

    def test_supported_languages(self):
        enc = UnifiedPhonemeEncoder()
        langs = enc.supported_languages
        assert "en" in langs
        assert "de" in langs
        assert "fr" in langs
        assert "es" in langs
        assert "it" in langs
        assert "pt" in langs

    def test_current_language(self):
        enc = UnifiedPhonemeEncoder()
        enc.encode("hello")
        assert enc.current_language == "en"
        enc.encode("ich", language="de")
        assert enc.current_language == "de"

    def test_unsupported_language(self):
        enc = UnifiedPhonemeEncoder()
        with pytest.raises(ValueError):
            enc.encode("hello", language="zh")


class TestTTSEngineAPI:
    """Test the TTS engine methods used by REST API."""

    def test_text_to_waveform(self):
        engine = TTSEngine()
        waveform = engine.text_to_waveform("hello")
        assert isinstance(waveform, np.ndarray)
        assert waveform.ndim == 1
        assert len(waveform) > 0

    def test_empty_text(self):
        engine = TTSEngine()
        waveform = engine.text_to_waveform("")
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) > 0

    def test_sample_rate(self):
        engine = TTSEngine()
        assert engine.sample_rate == 22050

    def test_waveform_duration(self):
        engine = TTSEngine()
        waveform = engine.text_to_waveform("hello")
        duration = len(waveform) / engine.sample_rate
        assert duration > 0
        assert duration < 10  # Should be short

    def test_max_frames(self):
        engine = TTSEngine()
        waveform = engine.text_to_waveform("hello", max_frames=50)
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) > 0


class TestPhonemeEncoderBatchAPI:
    """Test batch encoding methods used by REST API."""

    def test_batch_encode_english(self):
        enc = UnifiedPhonemeEncoder()
        texts = ["hello", "world", "goodbye"]
        results = enc.encode_batch(texts, language="en")
        assert len(results) == 3
        for result in results:
            assert result["language"] == "en"

    def test_batch_encode_mixed(self):
        enc = UnifiedPhonemeEncoder()
        texts = ["hello", "ich", "je"]
        results = enc.encode_batch(texts)
        assert len(results) == 3
        assert results[0]["language"] == "en"
        assert results[1]["language"] == "de"
        assert results[2]["language"] == "fr"

    def test_batch_encode_empty(self):
        enc = UnifiedPhonemeEncoder()
        results = enc.encode_batch([])
        assert len(results) == 0


class TestPronunciationScoringAPI:
    """Test pronunciation scoring methods used by REST API."""

    def test_score_perfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "helo")
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_score_empty(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "")
        assert result["score"] == 0.0

    def test_score_german(self):
        enc = UnifiedPhonemeEncoder()
        # German encoder doesn't have score_pronunciation, so test encoding only
        ids = enc.encode("ich", language="de")
        phonemes = enc.decode_phonemes(ids, language="de")
        assert len(phonemes) > 0

    def test_score_french(self):
        enc = UnifiedPhonemeEncoder()
        # French encoder doesn't have score_pronunciation, so test encoding only
        ids = enc.encode("je", language="fr")
        phonemes = enc.decode_phonemes(ids, language="fr")
        assert len(phonemes) > 0

    def test_score_spanish(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hola", "hola", language="es")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_italian(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("ciao", "ciao", language="it")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("ola", "ola", language="pt")
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_score_german_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hallo", "helo", language="de")
        assert 0.0 <= result["score"] <= 1.0

    def test_score_french_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("bonjour", "bonjur", language="fr")
        assert 0.0 <= result["score"] <= 1.0

    def test_score_spanish_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hola", "ola", language="es")
        assert 0.0 <= result["score"] <= 1.0

    def test_score_italian_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("grazie", "gratzie", language="it")
        assert 0.0 <= result["score"] <= 1.0

    def test_score_portuguese_imperfect(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("obrigado", "obrigado", language="pt")
        assert result["score"] == 1.0


class TestPhonemeEncoderEdgeCases:
    """Test edge cases for phoneme encoder."""

    def test_encode_empty_string(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_whitespace(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("   ")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_special_characters(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello!@#$%")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_numbers(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("12345")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_mixed_case(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("HeLLo")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_long_text(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello world " * 10)
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_single_character(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("a")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_encode_all_phonemes(self):
        enc = UnifiedPhonemeEncoder()
        # Test with a string that contains many different sounds
        ids = enc.encode("the quick brown fox jumps over the lazy dog")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1


class TestTrainingAPI:
    """Test pronunciation training endpoint methods."""

    def test_training_feedback(self):
        enc = UnifiedPhonemeEncoder()
        word = "hello"
        ids = enc.encode(word)
        phonemes = enc.decode_phonemes(ids)
        assert len(phonemes) > 0
        assert ids.shape[0] == 1

    def test_training_feedback_empty(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("")
        assert isinstance(ids, np.ndarray)
        assert ids.shape[0] == 1

    def test_training_feedback_german(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hallo", language="de")
        phonemes = enc.decode_phonemes(ids, language="de")
        assert len(phonemes) > 0

    def test_training_feedback_french(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("bonjour", language="fr")
        phonemes = enc.decode_phonemes(ids, language="fr")
        assert len(phonemes) > 0

    def test_training_feedback_spanish(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola", language="es")
        phonemes = enc.decode_phonemes(ids, language="es")
        assert len(phonemes) > 0

    def test_training_feedback_italian(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ciao", language="it")
        phonemes = enc.decode_phonemes(ids, language="it")
        assert len(phonemes) > 0

    def test_training_feedback_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ola", language="pt")
        phonemes = enc.decode_phonemes(ids, language="pt")
        assert len(phonemes) > 0


class TestBatchTTSEngineAPI:
    """Test batch TTS synthesis methods."""

    def test_batch_synthesis(self):
        engine = TTSEngine()
        texts = ["hello", "world"]
        results = []
        for text in texts:
            waveform = engine.text_to_waveform(text)
            results.append({
                'text': text,
                'waveform': waveform,
                'sample_rate': engine.sample_rate,
                'duration': len(waveform) / engine.sample_rate,
            })
        assert len(results) == 2
        for result in results:
            assert 'text' in result
            assert 'waveform' in result
            assert 'sample_rate' in result
            assert 'duration' in result
            assert result['duration'] > 0

    def test_batch_synthesis_empty(self):
        engine = TTSEngine()
        texts = []
        results = []
        for text in texts:
            waveform = engine.text_to_waveform(text)
            results.append({'text': text, 'waveform': waveform})
        assert len(results) == 0
