"""Tests for multimodal/unified_phoneme_encoder.py — UnifiedPhonemeEncoder."""

import numpy as np
import pytest
from domains.multimodal.unified_phoneme_encoder import UnifiedPhonemeEncoder, detect_language


class TestLanguageDetection:
    def test_detect_english(self):
        assert detect_language("hello world") == "en"

    def test_detect_german(self):
        assert detect_language("ich bin") == "de"

    def test_detect_german_greeting(self):
        assert detect_language("guten morgen") == "de"

    def test_detect_french(self):
        assert detect_language("je suis") == "fr"

    def test_detect_french_greeting(self):
        assert detect_language("bonjour") == "fr"

    def test_detect_spanish(self):
        assert detect_language("hola") == "es"

    def test_detect_spanish_greeting(self):
        assert detect_language("hola amigos") == "es"

    def test_default_english(self):
        assert detect_language("unknown text") == "en"

    def test_detect_german_words(self):
        german_words = ["ich", "du", "er", "sie", "wir", "danke", "bitte"]
        for word in german_words:
            assert detect_language(word) == "de"

    def test_detect_french_words(self):
        french_words = ["je", "tu", "il", "elle", "bonjour", "merci", "oui"]
        for word in french_words:
            assert detect_language(word) == "fr"

    def test_detect_spanish_words(self):
        spanish_words = ["hola", "si", "no", "bueno", "gracias", "adios"]
        for word in spanish_words:
            assert detect_language(word) == "es"

    def test_detect_italian(self):
        assert detect_language("ciao") == "it"

    def test_detect_italian_greeting(self):
        assert detect_language("buongiorno") == "it"

    def test_detect_italian_words(self):
        italian_words = ["io", "lui", "lei", "grazie", "prego", "bene"]
        for word in italian_words:
            assert detect_language(word) == "it"

    def test_detect_portuguese(self):
        assert detect_language("ola") == "pt"

    def test_detect_portuguese_greeting(self):
        assert detect_language("obrigado") == "pt"

    def test_detect_portuguese_words(self):
        portuguese_words = ["eu", "ele", "ela", "obrigado", "bem", "sim"]
        for word in portuguese_words:
            assert detect_language(word) == "pt"


class TestUnifiedPhonemeEncoder:
    def test_encode_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        assert ids.flatten()[0] == 40  # BOS (English)

    def test_encode_german(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ich")
        assert ids.flatten()[0] == 44  # BOS (German)

    def test_encode_french(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("je")
        assert ids.flatten()[0] == 43  # BOS (French)

    def test_encode_spanish(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola")
        assert ids.flatten()[0] == 43  # BOS (Spanish)

    def test_encode_with_language(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello", language="en")
        assert ids.flatten()[0] == 40  # BOS

    def test_encode_german_with_language(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ich", language="de")
        assert ids.flatten()[0] == 44  # BOS

    def test_encode_french_with_language(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("je", language="fr")
        assert ids.flatten()[0] == 43  # BOS

    def test_encode_spanish_with_language(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola", language="es")
        assert ids.flatten()[0] == 43  # BOS

    def test_decode_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        text = enc.decode(ids)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_german(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ich")
        text = enc.decode(ids, language="de")
        assert text == "ich"

    def test_decode_french(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("je")
        text = enc.decode(ids, language="fr")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_spanish(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola")
        text = enc.decode(ids, language="es")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_phonemes_english(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hello")
        phonemes = enc.decode_phonemes(ids)
        assert phonemes == ["HH", "EH", "L", "OW"]

    def test_decode_phonemes_german(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ich")
        phonemes = enc.decode_phonemes(ids, language="de")
        assert phonemes == ["IH", "CH"]

    def test_decode_phonemes_french(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("je")
        phonemes = enc.decode_phonemes(ids, language="fr")
        assert phonemes == ["ZH", "UH"]

    def test_decode_phonemes_spanish(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("hola")
        phonemes = enc.decode_phonemes(ids, language="es")
        assert phonemes == ["OW", "L", "AH"]

    def test_visualize_english(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("hello")
        assert "hello" in result
        assert "HH" in result

    def test_visualize_german(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("ich")
        assert "ich" in result
        assert "IH" in result

    def test_visualize_french(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("je")
        assert "je" in result
        assert "ZH" in result

    def test_visualize_spanish(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("hola")
        assert "hola" in result
        assert "OW" in result

    def test_encode_italian(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ciao", language="it")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32
        assert ids.shape[0] == 1

    def test_decode_italian(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ciao", language="it")
        text = enc.decode(ids, language="it")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_phonemes_italian(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ciao", language="it")
        phonemes = enc.decode_phonemes(ids, language="it")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
        assert "CH" in phonemes
        assert "AW" in phonemes

    def test_visualize_italian(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("ciao")
        assert "ciao" in result
        assert "CH" in result

    def test_encode_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ola", language="pt")
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.int32
        assert ids.shape[0] == 1

    def test_decode_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ola", language="pt")
        text = enc.decode(ids, language="pt")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_decode_phonemes_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        ids = enc.encode("ola", language="pt")
        phonemes = enc.decode_phonemes(ids, language="pt")
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
        assert "OW" in phonemes
        assert "L" in phonemes

    def test_visualize_portuguese(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.visualize("ola")
        assert "ola" in result
        assert "OW" in result

    def test_score_pronunciation(self):
        enc = UnifiedPhonemeEncoder()
        result = enc.score_pronunciation("hello", "hello")
        assert result["score"] == 1.0

    def test_supported_languages(self):
        enc = UnifiedPhonemeEncoder()
        assert "en" in enc.supported_languages
        assert "de" in enc.supported_languages
        assert "fr" in enc.supported_languages
        assert "es" in enc.supported_languages
        assert "it" in enc.supported_languages
        assert "pt" in enc.supported_languages

    def test_current_language(self):
        enc = UnifiedPhonemeEncoder()
        enc.encode("hello")
        assert enc.current_language == "en"
        enc.encode("ich")
        assert enc.current_language == "de"
        enc.encode("je")
        assert enc.current_language == "fr"
        enc.encode("hola")
        assert enc.current_language == "es"
        enc.encode("ciao")
        assert enc.current_language == "it"
        enc.encode("ola")
        assert enc.current_language == "pt"

    def test_unsupported_language(self):
        enc = UnifiedPhonemeEncoder()
        with pytest.raises(ValueError):
            enc.encode("hello", language="zh")
