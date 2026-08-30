"""Tests for domains.multimodal.speech — TranscriptionResult, BrowserSpeechRecognizer,
ServerSpeechRecognizer, get_speech_recognizer.

Covers: dataclass creation, browser config, server recognizer init, factory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.multimodal.speech import (
    TranscriptionResult,
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    get_speech_recognizer,
)


# ── TranscriptionResult ──────────────────────────────────────────────────


class TestTranscriptionResult:
    def test_creation(self):
        r = TranscriptionResult(text="hello world", confidence=0.9, language="en-US")
        assert r.text == "hello world"
        assert r.confidence == 0.9
        assert r.language == "en-US"
        assert r.duration is None

    def test_with_duration(self):
        r = TranscriptionResult(text="test", confidence=0.8, language="en", duration=5.0)
        assert r.duration == 5.0

    def test_is_valid_default(self):
        r = TranscriptionResult(text="ok", confidence=1.0, language="en")
        assert r.is_valid is True

    def test_is_valid_false(self):
        r = TranscriptionResult(text="", confidence=0.0, language="en", is_valid=False)
        assert r.is_valid is False

    def test_empty_text(self):
        r = TranscriptionResult(text="", confidence=0.0, language="en")
        assert r.text == ""

    def test_zero_confidence(self):
        r = TranscriptionResult(text="x", confidence=0.0, language="en")
        assert r.confidence == 0.0

    def test_max_confidence(self):
        r = TranscriptionResult(text="x", confidence=1.0, language="en")
        assert r.confidence == 1.0

    def test_long_text(self):
        long = "word " * 1000
        r = TranscriptionResult(text=long, confidence=0.5, language="en")
        assert len(r.text) > 4000

    def test_unicode_text(self):
        r = TranscriptionResult(text="こんにちは世界", confidence=0.95, language="ja")
        assert r.text == "こんにちは世界"

    def test_special_characters(self):
        r = TranscriptionResult(text="hello <>&\"'", confidence=0.8, language="en")
        assert r.text == "hello <>&\"'"

    def test_all_fields_explicit(self):
        r = TranscriptionResult(text="t", confidence=0.7, language="fr", duration=3.14, is_valid=False)
        assert r.text == "t"
        assert r.confidence == 0.7
        assert r.language == "fr"
        assert r.duration == 3.14
        assert r.is_valid is False

    def test_negative_duration(self):
        r = TranscriptionResult(text="x", confidence=0.5, language="en", duration=-1.0)
        assert r.duration == -1.0

    def test_many_languages(self):
        for lang in ["en-US", "fr-FR", "de-DE", "ja-JP", "zh-CN", "ko-KR", "ar-SA"]:
            r = TranscriptionResult(text="hi", confidence=0.9, language=lang)
            assert r.language == lang

    def test_equality(self):
        r1 = TranscriptionResult(text="a", confidence=0.8, language="en")
        r2 = TranscriptionResult(text="a", confidence=0.8, language="en")
        assert r1 == r2

    def test_repr(self):
        r = TranscriptionResult(text="hi", confidence=0.9, language="en")
        assert "TranscriptionResult" in repr(r)


# ── BrowserSpeechRecognizer ──────────────────────────────────────────────


class TestBrowserSpeechRecognizer:
    def test_default_config(self):
        r = BrowserSpeechRecognizer()
        config = r.get_config()
        assert config["language"] == "en-US"
        assert config["continuous"] is False
        assert config["interimResults"] is True

    def test_custom_language(self):
        r = BrowserSpeechRecognizer(language="fr-FR")
        assert r.language == "fr-FR"
        config = r.get_config()
        assert config["language"] == "fr-FR"

    def test_default_language(self):
        r = BrowserSpeechRecognizer()
        assert r.language == "en-US"

    def test_continuous_default(self):
        r = BrowserSpeechRecognizer()
        assert r.continuous is False

    def test_interim_results_default(self):
        r = BrowserSpeechRecognizer()
        assert r.interim_results is True

    def test_config_keys(self):
        r = BrowserSpeechRecognizer()
        config = r.get_config()
        assert set(config.keys()) == {"language", "continuous", "interimResults"}

    def test_config_values_types(self):
        r = BrowserSpeechRecognizer()
        config = r.get_config()
        assert isinstance(config["language"], str)
        assert isinstance(config["continuous"], bool)
        assert isinstance(config["interimResults"], bool)

    def test_multiple_languages(self):
        for lang in ["de-DE", "es-ES", "it-IT", "pt-BR", "zh-CN", "ja-JP"]:
            r = BrowserSpeechRecognizer(language=lang)
            assert r.get_config()["language"] == lang

    def test_config_dict_is_new(self):
        r = BrowserSpeechRecognizer()
        c1 = r.get_config()
        c2 = r.get_config()
        assert c1 == c2
        c1["language"] = "modified"
        assert r.get_config()["language"] == "en-US"

    def test_init_only_sets_language(self):
        r = BrowserSpeechRecognizer(language="ko-KR")
        assert r.language == "ko-KR"
        assert r.continuous is False
        assert r.interim_results is True

    def test_empty_language(self):
        r = BrowserSpeechRecognizer(language="")
        assert r.get_config()["language"] == ""

    def test_no_args(self):
        r = BrowserSpeechRecognizer()
        assert r is not None
        assert isinstance(r, BrowserSpeechRecognizer)


# ── ServerSpeechRecognizer ───────────────────────────────────────────────


class TestServerSpeechRecognizer:
    def test_init_default(self):
        r = ServerSpeechRecognizer()
        assert r.model_name == "base"
        assert r._model is None
        assert r._processor is None
        assert r._backend is None

    def test_init_custom_model(self):
        r = ServerSpeechRecognizer(model_name="medium")
        assert r.model_name == "medium"

    def test_recognize_no_backend(self):
        r = ServerSpeechRecognizer()
        result = r.recognize(b"audio_data", language="en")
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.is_valid is False

    def test_recognize_no_backend_fr(self):
        r = ServerSpeechRecognizer()
        result = r.recognize(b"audio", language="fr")
        assert result.language == "fr"
        assert result.is_valid is False

    def test_load_model_no_backends(self):
        r = ServerSpeechRecognizer()
        r.load_model()
        assert r._backend is None

    def test_decode_vosk_no_import(self):
        r = ServerSpeechRecognizer()
        text = r._decode_vosk(b"audio")
        assert text == ""

    def test_model_name_preserved(self):
        r = ServerSpeechRecognizer(model_name="large")
        r.load_model()
        assert r.model_name == "large"

    def test_recognize_empty_bytes(self):
        r = ServerSpeechRecognizer()
        result = r.recognize(b"", language="en")
        assert result.text == ""
        assert result.is_valid is False

    def test_recognize_sets_language(self):
        r = ServerSpeechRecognizer()
        result = r.recognize(b"data", language="de")
        assert result.language == "de"

    def test_multiple_recognize_calls(self):
        r = ServerSpeechRecognizer()
        r1 = r.recognize(b"a", language="en")
        r2 = r.recognize(b"b", language="en")
        assert r1.is_valid is False
        assert r2.is_valid is False


# ── get_speech_recognizer ────────────────────────────────────────────────


class TestGetSpeechRecognizer:
    def test_returns_recognizer(self):
        r = get_speech_recognizer()
        assert r is not None

    def test_default_is_browser(self):
        r = get_speech_recognizer()
        assert isinstance(r, BrowserSpeechRecognizer)

    def test_server_flag(self):
        r = get_speech_recognizer(use_server=True)
        assert isinstance(r, ServerSpeechRecognizer)

    def test_default_is_browser_when_false(self):
        r = get_speech_recognizer(use_server=False)
        assert isinstance(r, BrowserSpeechRecognizer)

    def test_server_model_name(self):
        r = get_speech_recognizer(use_server=True, model_name="large")
        assert isinstance(r, ServerSpeechRecognizer)
        assert r.model_name == "large"

    def test_default_model_name(self):
        r = get_speech_recognizer(use_server=True)
        assert r.model_name == "base"

    def test_browser_with_model_name_ignored(self):
        r = get_speech_recognizer(use_server=False, model_name="large")
        assert isinstance(r, BrowserSpeechRecognizer)
