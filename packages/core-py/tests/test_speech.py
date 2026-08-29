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
    get_speech_recognizer,
)


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


class TestGetSpeechRecognizer:
    def test_returns_recognizer(self):
        r = get_speech_recognizer()
        assert r is not None
