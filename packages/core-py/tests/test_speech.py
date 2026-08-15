"""Tests for the multimodal speech recognition module."""

import os
import sys
from types import SimpleNamespace

import pytest

from domains.multimodal.speech import (
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    SpeechRecognizer,
    TranscriptionResult,
    get_speech_recognizer,
)


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_defaults(self):
        r = TranscriptionResult(text="hi", confidence=0.9, language="en")
        assert r.text == "hi"
        assert r.confidence == 0.9
        assert r.language == "en"
        assert r.duration is None

    def test_with_duration(self):
        r = TranscriptionResult(text="hi", confidence=0.5, language="fr", duration=1.5)
        assert r.duration == 1.5


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestSpeechRecognizerProtocol:
    def test_protocol_interface(self):
        assert hasattr(SpeechRecognizer, "recognize")
        assert hasattr(SpeechRecognizer, "recognize_stream")

    def test_implementations_have_recognize(self):
        assert callable(BrowserSpeechRecognizer().get_config)
        assert callable(ServerSpeechRecognizer().recognize)


# ---------------------------------------------------------------------------
# BrowserSpeechRecognizer
# ---------------------------------------------------------------------------

class TestBrowserSpeechRecognizer:
    def test_default_language(self):
        rec = BrowserSpeechRecognizer()
        assert rec.language == "en-US"
        assert rec.continuous is False
        assert rec.interim_results is True

    def test_custom_language(self):
        rec = BrowserSpeechRecognizer(language="es-ES")
        assert rec.language == "es-ES"

    def test_get_config(self):
        rec = BrowserSpeechRecognizer(language="fr-FR")
        cfg = rec.get_config()
        assert cfg == {
            "language": "fr-FR",
            "continuous": False,
            "interimResults": True,
        }


# ---------------------------------------------------------------------------
# ServerSpeechRecognizer — backend discovery
# ---------------------------------------------------------------------------

class FakeVoskModel:
    def __init__(self, path):
        self.path = path


class FakeKaldiRecognizer:
    def __init__(self, model, rate):
        self.model = model
        self.rate = rate
        self.waveform = None

    def AcceptWaveform(self, data):
        self.waveform = data

    def FinalResult(self):
        return '{"text": "hello world"}'


FAKE_VOSK = SimpleNamespace(Model=FakeVoskModel, KaldiRecognizer=FakeKaldiRecognizer)


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def recognize_google(self, audio, language="en-US"):
        self.calls.append((audio, language))
        return "recognized text"


class FakeAudioData:
    def __init__(self, data, sample_rate, sample_width):
        self.data = data
        self.sample_rate = sample_rate
        self.sample_width = sample_width


FAKE_SR = SimpleNamespace(Recognizer=FakeRecognizer, AudioData=FakeAudioData)


@pytest.fixture
def no_backends(monkeypatch):
    monkeypatch.delitem(sys.modules, "vosk", raising=False)
    monkeypatch.delitem(sys.modules, "speech_recognition", raising=False)


@pytest.fixture
def with_vosk(monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", FAKE_VOSK)
    monkeypatch.delitem(sys.modules, "speech_recognition", raising=False)


@pytest.fixture
def with_speech_recognition(monkeypatch):
    monkeypatch.delitem(sys.modules, "vosk", raising=False)
    monkeypatch.setitem(sys.modules, "speech_recognition", FAKE_SR)


class TestServerInit:
    def test_defaults(self):
        rec = ServerSpeechRecognizer()
        assert rec.model_name == "base"
        assert rec._model is None
        assert rec._processor is None
        assert rec._backend is None

    def test_custom_model_name(self):
        rec = ServerSpeechRecognizer(model_name="large")
        assert rec.model_name == "large"


class TestLoadModel:
    def test_no_backend_available(self, no_backends, caplog):
        rec = ServerSpeechRecognizer()
        with caplog.at_level("WARNING"):
            rec.load_model()
        assert rec._backend is None
        assert any("No ASR backend available" in r.message for r in caplog.records)

    def test_vosk_backend(self, with_vosk):
        rec = ServerSpeechRecognizer()
        rec.load_model()
        assert rec._backend == "vosk"
        assert rec._model is None

    def test_speech_recognition_backend(self, with_speech_recognition):
        rec = ServerSpeechRecognizer()
        rec.load_model()
        assert rec._backend == "speech_recognition"
        assert isinstance(rec._model, FakeRecognizer)

    def test_vosk_takes_precedence(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vosk", FAKE_VOSK)
        monkeypatch.setitem(sys.modules, "speech_recognition", FAKE_SR)
        rec = ServerSpeechRecognizer()
        rec.load_model()
        assert rec._backend == "vosk"


class TestRecognizeNoBackend:
    def test_empty_result_without_backend(self, no_backends):
        rec = ServerSpeechRecognizer()
        result = rec.recognize(b"\x00\x01", language="en")
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.language == "en"


class TestRecognizeSpeechRecognition:
    def test_success(self, with_speech_recognition):
        rec = ServerSpeechRecognizer()
        rec.load_model()
        result = rec.recognize(b"audio-bytes", language="fr")
        assert result.text == "recognized text"
        assert result.confidence == 0.9
        assert result.language == "fr"
        audio, lang = rec._model.calls[-1]
        assert lang == "fr"
        assert audio.sample_rate == 16000
        assert audio.sample_width == 2

    def test_error_returns_empty(self, with_speech_recognition):
        rec = ServerSpeechRecognizer()
        rec.load_model()
        rec._model.recognize_google = lambda audio, language="en-US": (
            (_ for _ in ()).throw(RuntimeError("boom"))
        )
        result = rec.recognize(b"audio-bytes")
        assert result.text == ""
        assert result.confidence == 0.0

    def test_recognize_auto_loads(self, with_speech_recognition):
        rec = ServerSpeechRecognizer()
        assert rec._backend is None
        result = rec.recognize(b"bytes")
        assert result.text == "recognized text"


class TestRecognizeVosk:
    def test_success(self, with_vosk, monkeypatch, tmp_path):
        monkeypatch.setenv("VOSK_MODEL_PATH", str(tmp_path))
        rec = ServerSpeechRecognizer()
        rec.load_model()
        result = rec.recognize(b"pcm-data")
        assert result.text == "hello world"
        assert result.confidence == 0.9

    def test_decode_vosk_import_error(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "vosk", raising=False)
        rec = ServerSpeechRecognizer()
        assert rec._decode_vosk(b"data") == ""

    def test_decode_vosk_missing_model_path(self, with_vosk, monkeypatch, caplog):
        monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
        rec = ServerSpeechRecognizer()
        rec._backend = "vosk"
        with caplog.at_level("WARNING"):
            text = rec._decode_vosk(b"data")
        assert text == ""
        assert any("vosk model not found" in r.message for r in caplog.records)

    def test_decode_vosk_uses_env_path(self, with_vosk, monkeypatch, tmp_path):
        monkeypatch.setenv("VOSK_MODEL_PATH", str(tmp_path))
        rec = ServerSpeechRecognizer()
        rec._backend = "vosk"
        text = rec._decode_vosk(b"data")
        assert text == "hello world"
        assert isinstance(rec._model, FakeVoskModel)
        assert rec._model.path == str(tmp_path)

    def test_decode_vosk_reuses_loaded_model(self, with_vosk, monkeypatch, tmp_path):
        rec = ServerSpeechRecognizer()
        rec._backend = "vosk"
        rec._model = FakeVoskModel(str(tmp_path))
        text = rec._decode_vosk(b"data")
        assert text == "hello world"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetSpeechRecognizer:
    def test_default_returns_browser(self):
        rec = get_speech_recognizer()
        assert isinstance(rec, BrowserSpeechRecognizer)

    def test_server_mode(self):
        rec = get_speech_recognizer(use_server=True, model_name="medium")
        assert isinstance(rec, ServerSpeechRecognizer)
        assert rec.model_name == "medium"
