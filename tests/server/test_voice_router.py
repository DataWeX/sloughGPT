"""
Tests for the voice router — POST /voice/tts and GET /voice/status.
"""

import base64
import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.voice import router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _make_wav(frames: int = 24000, rate: int = 24000) -> bytes:
    """Build a 1-second mono 16-bit WAV payload."""
    buf = io.BytesIO()
    data = np.zeros(frames, dtype=np.int16).tobytes()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)
    buf.seek(0)
    return buf.read()


def _voice_router_instance():
    """Recover the module VoiceRouter instance via a bound route endpoint."""
    for route in router.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and getattr(endpoint, "__self__", None) is not None:
            return endpoint.__self__
    raise RuntimeError("no bound endpoint found")


def _mock_engine(waveform=None, fail=False):
    """Mock voice engine matching the synthesize()/status() contract."""
    engine = MagicMock()
    if fail:
        engine.synthesize.side_effect = RuntimeError("boom")
    else:
        result = MagicMock()
        result.success = True
        result.data = waveform if waveform is not None else np.zeros(22050, dtype=np.float32)
        result.error = None
        engine.synthesize.return_value = result
    return engine


class TestTTS:
    def test_produces_audio_with_voice_engine(self, client):
        with patch.object(_voice_router_instance(), "_get_engine", return_value=_mock_engine()):
            resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "voice-engine"
        assert data["audio"] != ""
        assert data["sample_rate"] == 22050

    def test_rejects_empty_text(self, client):
        resp = client.post("/voice/tts", json={"text": ""})
        assert resp.status_code == 422

    def test_rejects_whitespace_only(self, client):
        resp = client.post("/voice/tts", json={"text": "   "})
        assert resp.status_code == 400

    def test_long_text(self, client):
        resp = client.post("/voice/tts", json={"text": "word " * 100})
        assert resp.status_code == 200

    def test_special_characters(self, client):
        resp = client.post("/voice/tts", json={"text": "<hello> & 'world' \"test\""})
        assert resp.status_code == 200

    def test_unicode_text(self, client):
        resp = client.post("/voice/tts", json={"text": "Hello 你好世界"})
        assert resp.status_code == 200

    def test_single_word(self, client):
        resp = client.post("/voice/tts", json={"text": "Hi"})
        assert resp.status_code == 200

    def test_voice_param_forwarded_to_engine(self, client):
        engine = _mock_engine()
        with patch.object(_voice_router_instance(), "_get_engine", return_value=engine):
            resp = client.post("/voice/tts", json={"text": "test", "voice": "custom"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "voice-engine"
        engine.synthesize.assert_called_once_with("test", "custom")


class TestTTSSuccessPath:
    """POST /voice/tts with a mocked TTS backend."""

    @pytest.fixture(autouse=True)
    def _backend(self):
        backend = _mock_engine()
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            yield backend

    def test_returns_audio_when_backend_loaded(self, client):
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "voice-engine"
        assert data["audio"] != ""
        assert data["sample_rate"] == 22050
        assert data["duration_ms"] == 1000

    def test_sample_rate_is_fixed(self, client, _backend):
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["sample_rate"] == 22050

    def test_duration_from_frame_count(self, client, _backend):
        result = MagicMock()
        result.success = True
        result.data = np.zeros(8000, dtype=np.float32)
        result.error = None
        _backend.synthesize.return_value = result
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["duration_ms"] == 362

    def test_audio_decodes_as_wav(self, client):
        resp = client.post("/voice/tts", json={"text": "hello"})
        raw = base64.b64decode(resp.json()["audio"])
        with wave.open(io.BytesIO(raw)) as wf:
            assert wf.getnframes() == 22050

    def test_backend_failure_falls_back(self, client, _backend):
        _backend.synthesize.side_effect = RuntimeError("boom")
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"

    def test_load_failure_falls_back(self, client, _backend):
        _backend.synthesize.side_effect = RuntimeError("engine unavailable")
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["backend"] == "browser-fallback"


class TestVoiceStatusBackend:
    """GET /voice/status with a controllable backend."""

    def test_reports_available(self, client):
        backend = _mock_engine()
        backend.status.return_value = {"capabilities": ["tts"], "model": "voice"}
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["capabilities"] == ["tts"]

    def test_reports_unavailable_with_error(self, client):
        backend = _mock_engine()
        backend.status.side_effect = RuntimeError("engine load failed")
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            resp = client.get("/voice/status")
        assert resp.status_code == 500


class TestStatus:
    def test_returns_status(self, client):
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "server_tts" in data

    def test_status_has_expected_fields(self, client):
        resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert "server_tts" in data
        assert "capabilities" in data or "loaded" in data

    def test_status_structure(self, client):
        resp = client.get("/voice/status")
        body = resp.json()
        assert "data" in body
        assert "status" in body


class TestTTSValidation:
    """Request body validation bounds."""

    def test_missing_text_422(self, client):
        resp = client.post("/voice/tts", json={})
        assert resp.status_code == 422

    def test_text_wrong_type_422(self, client):
        resp = client.post("/voice/tts", json={"text": 42})
        assert resp.status_code == 422

    def test_voice_wrong_type_422(self, client):
        resp = client.post("/voice/tts", json={"text": "hi", "voice": 7})
        assert resp.status_code == 422

    def test_voice_param_passed_to_backend(self, client):
        backend = _mock_engine()
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            client.post("/voice/tts", json={"text": "hello", "voice": "en-us-female"})
        backend.synthesize.assert_called_once_with("hello", "en-us-female")


class TestVoiceMethodMismatch:
    """Wrong HTTP methods on voice routes."""

    def test_tts_get_405(self, client):
        resp = client.get("/voice/tts")
        assert resp.status_code == 405

    def test_status_post_405(self, client):
        resp = client.post("/voice/status")
        assert resp.status_code == 405


class TestVoiceStatusFailure:
    """Status when backend load raises."""

    def test_status_reports_error_after_load_failure(self, client):
        backend = _mock_engine()
        backend.status.side_effect = RuntimeError("crashed")
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            resp = client.get("/voice/status")
        assert resp.status_code == 500

    def test_status_load_success_reports_model(self, client):
        backend = _mock_engine()
        backend.status.return_value = {"capabilities": ["tts"]}
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["capabilities"] == ["tts"]


class TestTTSErrorAfterLoad:
    """TTS failure inside the try block falls back to browser."""

    def test_wav_read_error_falls_back(self, client):
        backend = _mock_engine(fail=True)
        with patch.object(_voice_router_instance(), "_get_engine", return_value=backend):
            resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"
