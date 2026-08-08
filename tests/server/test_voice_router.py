"""
Tests for the voice router — POST /voice/tts and GET /voice/status.
"""

import io
import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.voice import router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestTTS:
    def test_returns_fallback_when_model_unavailable(self, client):
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio"] == ""
        assert data["backend"] == "browser-fallback"

    def test_rejects_empty_text(self, client):
        resp = client.post("/voice/tts", json={"text": ""})
        assert resp.status_code == 400

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

    def test_fallback_has_empty_audio(self, client):
        resp = client.post("/voice/tts", json={"text": "test"})
        data = resp.json()
        assert data["audio"] == ""
        assert data["sample_rate"] == 0
        assert data["duration_ms"] == 0

    def test_voice_param_ignored_when_no_model(self, client):
        resp = client.post("/voice/tts", json={"text": "test", "voice": "custom"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"


def _make_wav(frames: int = 24000, rate: int = 24000) -> bytes:
    """Build a 1-second mono 16-bit WAV payload."""
    import numpy as np
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


class TestTTSSuccessPath:
    """POST /voice/tts with a working TTS backend."""

    @pytest.fixture(autouse=True)
    def _backend(self):
        backend = MagicMock()
        backend.load.return_value = True
        backend.generate.return_value = _make_wav()
        with patch.object(_voice_router_instance(), "_tts_backend", backend):
            yield backend

    def test_returns_audio_when_backend_loaded(self, client):
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "hf-model"
        assert data["audio"] != ""
        assert data["sample_rate"] == 24000
        assert data["duration_ms"] == 1000

    def test_sample_rate_read_from_wav(self, client, _backend):
        _backend.generate.return_value = _make_wav(rate=16000)
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["sample_rate"] == 16000

    def test_duration_from_frame_count(self, client, _backend):
        _backend.generate.return_value = _make_wav(frames=8000, rate=16000)
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["duration_ms"] == 500

    def test_audio_decodes_as_wav(self, client):
        import base64
        resp = client.post("/voice/tts", json={"text": "hello"})
        raw = base64.b64decode(resp.json()["audio"])
        with wave.open(io.BytesIO(raw)) as wf:
            assert wf.getnframes() == 24000

    def test_backend_failure_falls_back(self, client, _backend):
        _backend.generate.side_effect = RuntimeError("boom")
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"

    def test_load_failure_falls_back(self, client, _backend):
        _backend.load.return_value = False
        resp = client.post("/voice/tts", json={"text": "hello"})
        assert resp.json()["backend"] == "browser-fallback"


class TestVoiceStatusBackend:
    """GET /voice/status with a controllable backend."""

    def test_reports_available_with_model(self, client):
        backend = MagicMock()
        backend.load.return_value = True
        backend._model_id = "suno/bark-small"
        backend._error = None
        with patch.object(_voice_router_instance(), "_tts_backend", backend):
            resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["model"] == "suno/bark-small"
        assert data["error"] is None

    def test_reports_unavailable_with_error(self, client):
        backend = MagicMock()
        backend.load.return_value = False
        backend._model_id = None
        backend._error = "transformers not available"
        with patch.object(_voice_router_instance(), "_tts_backend", backend):
            resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert data["server_tts"] is False
        assert data["model"] is None
        assert data["error"] == "transformers not available"


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
        assert "model" in data or "error" in data

    def test_status_model_none_when_unavailable(self, client):
        resp = client.get("/voice/status")
        data = resp.json()["data"]
        assert data["model"] is None

    def test_status_structure(self, client):
        resp = client.get("/voice/status")
        body = resp.json()
        assert "data" in body
        assert "status" in body
