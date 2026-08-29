"""Tests for the voice API router (routers/voice.py).

Covers: VoiceRouter TTS and status endpoints.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: VoiceRouter sets _tts_backend in __init__ as an instance attribute
on an anonymous instance (router = VoiceRouter().router). We patch
_TTSBackend to control what __init__ creates, then access the instance
via the closure or just re-instantiate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.voice import VoiceRouter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_with_backend(backend):
    """Create a fresh test app with a VoiceRouter using the given backend mock."""
    app = FastAPI()
    # Create a new VoiceRouter but replace _tts_backend before route registration
    vr = VoiceRouter.__new__(VoiceRouter)
    vr._tts_backend = backend
    # Re-register routes on a fresh router
    from fastapi import APIRouter
    vr.router = APIRouter(prefix="/voice", tags=["voice"])

    # Register the same endpoints as the real __init__
    vr.router.add_api_route("/tts", vr.text_to_speech, methods=["POST"])
    vr.router.add_api_route("/status", vr.voice_status, methods=["GET"])
    app.include_router(vr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


def _mock_backend(**overrides):
    """Create a MagicMock that behaves like _TTSBackend."""
    b = MagicMock()
    b.load.return_value = overrides.get("load_return", False)
    b._model_id = overrides.get("model_id", None)
    b._error = overrides.get("error", None)
    b.generate.return_value = overrides.get("generate_return", b"")
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVoiceStatus:
    def test_status_when_unavailable(self):
        backend = _mock_backend(load_return=False, error="transformers not available")
        client = TestClient(_app_with_backend(backend))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is False
        assert data["model"] is None

    def test_status_when_available(self):
        backend = _mock_backend(load_return=True, model_id="suno/bark-small")
        client = TestClient(_app_with_backend(backend))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["model"] == "suno/bark-small"


class TestTextToSpeech:
    def test_empty_text_returns_400(self):
        backend = _mock_backend()
        client = TestClient(_app_with_backend(backend), raise_server_exceptions=False)
        resp = client.post("/voice/tts", json={"text": "   "})
        assert resp.status_code == 400

    def test_backend_unavailable_returns_browser_fallback(self):
        backend = _mock_backend(load_return=False, error="not installed")
        client = TestClient(_app_with_backend(backend))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "browser-fallback"
        assert data["audio"] == ""

    def test_backend_generation_error_returns_fallback(self):
        backend = _mock_backend(load_return=True)
        backend.generate.side_effect = RuntimeError("GPU OOM")
        client = TestClient(_app_with_backend(backend))
        resp = client.post("/voice/tts", json={"text": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"

    def test_successful_generation(self):
        import wave
        import io
        import numpy as np

        # Build a minimal WAV file
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            audio_int16 = np.zeros(24000, dtype=np.int16)  # 1 second of silence
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = buf.getvalue()

        backend = _mock_backend(load_return=True, generate_return=wav_bytes)
        client = TestClient(_app_with_backend(backend))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "hf-model"
        assert data["sample_rate"] == 24000
        assert data["duration_ms"] == 1000
        assert len(data["audio"]) > 0
