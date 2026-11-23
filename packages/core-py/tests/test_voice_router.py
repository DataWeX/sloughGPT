"""Tests for the voice API router (routers/voice.py).

Covers: VoiceRouter TTS and status endpoints against the VoiceEngine API
(``engine.synthesize()`` → VoiceResult, ``engine.status()`` → dict).
All domain calls are mocked; only HTTP-level behavior is tested.

VoiceRouter lazily creates ``self._engine`` via ``domain.voice.get_voice_engine``.
Tests set ``vr._engine`` directly on a bare instance and re-register routes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

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

SAMPLE_RATE = 22050  # hardcoded in routers/voice.py text_to_speech


def _app_with_engine(engine):
    """Create a fresh test app with a VoiceRouter using the given engine mock."""
    app = FastAPI()
    vr = VoiceRouter.__new__(VoiceRouter)
    vr._engine = engine

    from fastapi import APIRouter
    from routers.voice import TTSResponse

    vr.router = APIRouter(prefix="/voice", tags=["voice"])
    vr.router.add_api_route("/tts", vr.text_to_speech, methods=["POST"], response_model=TTSResponse)
    vr.router.add_api_route("/status", vr.voice_status, methods=["GET"])
    app.include_router(vr.router)

    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


def _voice_result(success=True, data=None, error=None):
    return SimpleNamespace(success=success, data=data, error=error, metadata={})


def _mock_engine(*, status=None, synthesize=None, status_raises=None, synth_raises=None):
    engine = MagicMock()
    if status_raises is not None:
        engine.status.side_effect = status_raises
    else:
        engine.status.return_value = (
            status
            if status is not None
            else {
                "tts": True,
                "recognizer": False,
                "encoder": True,
                "capabilities": ["tts", "stt", "phonemes", "language_detection"],
            }
        )
    if synth_raises is not None:
        engine.synthesize.side_effect = synth_raises
    elif synthesize is not None:
        engine.synthesize.return_value = synthesize
    else:
        engine.synthesize.return_value = _voice_result(success=False, error="not loaded")
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceStatus:
    def test_status_when_unavailable(self):
        engine = _mock_engine(status_raises=RuntimeError("engine load failed"))
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is False
        assert data["loaded"] is None
        assert data["capabilities"] == []

    def test_status_when_available(self):
        engine = _mock_engine()
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert "tts" in data["capabilities"]
        assert data["loaded"]["tts"] is True


class TestTextToSpeech:
    def test_empty_text_returns_400(self):
        engine = _mock_engine()
        client = TestClient(_app_with_engine(engine), raise_server_exceptions=False)
        resp = client.post("/voice/tts", json={"text": "   "})
        assert resp.status_code == 400

    def test_backend_unavailable_returns_browser_fallback(self):
        engine = _mock_engine(synthesize=_voice_result(success=False, error="not installed"))
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "browser-fallback"
        assert data["audio"] == ""

    def test_backend_generation_error_returns_fallback(self):
        engine = _mock_engine(synth_raises=RuntimeError("GPU OOM"))
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"

    def test_successful_generation(self):
        waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        engine = _mock_engine(synthesize=_voice_result(success=True, data=waveform))
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "voice-engine"
        assert data["sample_rate"] == SAMPLE_RATE
        assert data["duration_ms"] == 1000
        assert len(data["audio"]) > 0
