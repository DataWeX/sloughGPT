"""Tests for the voice API router (routers/voice.py).

These exercise the **consolidated VoiceEngine facade seam**: VoiceRouter lazily
builds the production facade via `_get_engine()` (cached at `self._engine`) and
`engine.synthesize(text, voice)` yields a `VoiceResult(.success/.error/.data=
numpy float64 waveform)`. The router re-encodes that waveform to a 22050 Hz
WAV and reports `backend="voice-engine"` on success, falling back to
`"browser-fallback"` (empty audio) on any generation failure or unavailable
engine.

All domain calls are mocked at the facade seam; only HTTP-level behavior and
the prod envelope literals are tested.
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SERVER_DIR = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
import sys  # noqa: E402

if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# ---------------------------------------------------------------------------
# Helpers — facade-shaped engine mock
# ---------------------------------------------------------------------------


def _mock_engine(available: bool = True, capabilities: list | None = None) -> MagicMock:
    """Facade-shaped VoiceEngine mock: synthesize→VoiceResult, status→dict."""
    engine = MagicMock()

    def synthesize(text: str, voice: str | None = None):
        if not available:
            return MagicMock(success=False, error="engine unavailable", data=None)
        frames = 22050  # 1 second at the prod re-encode rate
        waveform = np.zeros(frames, dtype=np.float64)
        return MagicMock(success=True, error=None, data=waveform)

    engine.synthesize.side_effect = synthesize
    engine.status.return_value = (
        {"capabilities": capabilities if capabilities is not None else ["web-speech-api"]}
        if available
        else {"capabilities": []}
    )
    return engine


def _app_with_engine(engine: MagicMock) -> FastAPI:
    """Build a fresh app whose VoiceRouter uses the given engine at the seam."""
    from fastapi import APIRouter  # noqa: PLC0415
    from routers import voice as voice_module

    app = FastAPI()

    vr = voice_module.VoiceRouter.__new__(voice_module.VoiceRouter)
    vr._engine = engine
    vr.router = APIRouter(prefix="/voice", tags=["voice"])

    original_get_engine = voice_module.VoiceRouter._get_engine

    def _inject(self):
        return engine  # always the injected mock at the facade seam

    voice_module.VoiceRouter._get_engine = _inject
    try:
        vr.router.add_api_route("/tts", vr.text_to_speech, methods=["POST"])
        vr.router.add_api_route("/status", vr.voice_status, methods=["GET"])
        app.include_router(vr.router)
    finally:
        voice_module.VoiceRouter._get_engine = original_get_engine
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceStatus:
    def test_status_when_unavailable(self):
        engine = _mock_engine(available=False)
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        # Prod envelope hardcodes server_tts True; unavailability is signaled
        # via empty capabilities + loaded status, not the flag.
        assert data["server_tts"] is True  # prod literal (always True)
        assert data["capabilities"] == []
        assert data["loaded"] == {"capabilities": []}

    def test_status_when_available(self):
        engine = _mock_engine(available=True, capabilities=["web-speech-api"])
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["capabilities"] == ["web-speech-api"]
        assert data["loaded"] == {"capabilities": ["web-speech-api"]}


class TestTextToSpeech:
    def test_success_returns_encoded_wav(self):
        engine = _mock_engine(available=True)
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        # TTS endpoint returns the TTSResponse envelope at top level
        data = resp.json()
        assert data["backend"] == "voice-engine"  # prod literal
        assert data["sample_rate"] == 22050  # prod re-encode rate
        # 22050 frames of silence → 1000ms at 22050
        assert data["duration_ms"] == 1000

        audio = data["audio"]
        assert audio
        with wave.open(io_reader(audio), "rb") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 22050

    def test_backend_unavailable_returns_browser_fallback(self):
        engine = _mock_engine(available=False)
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "browser-fallback"
        assert data["audio"] == ""
        assert data["duration_ms"] == 0

    def test_backend_generation_error_returns_fallback(self):
        engine = _mock_engine(available=True)
        engine.synthesize.side_effect = RuntimeError("GPU OOM")
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "browser-fallback"
        assert data["audio"] == ""


def io_reader(b64data: str):
    """Return a BytesIO of decoded base64 WAV bytes."""
    import base64
    import io

    return io.BytesIO(base64.b64decode(b64data))
