"""Tests for the voice API router (routers/voice.py).

Covers: VoiceRouter TTS and status endpoints.

The router reaches the consolidated `domain.voice` facade lazily via
`VoiceRouter._get_engine()` and caches it at `self._engine` (facade-shaped:
`.synthesize(text, voice) -> VoiceResult(.success/.error/.data=numpy float
waveform)` and `.status() -> dict`). All facade calls are mocked; only
HTTP-level behavior is tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from routers.voice import VoiceRouter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_engine(**overrides) -> MagicMock:
    """Create a facade-shaped VoiceEngine mock.

    Seam: prod `VoiceRouter._get_engine()` returns this object and caches it
    at `vr._engine`. The router calls:
      - engine.synthesize(text, voice) -> VoiceResult(success, error, data)
      - engine.status() -> dict with capabilities / server_tts keys
    """
    e = MagicMock()
    e.status.return_value = {
        "server_tts": overrides.get("status_server_tts", True),
        "capabilities": overrides.get("capabilities", ["tts", "stt"]),
        "model": "native-numpy",
    }

    def _synthesize(text, voice=None):
        result = MagicMock()
        result.success = overrides.get("success", True)
        result.error = overrides.get("error", None)
        seconds = overrides.get("seconds", 1.0)
        if result.success:
            result.data = np.zeros(int(22050 * seconds), dtype=np.float64)
        else:
            result.data = None
        return result

    e.synthesize.side_effect = _synthesize
    return e


def _app_with_engine(engine) -> FastAPI:
    """Fresh app where the VoiceRouter's `_engine` is the given mock.

    Matches the lazy-cache seam used by prod: `_get_engine()` returns
    `self._engine` when set, otherwise builds it.
    """
    app = FastAPI()
    vr = VoiceRouter.__new__(VoiceRouter)
    vr._engine = engine
    vr.router = APIRouter(prefix="/voice", tags=["voice"])
    vr.router.add_api_route("/tts", vr.text_to_speech, methods=["POST"])
    vr.router.add_api_route("/status", vr.voice_status, methods=["GET"])
    app.include_router(vr.router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceStatus:
    def test_status_when_unavailable(self):
        engine = _mock_engine(status_server_tts=False, capabilities=[])
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        # Prod hardcodes server_tts True in the envelope; "unavailable" surfaces
        # via empty capabilities + the engine-facing `loaded` status.
        assert data["server_tts"] is True
        assert data["capabilities"] == []
        assert data["loaded"]["server_tts"] is False

    def test_status_when_available(self):
        engine = _mock_engine(status_server_tts=True, capabilities=["tts"])
        client = TestClient(_app_with_engine(engine))
        resp = client.get("/voice/status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["server_tts"] is True
        assert data["capabilities"] == ["tts"]


class TestTextToSpeech:
    def test_empty_text_returns_400(self):
        engine = _mock_engine()
        client = TestClient(_app_with_engine(engine), raise_server_exceptions=False)
        resp = client.post("/voice/tts", json={"text": "   "})
        assert resp.status_code == 400

    def test_backend_unavailable_returns_browser_fallback(self):
        engine = _mock_engine(success=False, error="engine load failed")
        client = TestClient(_app_with_engine(engine), raise_server_exceptions=False)
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "browser-fallback"
        assert data["audio"] == ""

    def test_backend_generation_error_returns_fallback(self):
        engine = _mock_engine()
        engine.synthesize.side_effect = RuntimeError("GPU OOM")
        client = TestClient(_app_with_engine(engine), raise_server_exceptions=False)
        resp = client.post("/voice/tts", json={"text": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["backend"] == "browser-fallback"

    def test_successful_generation(self):
        engine = _mock_engine(seconds=1.0)
        client = TestClient(_app_with_engine(engine))
        resp = client.post("/voice/tts", json={"text": "Hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "voice-engine"
        assert data["sample_rate"] == 22050
        assert data["duration_ms"] == 1000
        assert len(data["audio"]) > 0
