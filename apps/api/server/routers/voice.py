"""Voice Router - text-to-speech endpoint with optional HF model backend."""

import asyncio
import base64
import io
import logging
import time as _time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from schemas.common import raise_error, success_response, safe_audit_log, classify_and_raise

logger = logging.getLogger("slo.routers.voice")


# ── TTS backend state (lazy-loaded) ─────────────────────────────────────

class _TTSBackend:
    """Lazy-loaded TTS engine using HuggingFace transformers."""

    def __init__(self):
        self._pipeline = None
        self._model_id = None
        self._loaded = False
        self._error = None

    def load(self) -> dict:
        """load."""
        if self._loaded:
            return True
        self._error = "Text-to-speech requires transformers, which is not supported"
        logger.warning("TTS: transformers not available", extra={"tag": "MODEL"})
        return False

    def generate(self, text: str) -> bytes:
        """generate."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError(f"TTS unavailable: {self._error}")
        result = self._pipeline(text)
        audio_array = result["audio"]
        sample_rate = result["sampling_rate"]
        import numpy as np
        audio_int16 = (audio_array * 32767).astype(np.int16)
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.read()


# ── Schema ──────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class TTSResponse(BaseModel):
    audio: str
    sample_rate: int
    duration_ms: int
    backend: str


# ── Router ──────────────────────────────────────────────────────────────

class VoiceRouter:
    def __init__(self):
        self._tts_backend = _TTSBackend()
        self.router = APIRouter(prefix="/voice", tags=["voice"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/tts", self.text_to_speech, methods=["POST"], response_model=TTSResponse)
        self.router.add_api_route("/status", self.voice_status, methods=["GET"])

    async def text_to_speech(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech audio."""
        if not request.text.strip():
            raise_error("No text provided", "E_BAD_REQUEST", status_code=400)

        _t0 = _time.monotonic()
        try:
            if self._tts_backend.load():
                audio_bytes = await asyncio.to_thread(self._tts_backend.generate, request.text)
                import wave
                with wave.open(io.BytesIO(audio_bytes)) as wf:
                    frames = wf.getnframes()
                    sr = wf.getframerate()
                    duration_ms = int(frames / sr * 1000) if sr > 0 else 0

                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                logger.info("TTS generated in %.1fms (duration=%dms)", _elapsed_ms, duration_ms)
                safe_audit_log("voice.tts", resource=request.text[:80], detail=f"duration={duration_ms}ms elapsed={_elapsed_ms:.0f}ms")
                return TTSResponse(
                    audio=base64.b64encode(audio_bytes).decode("utf-8"),
                    sample_rate=sr,
                    duration_ms=duration_ms,
                    backend="hf-model",
                )
        except Exception as e:
            logger.warning("TTS generation failed, falling back to browser: %s", e, extra={"tag": "MODEL"})

        return TTSResponse(
            audio="",
            sample_rate=0,
            duration_ms=0,
            backend="browser-fallback",
        )

    async def voice_status(self) -> dict:
        """Check if server-side TTS model is available."""
        try:
            available = self._tts_backend.load()
            return success_response(data={
                "server_tts": available,
                "model": self._tts_backend._model_id if available else None,
                "error": self._tts_backend._error,
            })
        except Exception as e:
            classify_and_raise(e, source="voice.status")


router = VoiceRouter().router
