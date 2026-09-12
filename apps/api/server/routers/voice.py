"""Voice Router - text-to-speech endpoint using native numpy TTS engine."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time as _time
import wave

import numpy as np

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, endpoint, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.voice")


# ── TTS backend state (lazy-loaded) ─────────────────────────────────────


class _TTSBackend:
    """Native TTS engine using phoneme encoder + spectrogram decoder + Griffin-Lim vocoder.

    All pure numpy — no torch or transformers dependency.
    """

    def __init__(self):
        self._engine = None
        self._loaded = False
        self._error = None

    def load(self) -> bool:
        """Load the native TTS engine."""
        if self._loaded:
            return True
        try:
            from domains.multimodal.tts import TTSEngine

            self._engine = TTSEngine()
            self._loaded = True
            self._error = None
            logger.info("Native TTS engine loaded (phoneme + Griffin-Lim)", extra={"tag": "MODEL"})
            return True
        except Exception as e:
            self._error = f"TTS engine load failed: {e}"
            logger.warning("TTS: engine load failed: %s", e, extra={"tag": "MODEL"})
            return False

    def generate(self, text: str) -> tuple[bytes, int]:
        """Generate WAV audio bytes from text.

        Returns:
            (wav_bytes, sample_rate) tuple
        """
        if not self._loaded:
            if not self.load():
                raise RuntimeError(f"TTS unavailable: {self._error}")
        try:
            waveform = self._engine.text_to_waveform(text)
            sample_rate = self._engine.sample_rate

            audio_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            buf.seek(0)
            return buf.read(), sample_rate

        except Exception as e:
            classify_and_raise(e, source="voice.generate")


# ── Schema ──────────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to convert to speech")
    voice: str | None = Field(default=None, max_length=100, description="Voice identifier")


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
        self.router.add_api_route(
            "/tts", self.text_to_speech, methods=["POST"], response_model=TTSResponse
        )
        self.router.add_api_route("/status", self.voice_status, methods=["GET"])

    @endpoint("voice.tts")
    async def text_to_speech(
        self, request: TTSRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> TTSResponse:
        """Convert text to speech audio."""
        try:
            if not request.text.strip():
                raise_error("No text provided", "E_BAD_REQUEST", status_code=400)

            _t0 = _time.monotonic()
            try:
                if self._tts_backend.load():
                    audio_bytes, sr = await asyncio.to_thread(
                        self._tts_backend.generate, request.text
                    )

                    with wave.open(io.BytesIO(audio_bytes)) as wf:
                        frames = wf.getnframes()
                        duration_ms = int(frames / sr * 1000) if sr > 0 else 0

                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    logger.info("TTS generated in %.1fms (duration=%dms)", _elapsed_ms, duration_ms)
                    safe_audit_log(
                        "voice.tts",
                        resource=request.text[:80],
                        detail=f"duration={duration_ms}ms elapsed={_elapsed_ms:.0f}ms",
                    )
                    return TTSResponse(
                        audio=base64.b64encode(audio_bytes).decode("utf-8"),
                        sample_rate=sr,
                        duration_ms=duration_ms,
                        backend="native-numpy",
                    )
            except Exception as e:
                logger.warning(
                    "TTS generation failed, falling back to browser: %s", e, extra={"tag": "MODEL"}
                )

            return TTSResponse(
                audio="",
                sample_rate=0,
                duration_ms=0,
                backend="browser-fallback",
            )

        except Exception as e:
            classify_and_raise(e, source="voice.text_to_speech")

    @endpoint("voice.status")
    async def voice_status(self) -> dict:
        """Check if server-side TTS model is available."""
        try:
            available = self._tts_backend.load()
            return success_response(
                data={
                    "server_tts": available,
                    "model": "native-numpy" if available else None,
                    "error": self._tts_backend._error,
                }
            )
        except Exception as e:
            classify_and_raise(e, source="voice.status")


router = VoiceRouter().router
