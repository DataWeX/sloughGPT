"""Voice Router - text-to-speech endpoint with optional HF model backend."""

import asyncio
import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas.common import success_response

logger = logging.getLogger("slo.routers.voice")


# ── TTS backend state (lazy-loaded) ─────────────────────────────────────

class _TTSBackend:
    """Lazy-loaded TTS engine using HuggingFace transformers."""

    def __init__(self):
        self._pipeline = None
        self._model_id = None
        self._loaded = False
        self._error = None

    def load(self):
        if self._loaded:
            return True
        try:
            from transformers import pipeline
            model_id = "suno/bark-small"
            self._pipeline = pipeline(
                "text-to-speech",
                model=model_id,
                device=-1,
            )
            self._model_id = model_id
            self._loaded = True
            logger.info(f"TTS model loaded: {model_id}", extra={"tag": "MODEL"})
            return True
        except ImportError:
            self._error = "transformers not available"
            logger.warning("TTS: transformers not installed", extra={"tag": "MODEL"})
            return False
        except Exception as e:
            self._error = str(e)
            logger.warning(f"TTS: failed to load model: {e}", extra={"tag": "MODEL"})
            return False

    def generate(self, text: str) -> bytes:
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
        """Convert text to speech audio.

        Uses HuggingFace TTS model (bark-small) if available.
        Returns base64-encoded WAV audio with sample rate metadata.

        Falls back with a browser-fallback signal so the frontend
        can use native speechSynthesis instead.
        """
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        try:
            if self._tts_backend.load():
                audio_bytes = await asyncio.to_thread(self._tts_backend.generate, request.text)
                sample_rate = 24000
                import wave
                with wave.open(io.BytesIO(audio_bytes)) as wf:
                    frames = wf.getnframes()
                    sr = wf.getframerate()
                    duration_ms = int(frames / sr * 1000) if sr > 0 else 0

                return TTSResponse(
                    audio=base64.b64encode(audio_bytes).decode("utf-8"),
                    sample_rate=sr,
                    duration_ms=duration_ms,
                    backend="hf-model",
                )
        except Exception as e:
            logger.warning(f"TTS generation failed, falling back to browser: {e}", extra={"tag": "MODEL"})

        return TTSResponse(
            audio="",
            sample_rate=0,
            duration_ms=0,
            backend="browser-fallback",
        )

    async def voice_status(self):
        """Check if server-side TTS model is available."""
        available = self._tts_backend.load()
        return success_response(data={
            "server_tts": available,
            "model": self._tts_backend._model_id if available else None,
            "error": self._tts_backend._error,
        })


router = VoiceRouter().router
