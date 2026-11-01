"""Voice Router — TTS and voice status via VoiceEngine.

All voice operations delegate to domain.voice.VoiceEngine.
"""

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
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

logger = logging.getLogger("slo.routers.voice")


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
        self._engine = None
        self.router = APIRouter(prefix="/voice", tags=["voice"])
        self._register_routes()

    def _get_engine(self):
        if self._engine is None:
            from domain.voice import get_voice_engine

            self._engine = get_voice_engine()
        return self._engine

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

            engine = self._get_engine()
            _t0 = _time.monotonic()
            try:
                result = await asyncio.to_thread(engine.synthesize, request.text, request.voice)
                if not result.success:
                    raise RuntimeError(result.error)

                waveform = result.data
                sample_rate = 22050

                audio_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                audio_bytes = buf.getvalue()

                with wave.open(io.BytesIO(audio_bytes)) as wf:
                    frames = wf.getnframes()
                    duration_ms = int(frames / sample_rate * 1000) if sample_rate > 0 else 0

                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                logger.info("TTS generated in %.1fms (duration=%dms)", _elapsed_ms, duration_ms)
                safe_audit_log(
                    "voice.tts",
                    resource=request.text[:80],
                    detail=f"duration={duration_ms}ms elapsed={_elapsed_ms:.0f}ms",
                )
                return TTSResponse(
                    audio=base64.b64encode(audio_bytes).decode("utf-8"),
                    sample_rate=sample_rate,
                    duration_ms=duration_ms,
                    backend="voice-engine",
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
        """Check voice engine status and capabilities."""
        try:
            engine = self._get_engine()
            status = engine.status()
            return success_response(
                data={
                    "server_tts": True,
                    "capabilities": status.get("capabilities", []),
                    "loaded": status,
                }
            )
        except Exception as e:
            classify_and_raise(e, source="voice.status")


router = VoiceRouter().router
