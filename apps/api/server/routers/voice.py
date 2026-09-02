"""Voice Router - text-to-speech and speech-to-text endpoints."""

import asyncio
import base64
import io
import logging
import time as _time
from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Form
from pydantic import Field
from pydantic import BaseModel

from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, success_response, safe_audit_log, classify_and_raise

logger = logging.getLogger("slo.routers.voice")

# Common voice/language codes for MMS-TTS
_POPULAR_VOICES = [
    {"code": "eng", "name": "English", "family": "eng"},
    {"code": "spa", "name": "Spanish", "family": "spa"},
    {"code": "fra", "name": "French", "family": "fra"},
    {"code": "deu", "name": "German", "family": "deu"},
    {"code": "ita", "name": "Italian", "family": "ita"},
    {"code": "por", "name": "Portuguese", "family": "por"},
    {"code": "rus", "name": "Russian", "family": "rus"},
    {"code": "jpn", "name": "Japanese", "family": "jpn"},
    {"code": "kor", "name": "Korean", "family": "kor"},
    {"code": "zho", "name": "Chinese", "family": "zho"},
    {"code": "ara", "name": "Arabic", "family": "ara"},
    {"code": "hin", "name": "Hindi", "family": "hin"},
    {"code": "tur", "name": "Turkish", "family": "tur"},
    {"code": "nld", "name": "Dutch", "family": "nld"},
    {"code": "swe", "name": "Swedish", "family": "swe"},
    {"code": "pol", "name": "Polish", "family": "pol"},
    {"code": "ces", "name": "Czech", "family": "ces"},
    {"code": "ron", "name": "Romanian", "family": "ron"},
    {"code": "ukr", "name": "Ukrainian", "family": "ukr"},
    {"code": "vie", "name": "Vietnamese", "family": "vie"},
]


# ── TTS backend state (lazy-loaded) ─────────────────────────────────────

class _TTSBackend:
    """Lazy-loaded TTS engine using HuggingFace transformers."""

    def __init__(self):
        self._pipeline = None
        self._model_id = None
        self._loaded = False
        self._error = None

    def load(self) -> bool:
        """Load the TTS model pipeline from HuggingFace transformers."""
        if self._loaded:
            return True
        try:
            from transformers import pipeline as hf_pipeline
            self._model_id = "facebook/mms-tts-eng"
            self._pipeline = hf_pipeline("text-to-speech", model=self._model_id)
            self._loaded = True
            self._error = None
            logger.info("TTS model loaded: %s", self._model_id, extra={"tag": "MODEL"})
            return True
        except ImportError:
            self._error = "Text-to-speech requires transformers package"
            logger.warning("TTS: transformers not installed", extra={"tag": "MODEL"})
            return False
        except Exception as e:
            self._error = f"TTS model load failed: {e}"
            logger.warning("TTS: model load failed: %s", e, extra={"tag": "MODEL"})
            return False
    def generate(self, text: str, voice: str = "eng") -> bytes:
        """Generate WAV audio bytes from text using the loaded TTS pipeline."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError(f"TTS unavailable: {self._error}")
        try:
            result = self._pipeline(text, voice=voice)
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


        except Exception as e:
            classify_and_raise(e, source="voice.generate")


# ── STT backend state (lazy-loaded) ─────────────────────────────────────

class _STTBackend:
    """Lazy-loaded STT engine using ServerSpeechRecognizer."""

    def __init__(self):
        self._recognizer = None
        self._loaded = False
        self._error = None

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            from domains.multimodal.speech import ServerSpeechRecognizer
            self._recognizer = ServerSpeechRecognizer()
            self._recognizer.load_model()
            self._loaded = True
            self._error = None
            backend = self._recognizer._backend or "none"
            logger.info("STT backend loaded: %s", backend, extra={"tag": "MODEL"})
            return self._recognizer._backend is not None
        except ImportError:
            self._error = "Speech recognition requires vosk or speech_recognition package"
            logger.warning("STT: no ASR backend available", extra={"tag": "MODEL"})
            return False
        except Exception as e:
            self._error = f"STT load failed: {e}"
            logger.warning("STT: load failed: %s", e, extra={"tag": "MODEL"})
            return False

    def recognize(self, audio_data: bytes, language: str = "en") -> dict:
        if not self._loaded:
            if not self.load():
                return {"text": "", "confidence": 0.0, "language": language, "is_valid": False}
        result = self._recognizer.recognize(audio_data, language=language)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "is_valid": result.is_valid,
        }


# ── Schema ──────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to convert to speech")
    voice: Optional[str] = Field(default=None, max_length=100, description="Voice identifier")


class TTSResponse(BaseModel):
    audio: str
    sample_rate: int
    duration_ms: int
    backend: str


class STTResponse(BaseModel):
    text: str
    confidence: float
    language: str
    is_valid: bool
    backend: str


class VoiceInfo(BaseModel):
    code: str
    name: str
    family: str


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]
    default: str


# ── Router ──────────────────────────────────────────────────────────────

class VoiceRouter:
    def __init__(self):
        self._tts_backend = _TTSBackend()
        self._stt_backend = _STTBackend()
        self.router = APIRouter(prefix="/voice", tags=["voice"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/tts", self.text_to_speech, methods=["POST"], response_model=TTSResponse)
        self.router.add_api_route("/stt", self.speech_to_text, methods=["POST"], response_model=STTResponse)
        self.router.add_api_route("/voices", self.list_voices, methods=["GET"], response_model=VoiceListResponse)
        self.router.add_api_route("/status", self.voice_status, methods=["GET"])

    async def text_to_speech(self, request: TTSRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> TTSResponse:
        """Convert text to speech audio."""
        try:
            if not request.text.strip():
                raise_error("No text provided", "E_BAD_REQUEST", status_code=400)

            _t0 = _time.monotonic()
            try:
                voice = request.voice or "eng"
                if self._tts_backend.load():
                    audio_bytes = await asyncio.to_thread(self._tts_backend.generate, request.text, voice)
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

        except Exception as e:
            classify_and_raise(e, source="voice.text_to_speech")

    async def speech_to_text(
        self,
        audio: UploadFile = File(...),
        language: str = Form(default="en"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> STTResponse:
        """Convert uploaded audio to text."""
        try:
            audio_bytes = await audio.read()
            if not audio_bytes:
                raise_error("No audio data provided", "E_BAD_REQUEST", status_code=400)

            _t0 = _time.monotonic()
            result = await asyncio.to_thread(self._stt_backend.recognize, audio_bytes, language)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000

            logger.info("STT completed in %.1fms (valid=%s)", _elapsed_ms, result["is_valid"])
            safe_audit_log(
                "voice.stt",
                resource=audio.filename or "upload",
                detail=f"elapsed={_elapsed_ms:.0f}ms valid={result['is_valid']}",
            )
            return STTResponse(
                text=result["text"],
                confidence=result["confidence"],
                language=result["language"],
                is_valid=result["is_valid"],
                backend=self._stt_backend._recognizer._backend or "none",
            )
        except Exception as e:
            classify_and_raise(e, source="voice.speech_to_text")

    async def list_voices(self) -> VoiceListResponse:
        """List available TTS voices/languages."""
        try:
            voices = [VoiceInfo(**v) for v in _POPULAR_VOICES]
            return VoiceListResponse(voices=voices, default="eng")
        except Exception as e:
            classify_and_raise(e, source="voice.list_voices")

    async def voice_status(self) -> dict:
        """Check if server-side TTS/STT models are available."""
        try:
            tts_available = self._tts_backend.load()
            stt_available = self._stt_backend.load()
            return success_response(data={
                "server_tts": tts_available,
                "tts_model": self._tts_backend._model_id if tts_available else None,
                "tts_error": self._tts_backend._error,
                "server_stt": stt_available,
                "stt_backend": self._stt_backend._recognizer._backend if stt_available else None,
                "stt_error": self._stt_backend._error,
            })
        except Exception as e:
            classify_and_raise(e, source="voice.status")


router = VoiceRouter().router
