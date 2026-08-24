"""
Speech Recognition Module

Speech-to-text using Web Speech API (browser) with server fallback.
"""

from typing import Optional, Protocol
from dataclasses import dataclass
import logging

logger = logging.getLogger("slo.speech")


@dataclass
class TranscriptionResult:
    """Result from speech recognition."""
    text: str
    confidence: float
    language: str
    duration: Optional[float] = None


class SpeechRecognizer(Protocol):
    """Protocol for speech recognition implementations."""

    def recognize(self, audio_data: bytes, language: str) -> TranscriptionResult:
        """Convert audio to text."""
        ...

    def recognize_stream(self, audio_chunk: bytes) -> TranscriptionResult:
        """Stream recognition."""
        ...


class BrowserSpeechRecognizer:
    """
    Client-side speech recognizer using Web Speech API.

    Used via JavaScript in browser - this is the protocol definition.
    """

    def __init__(self, language: str = "en-US"):
        self.language = language
        self.continuous = False
        self.interim_results = True

    def get_config(self) -> dict:
        """Return config for browser Web Speech API."""
        return {
            "language": self.language,
            "continuous": self.continuous,
            "interimResults": self.interim_results,
        }


class ServerSpeechRecognizer:
    """
    Server-side speech recognition.

    Uses an optional pure-Python ASR backend if installed. No model files
    are downloaded; when no backend is available the recognizer degrades
    gracefully to an empty transcription.
    """

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._backend = None

    def load_model(self):
        """Load an ASR backend if one is installed."""
        # 1) vosk — offline, pure-Python inference (model loaded from disk)
        try:
            import vosk  # type: ignore
            self._backend = "vosk"
            logger.info("Loaded vosk ASR backend", extra={"tag": "MODEL"})
            return
        except ImportError:
            pass

        # 2) speech_recognition — wrapper over system/offline engines
        try:
            import speech_recognition as sr  # type: ignore
            self._backend = "speech_recognition"
            self._model = sr.Recognizer()
            logger.info("Loaded speech_recognition backend", extra={"tag": "MODEL"})
            return
        except ImportError:
            pass

        logger.warning(
            "No ASR backend available (vosk/speech_recognition). "
            "Server speech recognition disabled; browser Web Speech API still works.",
            extra={"tag": "MODEL"},
        )
        self._backend = None

    def recognize(self, audio_data: bytes, language: str = "en") -> TranscriptionResult:
        """Recognize speech from audio bytes using the available backend."""
        if self._backend is None:
            self.load_model()

        if self._backend is None:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
            )

        try:
            if self._backend == "speech_recognition":
                from speech_recognition import AudioData
                audio = AudioData(audio_data, sample_rate=16000, sample_width=2)
                text = self._model.recognize_google(audio, language=language)
                return TranscriptionResult(text=text, confidence=0.9, language=language)
            if self._backend == "vosk":
                text = self._decode_vosk(audio_data)
                return TranscriptionResult(text=text, confidence=0.9, language=language)
        except Exception as e:
            logger.error("Speech recognition error: %s", e, extra={"tag": "MODEL"})
        return TranscriptionResult(
            text="",
            confidence=0.0,
            language=language,
        )

    def _decode_vosk(self, audio_data: bytes) -> str:
        """Decode raw 16-bit PCM audio bytes with vosk."""
        import json as _json
        import os

        try:
            from vosk import Model, KaldiRecognizer
        except ImportError:
            return ""

        if not isinstance(self._model, Model):
            model_path = os.environ.get("VOSK_MODEL_PATH") or (
                f"/usr/share/vosk-model-small-{self.model_name}"
            )
            if not os.path.isdir(model_path):
                logger.warning(
                    "vosk model not found at %s (set VOSK_MODEL_PATH)", model_path,
                    extra={"tag": "MODEL"},
                )
                return ""
            self._model = Model(model_path)
        recognizer = KaldiRecognizer(self._model, 16000)
        recognizer.AcceptWaveform(audio_data)
        result = _json.loads(recognizer.FinalResult())
        return result.get("text", "")


def get_speech_recognizer(
    use_server: bool = False,
    model_name: str = "base",
) -> SpeechRecognizer:
    """
    Get speech recognizer instance.

    Args:
        use_server: Use server-side model instead of browser
        model_name: Server model (whisper base/medium/large)

    Returns:
        SpeechRecognizer implementation
    """
    if use_server:
        return ServerSpeechRecognizer(model_name)
    return BrowserSpeechRecognizer()


__all__ = [
    "TranscriptionResult",
    "SpeechRecognizer",
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "get_speech_recognizer",
]
