"""Speech Recognition — re-export from voice.speech (single source of truth)."""

from domain.voice._internal.speech import (  # noqa: F401
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    SpeechRecognizer,
    TranscriptionResult,
    get_speech_recognizer,
)

__all__ = [
    "TranscriptionResult",
    "SpeechRecognizer",
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "get_speech_recognizer",
]
