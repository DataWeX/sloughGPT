"""VoiceEngine — unified feature facade for voice capabilities.

Wraps TTS, speech recognition, phoneme encoding, and audio filtering
into a single cohesive API. Routers should use this, not import from
domain.voice._internal directly.

Usage:
    from domain.voice.engine import VoiceEngine

    engine = VoiceEngine()
    waveform = engine.synthesize("Hello world")
    result = engine.recognize(audio_bytes, "en")
    phonemes = engine.encode_phonemes("Hello world")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.voice.engine")


@dataclass
class VoiceResult:
    """Result from a voice operation."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceEngine:
    """Unified voice feature API.

    Provides a single entry point for all voice-related operations:
    - TTS (text-to-speech)
    - STT (speech-to-text)
    - Phoneme encoding
    - Language detection
    - Audio filtering
    """

    def __init__(self) -> None:
        self._tts: Any = None
        self._recognizer: Any = None
        self._encoder: Any = None

    def _get_tts(self) -> Any:
        if self._tts is None:
            from domain.voice._internal.tts import TTSEngine

            self._tts = TTSEngine()
        return self._tts

    def _get_recognizer(self) -> Any:
        if self._recognizer is None:
            from domain.voice._internal.speech import get_speech_recognizer

            self._recognizer = get_speech_recognizer(use_server=False)
        return self._recognizer

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from domain.voice._internal.unified_phoneme_encoder import UnifiedPhonemeEncoder

            self._encoder = UnifiedPhonemeEncoder()
        return self._encoder

    def synthesize(self, text: str, voice: str | None = None) -> VoiceResult:
        """Convert text to speech waveform.

        Args:
            text: The text to synthesize.
            voice: Optional voice preset name.

        Returns:
            VoiceResult with waveform data in data field.
        """
        try:
            tts = self._get_tts()
            waveform = tts.text_to_waveform(text)
            return VoiceResult(
                success=True,
                data=waveform,
                metadata={"text": text, "voice": voice, "samples": len(waveform)},
            )
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return VoiceResult(success=False, error=str(e))

    def recognize(self, audio_bytes: bytes, language: str = "en") -> VoiceResult:
        """Convert speech audio to text.

        Args:
            audio_bytes: Raw audio data.
            language: Language code (default: "en").

        Returns:
            VoiceResult with transcription text in data field.
        """
        try:
            recognizer = self._get_recognizer()
            result = recognizer.recognize(audio_bytes, language)
            return VoiceResult(
                success=True,
                data=result.text if hasattr(result, "text") else result,
                metadata={"language": language},
            )
        except Exception as e:
            logger.error("Speech recognition failed: %s", e)
            return VoiceResult(success=False, error=str(e))

    def encode_phonemes(self, text: str, language: str | None = None) -> VoiceResult:
        """Encode text to phoneme IDs.

        Args:
            text: The text to encode.
            language: Optional language hint.

        Returns:
            VoiceResult with list of phoneme IDs in data field.
        """
        try:
            encoder = self._get_encoder()
            if language:
                ids = encoder.encode(text, language=language)
            else:
                ids = encoder.encode(text)
            return VoiceResult(
                success=True,
                data=ids,
                metadata={"text": text, "language": language, "count": len(ids)},
            )
        except Exception as e:
            logger.error("Phoneme encoding failed: %s", e)
            return VoiceResult(success=False, error=str(e))

    def detect_language(self, text: str) -> VoiceResult:
        """Detect the language of input text.

        Args:
            text: Text to analyze.

        Returns:
            VoiceResult with detected language code in data field.
        """
        try:
            from domain.voice._internal.unified_phoneme_encoder import detect_language

            lang = detect_language(text)
            return VoiceResult(
                success=True,
                data=lang,
                metadata={"text": text},
            )
        except Exception as e:
            logger.error("Language detection failed: %s", e)
            return VoiceResult(success=False, error=str(e))

    def status(self) -> dict[str, Any]:
        """Get voice engine status.

        Returns:
            Dict with capability status.
        """
        return {
            "tts": self._tts is not None,
            "recognizer": self._recognizer is not None,
            "encoder": self._encoder is not None,
            "capabilities": ["tts", "stt", "phonemes", "language_detection"],
        }


_engine: VoiceEngine | None = None


def get_voice_engine() -> VoiceEngine:
    """Singleton accessor for VoiceEngine."""
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine
