"""PhonemeEngine — phoneme encoding, scoring, pronunciation learning.

Lives under training because pronunciation is how the model learns to speak.
Links to VoiceEngine for TTS synthesis.

Usage:
    from domain.training.phoneme import PhonemeEngine

    engine = PhonemeEngine()
    result = engine.encode("hello world")
    scored = engine.score("hello", "hɛloʊ")
    audio = engine.synthesize("hɛloʊ")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.training.phoneme")


@dataclass
class PhonemeResult:
    """Result from a phoneme operation."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PhonemeEngine:
    """Phoneme domain engine — encoding, scoring, synthesis.

    Training concern: how the model learns to pronounce words.
    Links to VoiceEngine for TTS synthesis of phoneme sequences.
    """

    def __init__(self) -> None:
        self._encoder: Any = None
        self._voice: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from domain.voice._internal.unified_phoneme_encoder import UnifiedPhonemeEncoder

            self._encoder = UnifiedPhonemeEncoder()
        return self._encoder

    def _get_voice(self) -> Any:
        if self._voice is None:
            from domain.voice.engine import get_voice_engine

            self._voice = get_voice_engine()
        return self._voice

    # ── Encoding ──

    def encode(self, text: str, language: str | None = None) -> PhonemeResult:
        """Encode text to phoneme IDs."""
        try:
            encoder = self._get_encoder()
            ids = encoder.encode(text, language=language)
            phonemes = encoder.decode_phonemes(ids, language=encoder.current_language)
            return PhonemeResult(
                success=True,
                data={
                    "ids": ids.tolist() if hasattr(ids, "tolist") else list(ids),
                    "phonemes": phonemes,
                    "text": text,
                    "language": encoder.current_language,
                },
                metadata={"count": len(ids)},
            )
        except Exception as e:
            logger.error("Phoneme encode failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    def decode(self, ids: list[int], language: str | None = None) -> PhonemeResult:
        """Decode phoneme IDs back to text."""
        try:
            import numpy as np

            encoder = self._get_encoder()
            arr = np.array(ids)
            text = encoder.decode(arr, language=language)
            return PhonemeResult(
                success=True,
                data={"text": text, "ids": ids},
            )
        except Exception as e:
            logger.error("Phoneme decode failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    def visualize(self, text: str, language: str | None = None) -> PhonemeResult:
        """Get a visual representation of phoneme encoding."""
        try:
            encoder = self._get_encoder()
            viz = encoder.visualize(text, language=language)
            return PhonemeResult(
                success=True,
                data={"visualization": viz, "text": text},
            )
        except Exception as e:
            logger.error("Phoneme visualize failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    # ── Scoring ──

    def score(self, target: str, spoken: str, language: str | None = None) -> PhonemeResult:
        """Score pronunciation accuracy."""
        try:
            encoder = self._get_encoder()
            result = encoder.score_pronunciation(target, spoken, language=language)
            return PhonemeResult(
                success=True,
                data=result,
                metadata={"target": target, "spoken": spoken},
            )
        except Exception as e:
            logger.error("Phoneme score failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    # ── Batch ──

    def batch_encode(self, texts: list[str], language: str | None = None) -> PhonemeResult:
        """Encode multiple texts to phoneme IDs."""
        try:
            encoder = self._get_encoder()
            results = encoder.encode_batch(texts, language=language)
            return PhonemeResult(
                success=True,
                data=results,
                metadata={"count": len(texts)},
            )
        except Exception as e:
            logger.error("Phoneme batch encode failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    # ── Language detection ──

    def detect_language(self, text: str) -> PhonemeResult:
        """Detect the language of input text."""
        try:
            encoder = self._get_encoder()
            lang = encoder.detect_language(text)
            return PhonemeResult(
                success=True,
                data={"language": lang, "text": text},
            )
        except Exception as e:
            logger.error("Language detection failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    # ── TTS synthesis via voice engine ──

    def synthesize(self, text: str) -> PhonemeResult:
        """Synthesize text to speech via the voice engine."""
        try:
            voice = self._get_voice()
            return PhonemeResult(
                success=True,
                data=voice.synthesize(text).data,
                metadata={"text": text, "source": "voice_engine"},
            )
        except Exception as e:
            logger.error("Phoneme synthesis failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    # ── Info ──

    def supported_languages(self) -> PhonemeResult:
        """List supported languages."""
        try:
            encoder = self._get_encoder()
            langs = encoder.supported_languages()
            return PhonemeResult(success=True, data=langs)
        except Exception as e:
            logger.error("Supported languages failed: %s", e)
            return PhonemeResult(success=False, error=str(e))

    def status(self) -> dict[str, Any]:
        """Get phoneme engine status."""
        return {
            "encoder": self._encoder is not None,
            "voice_linked": self._voice is not None,
            "capabilities": ["encode", "decode", "score", "batch", "detect", "synthesize"],
        }


_phoneme_engine: PhonemeEngine | None = None


def get_phoneme_engine() -> PhonemeEngine:
    """Singleton accessor for PhonemeEngine."""
    global _phoneme_engine
    if _phoneme_engine is None:
        _phoneme_engine = PhonemeEngine()
    return _phoneme_engine
