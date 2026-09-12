"""Voice layer - TTS, STT, phoneme encoding, audio processing.

Nothing in this package knows about HTTP, chat schemas, or tasks.

Usage:
    from voice import TTSEngine, get_speech_recognizer, UnifiedPhonemeEncoder

    tts = TTSEngine()
    waveform = tts.text_to_waveform("Hello world")

    recognizer = get_speech_recognizer(use_server=False)
    result = recognizer.recognize(audio_bytes, "en")

    encoder = UnifiedPhonemeEncoder()
    ids = encoder.encode("Hello world")
"""

from __future__ import annotations

from voice._internal.speech import (
    TranscriptionResult,
    SpeechRecognizer,
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    get_speech_recognizer,
)
from voice._internal.audio_filter import (
    FilterMode,
    AudioFilterConfig,
    FilterResult,
    apply_noise_gate,
    apply_agc,
    normalize_loudness,
    detect_voice_activity,
    apply_audio_filter,
)
from voice._internal.phoneme_encoder import (
    PhonemeEncoder,
    NUM_PHONEMES,
    text_to_phonemes,
)
from voice._internal.unified_phoneme_encoder import (
    UnifiedPhonemeEncoder,
    detect_language,
)
from voice._internal.tts import (
    parse_ssml,
    SpectrogramDecoder,
    GriffinLimVocoder,
    TTSEngine,
)

__all__ = [
    # Speech recognition
    "TranscriptionResult",
    "SpeechRecognizer",
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "get_speech_recognizer",
    # Audio filtering
    "FilterMode",
    "AudioFilterConfig",
    "FilterResult",
    "apply_noise_gate",
    "apply_agc",
    "normalize_loudness",
    "detect_voice_activity",
    "apply_audio_filter",
    # Phoneme encoding
    "PhonemeEncoder",
    "NUM_PHONEMES",
    "text_to_phonemes",
    "UnifiedPhonemeEncoder",
    "detect_language",
    # TTS
    "parse_ssml",
    "SpectrogramDecoder",
    "GriffinLimVocoder",
    "TTSEngine",
]
