"""multimodal — Vision, video, diffusion, image captioning, and AUDIO
(TTS, speech recognition, phoneme encoding).

All audio/phoneme code was absorbed from the former domain.voice package
(card d6e498ad ruling: voice is a product surface served by routers/voice.py
and routers/phoneme.py over this package — it is not a domain of its own).

Public API:
    MultiModalConfig, MultimodalCapabilities, MultimodalManager
    get_multimodal_manager, initialize_multimodal
    TTSEngine, VoiceEngine, get_voice_engine, PhonemeEngine, get_phoneme_engine
    UnifiedPhonemeEncoder, PhonemeEncoder, get_speech_recognizer, ...
"""

from domain.multimodal._internal.audio_filter import (
    AudioFilterConfig,
    FilterMode,
    FilterResult,
    apply_agc,
    apply_audio_filter,
    apply_noise_gate,
    detect_voice_activity,
    normalize_loudness,
)
from domain.multimodal._internal.config import MultiModalConfig
from domain.multimodal._internal.diffusion import LatentDiffusionModel
from domain.multimodal._internal.engine import ReplayBuffer
from domain.multimodal._internal.manager import (
    MultimodalCapabilities,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)
from domain.multimodal._internal.phoneme import PhonemeEngine, get_phoneme_engine
from domain.multimodal._internal.phoneme_encoder import (
    NUM_PHONEMES,
    PhonemeEncoder,
    text_to_phonemes,
)
from domain.multimodal._internal.speech import (
    BrowserSpeechRecognizer,
    ServerSpeechRecognizer,
    SpeechRecognizer,
    TranscriptionResult,
    get_speech_recognizer,
)
from domain.multimodal._internal.text_encoder import TextEncoder
from domain.multimodal._internal.tts import (
    GriffinLimVocoder,
    SpectrogramDecoder,
    TTSEngine,
    parse_ssml,
)
from domain.multimodal._internal.unified_phoneme_encoder import (
    UnifiedPhonemeEncoder,
    detect_language,
)
from domain.multimodal._internal.vae import SloVAE
from domain.multimodal._internal.video import VideoProcessor
from domain.multimodal._internal.voice_engine import VoiceEngine, get_voice_engine

__all__ = [
    # Manager / vision / video / diffusion
    "LatentDiffusionModel",
    "MultiModalConfig",
    "MultimodalCapabilities",
    "MultimodalManager",
    "ReplayBuffer",
    "SloVAE",
    "TextEncoder",
    "VideoProcessor",
    "get_multimodal_manager",
    "initialize_multimodal",
    # Audio: TTS
    "GriffinLimVocoder",
    "SpectrogramDecoder",
    "TTSEngine",
    "parse_ssml",
    # Audio: speech recognition
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "SpeechRecognizer",
    "TranscriptionResult",
    "get_speech_recognizer",
    # Audio: phoneme encoding
    "NUM_PHONEMES",
    "PhonemeEncoder",
    "UnifiedPhonemeEncoder",
    "detect_language",
    "text_to_phonemes",
    # Audio: filters
    "AudioFilterConfig",
    "FilterMode",
    "FilterResult",
    "apply_agc",
    "apply_audio_filter",
    "apply_noise_gate",
    "detect_voice_activity",
    "normalize_loudness",
    # Engines
    "PhonemeEngine",
    "VoiceEngine",
    "get_phoneme_engine",
    "get_voice_engine",
]
