"""TTS — the canonical implementation lives in voice (lazy slonet import keeps this module torch-free at import time).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.tts import (  # noqa: F401
    GriffinLimVocoder,
    SpectrogramDecoder,
    TTSEngine,
    parse_ssml,
)

__all__ = [
    "GriffinLimVocoder",
    "SpectrogramDecoder",
    "TTSEngine",
    "parse_ssml",
]
