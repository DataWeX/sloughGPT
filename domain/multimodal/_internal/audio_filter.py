"""Audio filtering — re-export from voice._internal.audio_filter (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.audio_filter import (  # noqa: F401
    AudioFilterConfig,
    FilterMode,
    FilterResult,
    apply_agc,
    apply_audio_filter,
    apply_noise_gate,
    detect_voice_activity,
    normalize_loudness,
)

__all__ = [
    "AudioFilterConfig",
    "FilterMode",
    "FilterResult",
    "apply_agc",
    "apply_audio_filter",
    "apply_noise_gate",
    "detect_voice_activity",
    "normalize_loudness",
]
