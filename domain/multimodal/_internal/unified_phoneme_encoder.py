"""Unified phoneme encoder — re-export from voice._internal.unified_phoneme_encoder (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.unified_phoneme_encoder import (  # noqa: F401
    UnifiedPhonemeEncoder,
    detect_language,
)

__all__ = [
    "UnifiedPhonemeEncoder",
    "detect_language",
]
