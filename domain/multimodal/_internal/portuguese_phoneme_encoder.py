"""Portuguese phonemes — re-export from voice._internal.portuguese_phoneme_encoder (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.portuguese_phoneme_encoder import (  # noqa: F401
    BOS,
    EOS,
    NUM_PORTUGUESE_PHONEMES,
    PAD,
    PORTUGUESE_CONSONANTS,
    PORTUGUESE_ID_TO_PHONEME,
    PORTUGUESE_VOWELS,
    SILENCE,
    SPACE,
    PortuguesePhonemeEncoder,
    portuguese_text_to_phonemes,
)

__all__ = [
    "BOS",
    "EOS",
    "NUM_PORTUGUESE_PHONEMES",
    "PAD",
    "PORTUGUESE_CONSONANTS",
    "PORTUGUESE_ID_TO_PHONEME",
    "PORTUGUESE_VOWELS",
    "SILENCE",
    "SPACE",
    "PortuguesePhonemeEncoder",
    "portuguese_text_to_phonemes",
]
