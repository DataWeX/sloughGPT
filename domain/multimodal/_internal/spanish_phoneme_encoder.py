"""Spanish phonemes — re-export from voice._internal.spanish_phoneme_encoder (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.spanish_phoneme_encoder import (  # noqa: F401
    BOS,
    EOS,
    NUM_SPANISH_PHONEMES,
    PAD,
    SILENCE,
    SPACE,
    SPANISH_CONSONANTS,
    SPANISH_ID_TO_PHONEME,
    SPANISH_VOWELS,
    SpanishPhonemeEncoder,
    spanish_text_to_phonemes,
)

__all__ = [
    "BOS",
    "EOS",
    "NUM_SPANISH_PHONEMES",
    "PAD",
    "SILENCE",
    "SPACE",
    "SPANISH_CONSONANTS",
    "SPANISH_ID_TO_PHONEME",
    "SPANISH_VOWELS",
    "SpanishPhonemeEncoder",
    "spanish_text_to_phonemes",
]
