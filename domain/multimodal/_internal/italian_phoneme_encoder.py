"""Italian phonemes — re-export from voice._internal.italian_phoneme_encoder (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.italian_phoneme_encoder import (  # noqa: F401
    BOS,
    EOS,
    ITALIAN_CONSONANTS,
    ITALIAN_ID_TO_PHONEME,
    ITALIAN_VOWELS,
    NUM_ITALIAN_PHONEMES,
    PAD,
    SILENCE,
    SPACE,
    ItalianPhonemeEncoder,
    italian_text_to_phonemes,
)

__all__ = [
    "BOS",
    "EOS",
    "ITALIAN_CONSONANTS",
    "ITALIAN_ID_TO_PHONEME",
    "ITALIAN_VOWELS",
    "NUM_ITALIAN_PHONEMES",
    "PAD",
    "SILENCE",
    "SPACE",
    "ItalianPhonemeEncoder",
    "italian_text_to_phonemes",
]
