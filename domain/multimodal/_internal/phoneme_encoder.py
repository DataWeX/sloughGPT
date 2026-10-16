"""Phoneme encoder — re-export from voice._internal.phoneme_encoder (single source of truth).

Re-export shim: domain.voice is canonical for audio/phoneme (card d6e498ad —
the fork reconciliation; precedent set by this package's speech.py). Do not
add implementations here; change them in voice so every consumer sees one fix.
"""

from domain.voice._internal.phoneme_encoder import (  # noqa: F401
    BOS,
    CONSONANTS,
    DIGRAPHS,
    EOS,
    ID_TO_PHONEME,
    NUM_PHONEMES,
    PAD,
    SILENCE,
    SPACE,
    VOWELS,
    PhonemeEncoder,
    text_to_phonemes,
)

__all__ = [
    "BOS",
    "CONSONANTS",
    "DIGRAPHS",
    "EOS",
    "ID_TO_PHONEME",
    "NUM_PHONEMES",
    "PAD",
    "SILENCE",
    "SPACE",
    "VOWELS",
    "PhonemeEncoder",
    "text_to_phonemes",
]
