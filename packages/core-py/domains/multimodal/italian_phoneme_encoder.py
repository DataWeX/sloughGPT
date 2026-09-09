"""
Italian Phoneme Encoder — rule-based grapheme-to-phoneme for Italian TTS.

Maps Italian text to a compact phoneme vocabulary. Based on ARPAbet with
Italian-specific additions.

Phoneme set based on IPA/ARPAbet hybrid for Italian.
"""

from __future__ import annotations

import re
import numpy as np


# ── Italian Phoneme inventory ──────────────────────────────────────────────

# Vowels
ITALIAN_VOWELS = {
    "AA": 0,   # padre (father)
    "AE": 1,   # gatto (cat)
    "AH": 2,   # but (butterfly)
    "AO": 3,   # boca (mouth)
    "AW": 4,   # auto (car)
    "AY": 5,   # aire (air)
    "EH": 6,   # bebe (baby)
    "ER": 7,   # comer (to eat)
    "EY": 8,   # pez (fish)
    "IH": 9,   # si (if)
    "IY": 10,  # chica (girl)
    "OW": 11,  # oso (bear)
    "OY": 12,  # oye (listen)
    "UH": 13,  #乌拉圭 (Uruguay)
    "UW": 14,  #乌龟 (turtle)
    "IX": 15,  # Italian specific
}

# Consonants
ITALIAN_CONSONANTS = {
    "B": 16,   # buono (good)
    "CH": 17,  # chiesa (church)
    "D": 18,   # dire (to say)
    "DH": 19,  # this (English loan)
    "F": 20,   # fare (to do)
    "G": 21,   # gatto (cat)
    "GH": 22,  # ghe (them)
    "HH": 23,  # house (English loan)
    "JH": 24,  # Joy (English loan)
    "K": 25,   # casa (house)
    "L": 26,   # luna (moon)
    "M": 27,   # mama (mom)
    "N": 28,   # nino (boy)
    "NG": 29,  # singing (English loan)
    "NY": 30,  # gn (gnomo)
    "P": 31,   # pane (bread)
    "Q": 32,   # quadro (picture)
    "R": 33,   # rosso (red)
    "S": 34,   # sole (sun)
    "SC": 35,  # scuola (school)
    "SH": 36,  # shoe (English loan)
    "T": 37,   # terra (earth)
    "TH": 38,  # think (English loan)
    "TS": 39,  # zeta (pizza)
    "V": 40,   # vino (wine)
    "W": 41,  # yes (English loan)
    "Y": 42,   # yacht (English loan)
    "Z": 43,   # zero
    "ZH": 44,  # vision (English loan)
}

# Special tokens
PAD = 45
BOS = 46
EOS = 47
SPACE = 48
SILENCE = 49

NUM_ITALIAN_PHONEMES = 50

# Reverse lookup
ITALIAN_ID_TO_PHONEME = {v: k for k, v in {**ITALIAN_VOWELS, **ITALIAN_CONSONANTS}.items()}
ITALIAN_ID_TO_PHONEME[PAD] = "_"
ITALIAN_ID_TO_PHONEME[BOS] = "<"
ITALIAN_ID_TO_PHONEME[EOS] = ">"
ITALIAN_ID_TO_PHONEME[SPACE] = " "
ITALIAN_ID_TO_PHONEME[SILENCE] = "-"

# Phoneme-to-grapheme mapping for decode
ITALIAN_PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Vowels
    "AA": "a", "AE": "a", "AH": "u", "AO": "o",
    "AW": "au", "AY": "ai", "EH": "e", "ER": "er",
    "EY": "e", "IH": "i", "IY": "i", "OW": "o",
    "OY": "oy", "UH": "u", "UW": "u",
    "IX": "i",
    # Consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "th",
    "F": "f", "G": "g", "GH": "gh", "HH": "h",
    "JH": "j", "K": "c", "L": "l", "M": "m",
    "N": "n", "NG": "ng", "NY": "gn", "P": "p",
    "Q": "qu", "R": "r", "S": "s", "SC": "sc",
    "SH": "sh", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}


# ── Italian Grapheme-to-Phoneme rules ─────────────────────────────────────

_ITALIAN_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "io": ["IY", "OW"],
    "tu": ["T", "UW"],
    "lui": ["L", "UW", "IY"],
    "lei": ["L", "AY"],
    "noi": ["N", "OW", "IY"],
    "voi": ["V", "OW", "IY"],
    "loro": ["L", "OW", "R", "OW"],
    "sono": ["S", "OW", "N", "OW"],
    "sei": ["S", "AY"],
    "e": ["EH"],
    "siamo": ["S", "IY", "AH", "M", "OW"],
    "siete": ["S", "IY", "EH", "T", "EH"],
    "hanno": ["HH", "AH", "N", "N", "OW"],
    "ho": ["HH", "OW"],
    "hai": ["HH", "AY"],
    "non": ["N", "OW", "N"],
    "si": ["S", "IY"],
    "no": ["N", "OW"],
    "ciao": ["CH", "IY", "AW"],
    "buongiorno": ["B", "UW", "OW", "NG", "JH", "OW", "R", "N", "OW"],
    "buonasera": ["B", "UW", "OW", "N", "AH", "S", "EH", "R", "AH"],
    "grazie": ["G", "R", "AH", "TS", "IY", "EH"],
    "prego": ["P", "R", "EH", "G", "OW"],
    "per favore": ["P", "EH", "R", "F", "AH", "V", "OW", "R", "EH"],
    "si": ["S", "IY"],
    "no": ["N", "OW"],
    "bene": ["B", "EH", "N", "EH"],
    "male": ["M", "AH", "L", "EH"],
    "grande": ["G", "R", "AH", "N", "D", "EH"],
    "piccolo": ["P", "IY", "K", "K", "OW", "L", "OW"],
    "nuovo": ["N", "UW", "OW", "V", "OW"],
    "vecchio": ["V", "EH", "K", "K", "IY", "OW"],
    "casa": ["K", "AH", "S", "AH"],
    "scuola": ["S", "K", "UW", "OW", "L", "AH"],
    "libro": ["L", "IY", "B", "R", "OW"],
    "gatto": ["G", "AH", "T", "T", "OW"],
    "giorno": ["JH", "OW", "R", "N", "OW"],
    "notte": ["N", "OW", "T", "T", "EH"],
    "tempo": ["T", "EH", "M", "P", "OW"],
    "ora": ["OW", "R", "AH"],
    "lunedì": ["L", "UW", "N", "EH", "D", "IY"],
    "martedì": ["M", "AA", "R", "T", "EH", "D", "IY"],
    "mercoledì": ["M", "EH", "R", "K", "OW", "L", "EH", "D", "IY"],
    "giovedì": ["JH", "OW", "V", "EH", "D", "IY"],
    "venerdì": ["V", "EH", "N", "EH", "R", "D", "IY"],
    "sabato": ["S", "AH", "B", "AH", "T", "OW"],
    "domenica": ["D", "OW", "M", "EH", "N", "IY", "K", "AH"],
    "uno": ["UW", "N", "OW"],
    "due": ["D", "UW", "EH"],
    "tre": ["T", "R", "EH"],
    "quattro": ["K", "UW", "AH", "T", "T", "R", "OW"],
    "cinque": ["CH", "IY", "NG", "K", "UW", "EH"],
    "sei": ["S", "AY"],
    "sette": ["S", "EH", "T", "T", "EH"],
    "otto": ["OW", "T", "T", "OW"],
    "nove": ["N", "OW", "V", "EH"],
    "dieci": ["D", "IY", "EH", "CH", "IY"],
}


def _italian_apply_rules(word: str) -> list[str]:
    """Apply Italian letter-to-sound rules."""
    result: list[str] = []
    i = 0
    w = word.lower()

    # Italian digraphs/trigraphs
    italian_digraphs = [
        ("ch", ["K"]),
        ("gh", ["G"]),
        ("gl", ["L"]),
        ("gn", ["NY"]),
        ("sc", ["S", "K"]),
        ("ce", ["CH", "EH"]),
        ("ci", ["CH", "IY"]),
        ("ge", ["JH", "EH"]),
        ("gi", ["JH", "IY"]),
        ("qu", ["K", "UW"]),
    ]

    while i < len(w):
        matched = False

        # Try longest match first (2 chars)
        for length in (2,):
            chunk = w[i:i + length]
            for pattern, sounds in italian_digraphs:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            # Italian single letter rules
            italian_letter_rules = {
                "a": ["AH"], "b": ["B"], "c": ["K"],
                "d": ["D"], "e": ["EH"], "f": ["F"],
                "g": ["G"], "h": [], "i": ["IY"],
                "l": ["L"], "m": ["M"], "n": ["N"],
                "o": ["OW"], "p": ["P"], "q": ["K"],
                "r": ["R"], "s": ["S"], "t": ["T"],
                "u": ["UW"], "v": ["V"], "z": ["TS"],
            }
            if ch in italian_letter_rules:
                result.extend(italian_letter_rules[ch])
            i += 1

    return result


def italian_text_to_phonemes(text: str) -> list[str]:
    """Convert Italian text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [ITALIAN_ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-Zàâäéèêëïîôùûüÿçœæ]+|[^a-zA-Zàâäéèêëïîôùûüÿçœæ]+", text)

    for token in tokens:
        if token.isspace():
            result.append(ITALIAN_ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _ITALIAN_PRONUNCIATION_DICT:
                result.extend(_ITALIAN_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_italian_apply_rules(token))
            result.append(ITALIAN_ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(ITALIAN_ID_TO_PHONEME[SILENCE])

    result.append(ITALIAN_ID_TO_PHONEME[EOS])
    return result


class ItalianPhonemeEncoder:
    """Encodes Italian text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **ITALIAN_VOWELS, **ITALIAN_CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert Italian text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = italian_text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to Italian text string.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
        Returns:
            Decoded text string
        """
        id_list = ids.flatten().tolist()
        # Strip BOS and EOS
        id_list = [i for i in id_list if i not in (BOS, EOS, PAD)]
        # Map IDs to graphemes
        parts = []
        for pid in id_list:
            phoneme = ITALIAN_ID_TO_PHONEME.get(pid, "?")
            grapheme = ITALIAN_PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
            parts.append(grapheme)
        return "".join(parts)

    def decode_phonemes(self, ids: np.ndarray) -> list[str]:
        """Convert phoneme ID array to phoneme string list.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
        Returns:
            List of phoneme strings (BOS/EOS/PAD stripped)
        """
        id_list = ids.flatten().tolist()
        # Strip BOS, EOS, PAD
        id_list = [i for i in id_list if i not in (BOS, EOS, PAD)]
        return [ITALIAN_ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input Italian text string
        Returns:
            Multi-line string showing the encoding pipeline
        """
        ids = self.encode(text)
        phonemes = self.decode_phonemes(ids)
        decoded = self.decode(ids)

        lines = [
            f"Input:     {text!r}",
            f"Phonemes:  {' '.join(phonemes)}",
            f"IDs:       {ids.flatten().tolist()}",
            f"Decoded:   {decoded!r}",
        ]
        return "\n".join(lines)

    @property
    def vocab_size(self) -> int:
        return NUM_ITALIAN_PHONEMES
