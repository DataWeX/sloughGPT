"""
Portuguese Phoneme Encoder — rule-based grapheme-to-phoneme for Portuguese TTS.

Maps Portuguese text to a compact phoneme vocabulary. Based on ARPAbet with
Portuguese-specific additions.

Phoneme set based on IPA/ARPAbet hybrid for Portuguese.
"""

from __future__ import annotations

import re
import numpy as np


# ── Portuguese Phoneme inventory ──────────────────────────────────────────

# Vowels
PORTUGUESE_VOWELS = {
    "AA": 0,   # pai (father)
    "AE": 1,   # Feedback (cat)
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
    "UH": 13,  # uruguai (Uruguay)
    "UW": 14,  # tartaruga (turtle)
    "IX": 15,  # Portuguese specific
}

# Consonants
PORTUGUESE_CONSONANTS = {
    "B": 16,   # bombo (good)
    "CH": 17,  # church (church)
    "D": 18,   # dia (to say)
    "DH": 19,  # this (English loan)
    "F": 20,   # fazer (to do)
    "G": 21,   # gato (cat)
    "GH": 22,  # gue (them)
    "HH": 23,  # house (English loan)
    "JH": 24,  # Joy (English loan)
    "K": 25,   # casa (house)
    "L": 26,   # lua (moon)
    "M": 27,   # mama (mom)
    "N": 28,   # nino (boy)
    "NG": 29,  # singing (English loan)
    "NH": 30,  # nha (Portuguese specific)
    "P": 31,   # pao (bread)
    "Q": 32,   # quadro (picture)
    "R": 33,   # vermelho (red)
    "S": 34,   # sol (sun)
    "SC": 35,  # escola (school)
    "SH": 36,  # shoe (English loan)
    "T": 37,   # terra (earth)
    "TH": 38,  # think (English loan)
    "TS": 39,  # pizza (Portuguese specific)
    "V": 40,   # vinho (wine)
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

NUM_PORTUGUESE_PHONEMES = 50

# Reverse lookup
PORTUGUESE_ID_TO_PHONEME = {v: k for k, v in {**PORTUGUESE_VOWELS, **PORTUGUESE_CONSONANTS}.items()}
PORTUGUESE_ID_TO_PHONEME[PAD] = "_"
PORTUGUESE_ID_TO_PHONEME[BOS] = "<"
PORTUGUESE_ID_TO_PHONEME[EOS] = ">"
PORTUGUESE_ID_TO_PHONEME[SPACE] = " "
PORTUGUESE_ID_TO_PHONEME[SILENCE] = "-"

# Phoneme-to-grapheme mapping for decode
PORTUGUESE_PHONEME_TO_GRAPHEME: dict[str, str] = {
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
    "N": "n", "NG": "ng", "NH": "nh", "P": "p",
    "Q": "qu", "R": "r", "S": "s", "SC": "sc",
    "SH": "sh", "T": "t", "TH": "th", "TS": "ts",
    "V": "v", "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}


# ── Portuguese Grapheme-to-Phoneme rules ─────────────────────────────────

_PORTUGUESE_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "eu": ["AY", "UW"],
    "tu": ["T", "UW"],
    "ele": ["EH", "L", "EH"],
    "ela": ["EH", "L", "AH"],
    "nos": ["N", "OW", "S"],
    "voce": ["V", "OW", "S", "EH"],
    "eles": ["EH", "L", "EH", "S"],
    "elas": ["EH", "L", "AH", "S"],
    "sou": ["S", "OW", "UW"],
    "eres": ["EH", "R", "EH", "S"],
    "e": ["EH"],
    "somos": ["S", "OW", "M", "OW", "S"],
    "sao": ["S", "AH", "W", "N"],
    "tenho": ["T", "EH", "NH", "OW"],
    "ten": ["T", "EH", "N"],
    "mae": ["M", "AH", "EH"],
    "nao": ["N", "AH", "W"],
    "sim": ["S", "IY"],
    "nao": ["N", "AH", "W"],
    "oi": ["OY"],
    "ola": ["OW", "L", "AH"],
    "obrigado": ["OW", "B", "R", "IY", "G", "AH", "D", "OW"],
    "obrigada": ["OW", "B", "R", "IY", "G", "AH", "D", "AH"],
    "por favor": ["P", "OW", "R", "F", "AH", "V", "OW", "R"],
    "bem": ["B", "EH", "M"],
    "mal": ["M", "AH", "L"],
    "grande": ["G", "R", "AH", "N", "D", "EH"],
    "pequeno": ["P", "EH", "K", "EH", "N", "OW"],
    "novo": ["N", "OW", "V", "OW"],
    "velho": ["V", "EH", "L", "Y", "OW"],
    "casa": ["K", "AH", "Z", "AH"],
    "escola": ["EH", "S", "K", "OW", "L", "AH"],
    "livro": ["L", "IY", "V", "R", "OW"],
    "gato": ["G", "AH", "T", "OW"],
    "dia": ["D", "IY", "AH"],
    "noite": ["N", "OY", "T", "EH"],
    "tempo": ["T", "EH", "M", "P", "OW"],
    "hora": ["OW", "R", "AH"],
    "segunda": ["S", "EH", "G", "UW", "N", "D", "AH"],
    "terca": ["T", "EH", "R", "S", "AH"],
    "quarta": ["K", "UW", "AH", "R", "T", "AH"],
    "quinta": ["K", "IY", "N", "T", "AH"],
    "sexta": ["S", "EH", "K", "S", "T", "AH"],
    "sabado": ["S", "AH", "B", "AH", "D", "OW"],
    "domingo": ["D", "OW", "M", "IY", "N", "G", "OW"],
    "um": ["UW", "M"],
    "dois": ["D", "OY", "S"],
    "tres": ["T", "R", "EH", "S"],
    "quatro": ["K", "UW", "AH", "T", "R", "OW"],
    "cinco": ["S", "IY", "N", "K", "OW"],
    "seis": ["S", "AY", "S"],
    "sete": ["S", "EH", "T", "EH"],
    "oito": ["OY", "T", "OW"],
    "nove": ["N", "OW", "V", "EH"],
    "dez": ["D", "EH", "Z"],
}


def _portuguese_apply_rules(word: str) -> list[str]:
    """Apply Portuguese letter-to-sound rules."""
    result: list[str] = []
    i = 0
    w = word.lower()

    # Portuguese digraphs/trigraphs
    portuguese_digraphs = [
        ("ch", ["CH"]),
        ("lh", ["L", "IY"]),
        ("nh", ["NH"]),
        ("gu", ["G", "UW"]),
        ("qu", ["K", "UW"]),
        ("ss", ["S"]),
        ("rr", ["R"]),
        ("te", ["T", "EH"]),
        ("ti", ["T", "IY"]),
        ("de", ["D", "EH"]),
        ("di", ["D", "IY"]),
        ("se", ["S", "EH"]),
        ("si", ["S", "IY"]),
        ("ce", ["S", "EH"]),
        ("ci", ["S", "IY"]),
        ("ge", ["ZH", "EH"]),
        ("gi", ["ZH", "IY"]),
    ]

    while i < len(w):
        matched = False

        # Try longest match first (2 chars)
        for length in (2,):
            chunk = w[i:i + length]
            for pattern, sounds in portuguese_digraphs:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            # Portuguese single letter rules
            portuguese_letter_rules = {
                "a": ["AH"], "b": ["B"], "c": ["K"],
                "d": ["D"], "e": ["EH"], "f": ["F"],
                "g": ["G"], "h": [], "i": ["IY"],
                "j": ["ZH"], "k": ["K"], "l": ["L"],
                "m": ["M"], "n": ["N"], "o": ["OW"],
                "p": ["P"], "q": ["K"], "r": ["R"],
                "s": ["S"], "t": ["T"], "u": ["UW"],
                "v": ["V"], "w": ["V"], "x": ["SH"],
                "y": ["IY"], "z": ["Z"],
            }
            if ch in portuguese_letter_rules:
                result.extend(portuguese_letter_rules[ch])
            i += 1

    return result


def portuguese_text_to_phonemes(text: str) -> list[str]:
    """Convert Portuguese text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [PORTUGUESE_ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-Zàâäéèêëïîôùûüÿçœæ]+|[^a-zA-Zàâäéèêëïîôùûüÿçœæ]+", text)

    for token in tokens:
        if token.isspace():
            result.append(PORTUGUESE_ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _PORTUGUESE_PRONUNCIATION_DICT:
                result.extend(_PORTUGUESE_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_portuguese_apply_rules(token))
            result.append(PORTUGUESE_ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(PORTUGUESE_ID_TO_PHONEME[SILENCE])

    result.append(PORTUGUESE_ID_TO_PHONEME[EOS])
    return result


class PortuguesePhonemeEncoder:
    """Encodes Portuguese text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **PORTUGUESE_VOWELS, **PORTUGUESE_CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert Portuguese text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = portuguese_text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to Portuguese text string.

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
            phoneme = PORTUGUESE_ID_TO_PHONEME.get(pid, "?")
            grapheme = PORTUGUESE_PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
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
        return [PORTUGUESE_ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input Portuguese text string
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
        return NUM_PORTUGUESE_PHONEMES
