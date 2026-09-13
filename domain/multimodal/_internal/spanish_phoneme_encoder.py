"""
Spanish Phoneme Encoder — rule-based grapheme-to-phoneme for Spanish TTS.

Maps Spanish text to a compact phoneme vocabulary. Based on ARPAbet with
Spanish-specific additions.

Phoneme set based on IPA/ARPAbet hybrid for Spanish.
"""

from __future__ import annotations

import re
import numpy as np


# ── Spanish Phoneme inventory ──────────────────────────────────────────────

# Vowels
SPANISH_VOWELS = {
    "AA": 0,   # padre (father)
    "AE": 1,   # gato (cat)
    "AH": 2,   #-but (butterfly)
    "AO": 3,   # boca (mouth)
    "AW": 4,   #auto (car)
    "AY": 5,   #aire (air)
    "EH": 6,   # bebe (baby)
    "ER": 7,   # comer (to eat)
    "EY": 8,   # pez (fish)
    "IH": 9,   # si (if)
    "IY": 10,  # chica (girl)
    "OW": 11,  # oso (bear)
    "OY": 12,  # oye (listen)
    "UH": 13,  #乌拉圭 (Uruguay)
    "UW": 14,  #乌龟 (turtle)
    "UX": 15,  #乌龟 (turtle)
}

# Consonants
SPANISH_CONSONANTS = {
    "B": 16,   # bueno (good)
    "CH": 17,  # chico (boy)
    "D": 18,   # bueno (good)
    "DH": 19,  # this (English loan)
    "F": 20,   # fan (fan)
    "G": 21,   # gato (cat)
    "HH": 22,  # house (English loan)
    "JH": 23,  # Joy (English loan)
    "K": 24,   # casa (house)
    "L": 25,   # luna (moon)
    "M": 26,   # mama (mom)
    "N": 27,   # nino (boy)
    "NG": 28,  # singing (English loan)
    "NY": 29,  # nino (boy)
    "P": 30,   # pan (bread)
    "R": 31,   # pero (but)
    "RR": 32,  # perro (dog)
    "S": 33,   # sol (sun)
    "SH": 34,  # shoe (English loan)
    "T": 35,   # casa (house)
    "TH": 36,  # think (English loan)
    "V": 37,   # vino (wine)
    "W": 38,  # yes (English loan)
    "Y": 39,   #ll (llama)
    "Z": 40,   #zapato (shoe)
    "ZH": 41,  # vision (English loan)
}

# Special tokens
PAD = 42
BOS = 43
EOS = 44
SPACE = 45
SILENCE = 46

NUM_SPANISH_PHONEMES = 47

# Reverse lookup
SPANISH_ID_TO_PHONEME = {v: k for k, v in {**SPANISH_VOWELS, **SPANISH_CONSONANTS}.items()}
SPANISH_ID_TO_PHONEME[PAD] = "_"
SPANISH_ID_TO_PHONEME[BOS] = "<"
SPANISH_ID_TO_PHONEME[EOS] = ">"
SPANISH_ID_TO_PHONEME[SPACE] = " "
SPANISH_ID_TO_PHONEME[SILENCE] = "-"

# Phoneme-to-grapheme mapping for decode
SPANISH_PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Vowels
    "AA": "a", "AE": "a", "AH": "u", "AO": "o",
    "AW": "au", "AY": "ai", "EH": "e", "ER": "er",
    "EY": "e", "IH": "i", "IY": "i", "OW": "o",
    "OY": "oy", "UH": "u", "UW": "u",
    "UX": "u",
    # Consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "th",
    "F": "f", "G": "g", "HH": "h", "JH": "j",
    "K": "c", "L": "l", "M": "m", "N": "n",
    "NG": "ng", "NY": "n", "P": "p", "R": "r",
    "RR": "rr", "S": "s", "SH": "sh", "T": "t",
    "TH": "th", "V": "v", "W": "w", "Y": "ll",
    "Z": "z", "ZH": "zh",
}


# ── Spanish Grapheme-to-Phoneme rules ─────────────────────────────────────

_SPANISH_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "yo": ["Y", "OW"],
    "tu": ["T", "UW"],
    "el": ["EH", "L"],
    "ella": ["EH", "Y", "AH"],
    "nosotros": ["N", "OW", "S", "OW", "T", "R", "OW", "S"],
    "vosotros": ["V", "OW", "S", "OW", "T", "R", "OW", "S"],
    "ellos": ["EH", "Y", "OW", "S"],
    "ellas": ["EH", "Y", "AH", "S"],
    "soy": ["S", "OW", "I"],
    "eres": ["EH", "R", "EH", "S"],
    "es": ["EH", "S"],
    "somos": ["S", "OW", "M", "OW", "S"],
    "sois": ["S", "OW", "I", "S"],
    "son": ["S", "OW", "N"],
    "tengo": ["T", "EH", "NG", "G", "OW"],
    "tienes": ["T", "I", "EH", "N", "EH", "S"],
    "tiene": ["T", "I", "EH", "N", "EH"],
    "tenemos": ["T", "EH", "N", "EH", "M", "OW", "S"],
    "teneis": ["T", "EH", "N", "EH", "I", "S"],
    "tienen": ["T", "I", "EH", "N", "EH", "N"],
    "hola": ["OW", "L", "AH"],
    "adios": ["AH", "D", "I", "OW", "S"],
    "gracias": ["G", "R", "AH", "S", "I", "AH", "S"],
    "por favor": ["P", "OW", "R", "F", "AH", "V", "OW", "R"],
    "si": ["S", "I"],
    "no": ["N", "OW"],
    "bueno": ["B", "UW", "EH", "N", "OW"],
    "malo": ["M", "AH", "L", "OW"],
    "grande": ["G", "R", "AA", "N", "D", "EH"],
    "pequeno": ["P", "EH", "K", "EH", "N", "OW"],
    "nuevo": ["N", "UW", "EH", "V", "OW"],
    "viejo": ["V", "I", "EH", "H", "OW"],
    "casa": ["K", "AH", "S", "AH"],
    "escuela": ["EH", "S", "K", "UW", "EH", "L", "AH"],
    "libro": ["L", "I", "B", "R", "OW"],
    "perro": ["P", "EH", "R", "R", "OW"],
    "gato": ["G", "AH", "T", "OW"],
    "dia": ["D", "I", "AH"],
    "noche": ["N", "OW", "CH", "EH"],
    "tiempo": ["T", "I", "EH", "M", "P", "OW"],
    "hora": ["OW", "R", "AH"],
    "lunes": ["L", "UW", "N", "EH", "S"],
    "martes": ["M", "AA", "R", "T", "EH", "S"],
    "miercoles": ["M", "I", "EH", "R", "K", "OW", "L", "EH", "S"],
    "jueves": ["H", "UW", "EH", "V", "EH", "S"],
    "viernes": ["V", "I", "EH", "R", "N", "EH", "S"],
    "sabado": ["S", "AH", "B", "AH", "D", "OW"],
    "domingo": ["D", "OW", "M", "I", "NG", "G", "OW"],
    "uno": ["UW", "N", "OW"],
    "dos": ["D", "OW", "S"],
    "tres": ["T", "R", "EH", "S"],
    "cuatro": ["K", "UW", "AH", "T", "R", "OW"],
    "cinco": ["S", "I", "NG", "K", "OW"],
    "seis": ["S", "EH", "I", "S"],
    "siete": ["S", "I", "EH", "T", "EH"],
    "ocho": ["OW", "CH", "OW"],
    "nueve": ["N", "UW", "EH", "V", "EH"],
    "diez": ["D", "I", "EH", "S"],
    # Additional common Spanish words
    "comer": ["K", "OW", "M", "EH", "R"],
    "beber": ["B", "EH", "B", "EH", "R"],
    "dormir": ["D", "OW", "R", "M", "I", "R"],
    "caminar": ["K", "AH", "M", "I", "N", "AH", "R"],
    "correr": ["K", "OW", "R", "R", "EH", "R"],
    "nadar": ["N", "AH", "D", "AH", "R"],
    "volar": ["V", "OW", "L", "AH", "R"],
    "cantar": ["K", "AH", "N", "T", "AH", "R"],
    "bailar": ["B", "AY", "L", "AH", "R"],
    "jugar": ["H", "UW", "G", "AH", "R"],
    "trabajar": ["T", "R", "AH", "B", "AH", "H", "AH", "R"],
    "estudiar": ["EH", "S", "T", "UW", "D", "I", "AH", "R"],
    "aprender": ["AH", "P", "R", "EH", "N", "D", "EH", "R"],
    "hablar": ["AH", "B", "L", "AH", "R"],
    "escuchar": ["EH", "S", "K", "UW", "CH", "AH", "R"],
    "oír": ["OW", "I", "R"],
    "ver": ["B", "EH", "R"],
    "mirar": ["M", "I", "R", "AH", "R"],
    "tocar": ["T", "OW", "K", "AH", "R"],
    "sentir": ["S", "EH", "N", "T", "I", "R"],
    "oler": ["OW", "L", "EH", "R"],
    "probar": ["P", "R", "OW", "B", "AH", "R"],
    "tomar": ["T", "OW", "M", "AH", "R"],
    "dar": ["D", "AH", "R"],
    "recibir": ["R", "EH", "S", "I", "B", "I", "R"],
    "comprar": ["K", "OW", "M", "P", "R", "AH", "R"],
    "vender": ["B", "EH", "N", "D", "EH", "R"],
    "pagar": ["P", "AH", "G", "AH", "R"],
    "contar": ["K", "OW", "N", "T", "AH", "R"],
    "medir": ["M", "EH", "D", "I", "R"],
    "cortar": ["K", "OW", "R", "T", "AH", "R"],
    "romper": ["R", "OW", "M", "P", "EH", "R"],
    "arreglar": ["AH", "R", "R", "EH", "G", "L", "AH", "R"],
    "limpiar": ["L", "I", "M", "P", "I", "AH", "R"],
    "cocinar": ["K", "OW", "S", "I", "N", "AH", "R"],
    "hervir": ["EH", "R", "V", "I", "R"],
    "freír": ["F", "R", "EH", "I", "R"],
    "congelar": ["K", "OW", "N", "H", "EH", "L", "AH", "R"],
    "descongelar": ["D", "EH", "S", "K", "OW", "N", "H", "EH", "L", "AH", "R"],
    "cerrar": ["S", "EH", "R", "R", "AH", "R"],
    "abrir": ["AH", "B", "R", "I", "R"],
    "entrar": ["EH", "N", "T", "R", "AH", "R"],
    "salir": ["S", "AH", "L", "I", "R"],
    "ir": ["I", "R"],
    "venir": ["B", "EH", "N", "I", "R"],
    "volver": ["B", "OW", "L", "B", "EH", "R"],
    "regresar": ["R", "EH", "G", "R", "EH", "S", "AH", "R"],
    "subir": ["S", "UW", "B", "I", "R"],
    "bajar": ["B", "AH", "H", "AH", "R"],
    "caer": ["K", "AH", "EH", "R"],
}


def _spanish_apply_rules(word: str) -> list[str]:
    """Apply Spanish letter-to-sound rules."""
    result: list[str] = []
    i = 0
    w = word.lower()

    # Spanish digraphs/trigraphs
    spanish_digraphs = [
        ("ll", ["Y"]),
        ("ch", ["CH"]),
        ("rr", ["RR"]),
        ("qu", ["K"]),
        ("gu", ["G"]),
        ("ce", ["S"]),
        ("ci", ["S"]),
        ("z", ["S"]),
    ]

    while i < len(w):
        matched = False

        # Try longest match first (2 chars)
        for length in (2,):
            chunk = w[i:i + length]
            for pattern, sounds in spanish_digraphs:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            # Spanish single letter rules
            spanish_letter_rules = {
                "a": ["AH"], "b": ["B"], "c": ["K"],
                "d": ["D"], "e": ["EH"], "f": ["F"],
                "g": ["G"], "h": [], "i": ["I"],
                "j": ["H"], "k": ["K"], "l": ["L"],
                "m": ["M"], "n": ["N"], "o": ["OW"],
                "p": ["P"], "q": ["K"], "r": ["R"],
                "s": ["S"], "t": ["T"], "u": ["UW"],
                "v": ["B"], "w": ["W"], "x": ["K", "S"],
                "y": ["I"], "z": ["S"],
            }
            if ch in spanish_letter_rules:
                result.extend(spanish_letter_rules[ch])
            i += 1

    return result


def spanish_text_to_phonemes(text: str) -> list[str]:
    """Convert Spanish text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [SPANISH_ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-Záéíóúñü]+|[^a-zA-Záéíóúñü]+", text)

    for token in tokens:
        if token.isspace():
            result.append(SPANISH_ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _SPANISH_PRONUNCIATION_DICT:
                result.extend(_SPANISH_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_spanish_apply_rules(token))
            result.append(SPANISH_ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(SPANISH_ID_TO_PHONEME[SILENCE])

    result.append(SPANISH_ID_TO_PHONEME[EOS])
    return result


class SpanishPhonemeEncoder:
    """Encodes Spanish text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **SPANISH_VOWELS, **SPANISH_CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert Spanish text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = spanish_text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to Spanish text string.

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
            phoneme = SPANISH_ID_TO_PHONEME.get(pid, "?")
            grapheme = SPANISH_PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
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
        return [SPANISH_ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input Spanish text string
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

    def score_pronunciation(self, target: str, spoken: str) -> dict:
        """Score how well a spoken word matches the target pronunciation.

        Uses phoneme-level comparison to evaluate pronunciation accuracy.
        """
        target_ids = self.encode(target).flatten()
        spoken_ids = self.encode(spoken).flatten()

        target_phonemes = [SPANISH_ID_TO_PHONEME.get(i, "?") for i in target_ids if i not in (BOS, EOS, PAD)]
        spoken_phonemes = [SPANISH_ID_TO_PHONEME.get(i, "?") for i in spoken_ids if i not in (BOS, EOS, PAD)]

        lcs_len = self._lcs_length(target_phonemes, spoken_phonemes)
        target_len = len(target_phonemes)
        spoken_len = len(spoken_phonemes)

        precision = lcs_len / spoken_len if spoken_len > 0 else 0.0
        recall = lcs_len / target_len if target_len > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "score": f1,
            "precision": precision,
            "recall": recall,
            "target_phonemes": target_phonemes,
            "spoken_phonemes": spoken_phonemes,
            "target_len": target_len,
            "spoken_len": spoken_len,
        }

    def _lcs_length(self, a: list, b: list) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    @property
    def vocab_size(self) -> int:
        return NUM_SPANISH_PHONEMES
