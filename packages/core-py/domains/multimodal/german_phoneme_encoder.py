"""
German Phoneme Encoder — rule-based grapheme-to-phoneme for German TTS.

Maps German text to a compact phoneme vocabulary. Based on ARPAbet with
German-specific additions.

Phoneme set based on IPA/ARPAbet hybrid for German.
"""

from __future__ import annotations

import re
import numpy as np


# ── German Phoneme inventory ──────────────────────────────────────────────

# Vowels
GERMAN_VOWELS = {
    "AA": 0,   # Vater (father)
    "AE": 1,   # Katze (cat)
    "AH": 2,   # Butterfly (Schmetterling)
    "AO": 3,   # Born (source)
    "AW": 4,   # Haus (house)
    "AY": 5,   # Zeit (time)
    "EH": 6,   # Bett (bed)
    "ER": 7,   # maneuver
    "EY": 8,   # Keks (cookie)
    "IH": 9,   # Biene (bee)
    "IY": 10,  # Feedback
    "OW": 11,  # Auto (car)
    "OY": 12,  # Boy
    "UH": 13,  # Buch (book)
    "UW": 14,  # Fuus (foot)
    "OE": 15,  # schoen (beautiful)
    "UE": 16,  # ueber (over)
    "AEU": 17,  # Maedchen (girl)
}

# Consonants
GERMAN_CONSONANTS = {
    "B": 18,   # Ball
    "CH": 19,  # ich (I)
    "D": 20,   # Daum (thumb)
    "DH": 21,  # this (English loan)
    "F": 22,   # Frau (woman)
    "G": 23,   # Garten (garden)
    "HH": 24,  # Haus (house)
    "JH": 25,  # Joy (English loan)
    "K": 26,   # Kind (child)
    "L": 27,   # Liebe (love)
    "M": 28,   # Mutter (mother)
    "N": 29,   # Nein (no)
    "NG": 30,  # Finger
    "P": 31,   # Papa
    "R": 32,   # Rot (red)
    "S": 33,   # Sonne (sun)
    "SH": 34,  # Schule (school)
    "T": 35,   # Tag (day)
    "TH": 36,  # think (English loan)
    "V": 37,   # Vater (father) - pronounced as F
    "W": 38,   # Wasser (water)
    "Y": 39,   # Yacht
    "Z": 40,   # Zeit (time) - pronounced as TS
    "TS": 41,  # Zeug (thing)
    "ZH": 42,  # vision (English loan)
}

# Special tokens
PAD = 43
BOS = 44
EOS = 45
SPACE = 46
SILENCE = 47

NUM_GERMAN_PHONEMES = 48

# Reverse lookup
GERMAN_ID_TO_PHONEME = {v: k for k, v in {**GERMAN_VOWELS, **GERMAN_CONSONANTS}.items()}
GERMAN_ID_TO_PHONEME[PAD] = "_"
GERMAN_ID_TO_PHONEME[BOS] = "<"
GERMAN_ID_TO_PHONEME[EOS] = ">"
GERMAN_ID_TO_PHONEME[SPACE] = " "
GERMAN_ID_TO_PHONEME[SILENCE] = "-"

# Phoneme-to-grapheme mapping for decode
GERMAN_PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Vowels
    "AA": "a", "AE": "a", "AH": "u", "AO": "o",
    "AW": "au", "AY": "ei", "EH": "e", "ER": "er",
    "EY": "e", "IH": "i", "IY": "ie", "OW": "o",
    "OY": "eu", "UH": "u", "UW": "u",
    "OE": "oe", "UE": "ue", "AEU": "ae",
    # Consonants
    "B": "b", "CH": "ch", "D": "d", "DH": "th",
    "F": "f", "G": "g", "HH": "h", "JH": "j",
    "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ng", "P": "p", "R": "r", "S": "s",
    "SH": "sch", "T": "t", "TH": "th", "V": "v",
    "W": "w", "Y": "y", "Z": "z", "TS": "z",
    "ZH": "zh",
}


# ── German Grapheme-to-Phoneme rules ─────────────────────────────────────

_GERMAN_PRONUNCIATION_DICT: dict[str, list[str]] = {
    "ich": ["IH", "CH"],
    "du": ["D", "UW"],
    "er": ["EH", "R"],
    "sie": ["Z", "IY"],
    "wir": ["V", "IH", "R"],
    "ihr": ["IH", "R"],
    "sind": ["Z", "IH", "N", "D"],
    "bin": ["B", "IH", "N"],
    "ist": ["IH", "S", "T"],
    "hat": ["HH", "AA", "T"],
    "haben": ["HH", "AA", "B", "AH", "N"],
    "nicht": ["N", "IH", "CH", "T"],
    "ja": ["Y", "AA"],
    "nein": ["N", "AY", "N"],
    "danke": ["D", "AA", "NG", "K", "AH"],
    "guten": ["G", "UW", "T", "AH", "N"],
    "morgen": ["M", "AO", "R", "G", "AH", "N"],
    "tag": ["T", "AA", "G"],
    "abend": ["AA", "B", "AH", "N", "D"],
    "nacht": ["N", "AA", "CH", "T"],
    "hello": ["HH", "EH", "L", "OW"],
    "welt": ["V", "EH", "L", "T"],
    "hund": ["HH", "UH", "N", "D"],
    "katze": ["K", "AA", "TS", "AH"],
    "auto": ["AA", "UW", "T", "OW"],
    "haus": ["HH", "AW", "S"],
    "schule": ["SH", "UW", "L", "AH"],
    "buch": ["B", "UH", "CH"],
    "gut": ["G", "UW", "T"],
    "gross": ["G", "R", "OW", "S"],
    "klein": ["K", "L", "AY", "N"],
    "neu": ["N", "OY"],
    "alt": ["AA", "L", "T"],
    "heiss": ["HH", "AY", "S"],
    "kalt": ["K", "AA", "L", "T"],
    "schnell": ["SH", "N", "EH", "L"],
    "langsam": ["L", "AA", "NG", "Z", "AA", "M"],
    "bitte": ["B", "IH", "T", "T", "AH"],
    "please": ["P", "L", "IY", "Z"],
    "danke": ["D", "AA", "NG", "K", "AH"],
    "tschuss": ["CH", "UW", "S"],
    "goodbye": ["G", "UH", "D", "B", "AY"],
    "eins": ["AY", "N", "S"],
    "zwei": ["TS", "V", "AY"],
    "drei": ["D", "R", "AY"],
    "vier": ["F", "IH", "R"],
    "fuenf": ["F", "UE", "N", "F"],
    "sechs": ["Z", "EH", "K", "S"],
    "sieben": ["Z", "IY", "B", "AH", "N"],
    "acht": ["AA", "CH", "T"],
    "neun": ["N", "OY", "N"],
    "zehn": ["TS", "EY", "N"],
    # Additional common German words
    "wasser": ["V", "AA", "S", "ER"],
    "essen": ["EH", "S", "S", "AH", "N"],
    "trinken": ["T", "R", "IH", "NG", "K", "AH", "N"],
    "schlafen": ["SH", "L", "AA", "F", "AH", "N"],
    "gehen": ["G", "EY", "AH", "N"],
    "kommen": ["K", "OW", "M", "M", "AH", "N"],
    "sehen": ["Z", "EY", "AH", "N"],
    "hören": ["HH", "OE", "R", "AH", "N"],
    "sprechen": ["SH", "P", "R", "EH", "CH", "AH", "N"],
    "lesen": ["L", "EY", "Z", "AH", "N"],
    "schreiben": ["SH", "R", "AY", "B", "AH", "N"],
    "arbeiten": ["AA", "R", "B", "AY", "T", "AH", "N"],
    "lernen": ["L", "EH", "R", "N", "AH", "N"],
    "spielen": ["SH", "P", "IY", "L", "AH", "N"],
    "kaufen": ["K", "AW", "F", "AH", "N"],
    "verkaufen": ["F", "EH", "R", "K", "AW", "F", "AH", "N"],
    "helfen": ["HH", "EH", "L", "F", "AH", "N"],
    "brauchen": ["B", "R", "AW", "CH", "AH", "N"],
    "glauben": ["G", "L", "AW", "B", "AH", "N"],
    "denken": ["D", "EH", "NG", "K", "AH", "N"],
    "wissen": ["V", "IH", "S", "S", "AH", "N"],
    "kennen": ["K", "EH", "N", "N", "AH", "N"],
    "lieben": ["L", "IY", "B", "AH", "N"],
    "hassen": ["HH", "AA", "S", "S", "AH", "N"],
    "freuen": ["F", "R", "OY", "AH", "N"],
    "weinen": ["V", "AY", "N", "AH", "N"],
    "lachen": ["L", "AA", "CH", "AH", "N"],
    "tanzen": ["T", "AA", "N", "TS", "AH", "N"],
    "singen": ["Z", "IH", "NG", "AH", "N"],
    "kochen": ["K", "OW", "CH", "AH", "N"],
    "waschen": ["V", "AA", "SH", "AH", "N"],
    "fahren": ["F", "AA", "R", "AH", "N"],
    "fliegen": ["F", "L", "IY", "G", "AH", "N"],
    "laufen": ["L", "AW", "F", "AH", "N"],
    "springen": ["SH", "P", "R", "IH", "NG", "AH", "N"],
    "fallen": ["F", "AA", "L", "L", "AH", "N"],
    "heben": ["HH", "EY", "B", "AH", "N"],
    "legen": ["L", "EY", "G", "AH", "N"],
    "stellen": ["SH", "T", "EH", "L", "L", "AH", "N"],
    "suchen": ["Z", "UH", "CH", "AH", "N"],
    "finden": ["F", "IH", "N", "D", "AH", "N"],
    "bekommen": ["B", "EH", "K", "OW", "M", "M", "AH", "N"],
    "geben": ["G", "EY", "B", "AH", "N"],
    "nehmen": ["N", "EY", "AH", "M"],
    "lassen": ["L", "AA", "S", "S", "AH", "N"],
    "machen": ["M", "AA", "CH", "AH", "N"],
    "tun": ["T", "UW", "N"],
    "sagen": ["Z", "AA", "G", "AH", "N"],
    "fragen": ["F", "R", "AA", "G", "AH", "N"],
    "antworten": ["AA", "N", "T", "V", "AO", "R", "T", "AH", "N"],
    "erzählen": ["EH", "R", "TS", "EH", "L", "AH", "N"],
    "erklären": ["EH", "R", "K", "L", "EH", "R", "AH", "N"],
    "zeigen": ["TS", "AY", "G", "AH", "N"],
    "zeigen": ["TS", "AY", "G", "AH", "N"],
    "bringen": ["B", "R", "IH", "NG", "AH", "N"],
    "holen": ["HH", "OW", "L", "AH", "N"],
    "setzen": ["TS", "EH", "TS", "TS", "AH", "N"],
    "sitzen": ["Z", "IH", "TS", "TS", "AH", "N"],
    "liegen": ["L", "IY", "G", "AH", "N"],
    "stehen": ["S", "T", "EY", "AH", "N"],
    "bleiben": ["B", "L", "AY", "B", "AH", "N"],
    "werden": ["V", "EH", "R", "D", "AH", "N"],
    "haben": ["HH", "AA", "B", "AH", "N"],
    "sein": ["Z", "AY", "N"],
    "haben": ["HH", "AA", "B", "AH", "N"],
    "essen": ["EH", "S", "S", "AH", "N"],
    "trinken": ["T", "R", "IH", "NG", "K", "AH", "N"],
    "schlafen": ["SH", "L", "AA", "F", "AH", "N"],
    "gehen": ["G", "EY", "AH", "N"],
    "kommen": ["K", "OW", "M", "M", "AH", "N"],
    "sehen": ["Z", "EY", "AH", "N"],
    "hören": ["HH", "OE", "R", "AH", "N"],
    "sprechen": ["SH", "P", "R", "EH", "CH", "AH", "N"],
    "lesen": ["L", "EY", "Z", "AH", "N"],
    "schreiben": ["SH", "R", "AY", "B", "AH", "N"],
    "arbeiten": ["AA", "R", "B", "AY", "T", "AH", "N"],
    "lernen": ["L", "EH", "R", "N", "AH", "N"],
    "spielen": ["SH", "P", "IY", "L", "AH", "N"],
    "kaufen": ["K", "AW", "F", "AH", "N"],
    "verkaufen": ["F", "EH", "R", "K", "AW", "F", "AH", "N"],
    "helfen": ["HH", "EH", "L", "F", "AH", "N"],
    "brauchen": ["B", "R", "AW", "CH", "AH", "N"],
    "glauben": ["G", "L", "AW", "B", "AH", "N"],
    "denken": ["D", "EH", "NG", "K", "AH", "N"],
    "wissen": ["V", "IH", "S", "S", "AH", "N"],
    "kennen": ["K", "EH", "N", "N", "AH", "N"],
    "lieben": ["L", "IY", "B", "AH", "N"],
    "hassen": ["HH", "AA", "S", "S", "AH", "N"],
    "freuen": ["F", "R", "OY", "AH", "N"],
    "weinen": ["V", "AY", "N", "AH", "N"],
    "lachen": ["L", "AA", "CH", "AH", "N"],
    "tanzen": ["T", "AA", "N", "TS", "AH", "N"],
    "singen": ["Z", "IH", "NG", "AH", "N"],
    "kochen": ["K", "OW", "CH", "AH", "N"],
    "waschen": ["V", "AA", "SH", "AH", "N"],
    "fahren": ["F", "AA", "R", "AH", "N"],
    "fliegen": ["F", "L", "IY", "G", "AH", "N"],
    "laufen": ["L", "AW", "F", "AH", "N"],
    "springen": ["SH", "P", "R", "IH", "NG", "AH", "N"],
    "fallen": ["F", "AA", "L", "L", "AH", "N"],
    "heben": ["HH", "EY", "B", "AH", "N"],
    "legen": ["L", "EY", "G", "AH", "N"],
    "stellen": ["SH", "T", "EH", "L", "L", "AH", "N"],
    "suchen": ["Z", "UH", "CH", "AH", "N"],
    "finden": ["F", "IH", "N", "D", "AH", "N"],
    "bekommen": ["B", "EH", "K", "OW", "M", "M", "AH", "N"],
    "geben": ["G", "EY", "B", "AH", "N"],
    "nehmen": ["N", "EY", "AH", "M"],
    "lassen": ["L", "AA", "S", "S", "AH", "N"],
    "machen": ["M", "AA", "CH", "AH", "N"],
    "tun": ["T", "UW", "N"],
    "sagen": ["Z", "AA", "G", "AH", "N"],
    "fragen": ["F", "R", "AA", "G", "AH", "N"],
    "antworten": ["AA", "N", "T", "V", "AO", "R", "T", "AH", "N"],
    "erzählen": ["EH", "R", "TS", "EH", "L", "AH", "N"],
    "erklären": ["EH", "R", "K", "L", "EH", "R", "AH", "N"],
    "zeigen": ["TS", "AY", "G", "AH", "N"],
    "bringen": ["B", "R", "IH", "NG", "AH", "N"],
    "holen": ["HH", "OW", "L", "AH", "N"],
    "setzen": ["TS", "EH", "TS", "TS", "AH", "N"],
    "sitzen": ["Z", "IH", "TS", "TS", "AH", "N"],
    "liegen": ["L", "IY", "G", "AH", "N"],
    "stehen": ["S", "T", "EY", "AH", "N"],
    "bleiben": ["B", "L", "AY", "B", "AH", "N"],
    "werden": ["V", "EH", "R", "D", "AH", "N"],
}


def _german_apply_rules(word: str) -> list[str]:
    """Apply German letter-to-sound rules."""
    result: list[str] = []
    i = 0
    w = word.lower()

    # German digraphs/trigraphs
    german_digraphs = [
        ("sch", ["SH"]),
        ("ch", ["CH"]),
        ("sch", ["SH"]),
        ("ei", ["AY"]),
        ("ai", ["AY"]),
        ("au", ["AW"]),
        ("eu", ["OY"]),
        ("ae", ["EH"]),
        ("oe", ["OE"]),
        ("ue", ["UE"]),
        ("ie", ["IY"]),
        ("au", ["AW"]),
        ("ou", ["AW"]),
        ("st", ["S", "T"]),
        ("sp", ["SH", "P"]),
    ]

    while i < len(w):
        matched = False

        # Try longest match first (3, 2 chars)
        for length in (3, 2):
            chunk = w[i:i + length]
            for pattern, sounds in german_digraphs:
                if chunk == pattern:
                    result.extend(sounds)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = w[i]
            # German single letter rules
            german_letter_rules = {
                "a": ["AA"], "b": ["B"], "c": ["TS"],
                "d": ["D"], "e": ["EH"], "f": ["F"],
                "g": ["G"], "h": ["HH"], "i": ["IH"],
                "j": ["Y"], "k": ["K"], "l": ["L"],
                "m": ["M"], "n": ["N"], "o": ["OW"],
                "p": ["P"], "q": ["K"], "r": ["R"],
                "s": ["Z"], "t": ["T"], "u": ["UW"],
                "v": ["F"], "w": ["V"], "x": ["K", "S"],
                "y": ["Y"], "z": ["TS"],
            }
            if ch in german_letter_rules:
                result.extend(german_letter_rules[ch])
            i += 1

    return result


def german_text_to_phonemes(text: str) -> list[str]:
    """Convert German text to a list of phoneme strings.

    Uses dictionary lookup first, falls back to letter-to-sound rules.
    """
    result: list[str] = [GERMAN_ID_TO_PHONEME[BOS]]

    # Tokenize: words and whitespace/punctuation
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+|[^a-zA-ZäöüÄÖÜß]+", text)

    for token in tokens:
        if token.isspace():
            result.append(GERMAN_ID_TO_PHONEME[SPACE])
        elif token[0].isalpha():
            lower = token.lower().strip("'")
            if lower in _GERMAN_PRONUNCIATION_DICT:
                result.extend(_GERMAN_PRONUNCIATION_DICT[lower])
            else:
                result.extend(_german_apply_rules(token))
            result.append(GERMAN_ID_TO_PHONEME[SPACE])
        else:
            # Punctuation — add brief pause
            result.append(GERMAN_ID_TO_PHONEME[SILENCE])

    result.append(GERMAN_ID_TO_PHONEME[EOS])
    return result


class GermanPhonemeEncoder:
    """Encodes German text to phoneme ID sequences for the TTS model."""

    def __init__(self):
        self.phoneme_to_id: dict[str, int] = {
            **GERMAN_VOWELS, **GERMAN_CONSONANTS,
            "_": PAD, "<": BOS, ">": EOS, " ": SPACE, "-": SILENCE,
        }

    def encode(self, text: str) -> np.ndarray:
        """Convert German text to phoneme ID array.

        Returns:
            (1, seq_len) int32 array of phoneme IDs (space/padding tokens filtered out)
        """
        phonemes = german_text_to_phonemes(text)
        ids = [self.phoneme_to_id.get(p, PAD) for p in phonemes]
        # Remove space separators and padding tokens
        ids = [i for i in ids if i not in (PAD, SPACE)]
        if not ids:
            ids = [PAD]
        return np.array([ids], dtype=np.int32)

    def decode(self, ids: np.ndarray) -> str:
        """Convert phoneme ID array back to German text string.

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
            phoneme = GERMAN_ID_TO_PHONEME.get(pid, "?")
            grapheme = GERMAN_PHONEME_TO_GRAPHEME.get(phoneme, phoneme)
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
        return [GERMAN_ID_TO_PHONEME.get(pid, "?") for pid in id_list]

    def visualize(self, text: str) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input German text string
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

        target_phonemes = [GERMAN_ID_TO_PHONEME.get(i, "?") for i in target_ids if i not in (BOS, EOS, PAD)]
        spoken_phonemes = [GERMAN_ID_TO_PHONEME.get(i, "?") for i in spoken_ids if i not in (BOS, EOS, PAD)]

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
        return NUM_GERMAN_PHONEMES
