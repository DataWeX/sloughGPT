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
    # Additional common Portuguese words
    "comer": ["K", "OW", "M", "EH", "R"],
    "beber": ["B", "EH", "B", "EH", "R"],
    "dormir": ["D", "OW", "R", "M", "IY", "R"],
    "caminhar": ["K", "AH", "M", "IY", "NH", "AH", "R"],
    "correr": ["K", "OW", "R", "R", "EH", "R"],
    "nadhar": ["N", "AH", "D", "AH", "R"],
    "voar": ["V", "OW", "AH", "R"],
    "cantar": ["K", "AH", "N", "T", "AH", "R"],
    "dançar": ["D", "AH", "N", "S", "AH", "R"],
    "jogar": ["ZH", "OW", "G", "AH", "R"],
    "trabalhar": ["T", "R", "AH", "B", "AH", "L", "Y", "AH", "R"],
    "estudar": ["EH", "S", "T", "UW", "D", "AH", "R"],
    "aprender": ["AH", "P", "R", "EH", "N", "D", "EH", "R"],
    "falar": ["F", "AH", "L", "AH", "R"],
    "ouvir": ["OW", "V", "IY", "R"],
    "ver": ["V", "EH", "R"],
    "olhar": ["OW", "L", "Y", "AH", "R"],
    "tocar": ["T", "OW", "K", "AH", "R"],
    "sentir": ["S", "EH", "N", "T", "IY", "R"],
    "cheirar": ["SH", "AY", "R", "AH", "R"],
    "provar": ["P", "R", "OW", "V", "AH", "R"],
    "tomar": ["T", "OW", "M", "AH", "R"],
    "dar": ["D", "AH", "R"],
    "receber": ["R", "EH", "S", "EH", "B", "EH", "R"],
    "comprar": ["K", "OW", "M", "P", "R", "AH", "R"],
    "vender": ["V", "EH", "N", "D", "EH", "R"],
    "pagar": ["P", "AH", "G", "AH", "R"],
    "contar": ["K", "OW", "N", "T", "AH", "R"],
    "medir": ["M", "EH", "D", "IY", "R"],
    "cortar": ["K", "OW", "R", "T", "AH", "R"],
    "quebrar": ["K", "EH", "B", "R", "AH", "R"],
    "arrumar": ["AH", "R", "R", "UW", "M", "AH", "R"],
    "limpar": ["L", "IY", "M", "P", "AH", "R"],
    "cozinhar": ["K", "OW", "Z", "IY", "NH", "AH", "R"],
    "ferver": ["F", "EH", "R", "V", "EH", "R"],
    "fritar": ["F", "R", "IY", "T", "AH", "R"],
    "congelar": ["K", "OW", "N", "ZH", "EH", "L", "AH", "R"],
    "descongelar": ["D", "EH", "S", "K", "OW", "N", "ZH", "EH", "L", "AH", "R"],
    "fechar": ["F", "EH", "SH", "AH", "R"],
    "abrir": ["AH", "B", "R", "IY", "R"],
    "entrar": ["EH", "N", "T", "R", "AH", "R"],
    "sair": ["S", "AH", "IY", "R"],
    "ir": ["IY", "R"],
    "vir": ["V", "IY", "R"],
    "voltar": ["V", "OW", "L", "T", "AH", "R"],
    "retornar": ["R", "EH", "T", "OW", "R", "N", "AH", "R"],
    "subir": ["S", "UW", "B", "IY", "R"],
    "descer": ["D", "EH", "S", "EH", "R"],
    "cair": ["K", "AH", "IY", "R"],
    "nadhar": ["N", "AH", "D", "AH", "R"],
    "voar": ["V", "OW", "AH", "R"],
    "caminhar": ["K", "AH", "M", "IY", "NH", "AH", "R"],
    "correr": ["K", "OW", "R", "R", "EH", "R"],
    "nadhar": ["N", "AH", "D", "AH", "R"],
    "voar": ["V", "OW", "AH", "R"],
    "cantar": ["K", "AH", "N", "T", "AH", "R"],
    "dançar": ["D", "AH", "N", "S", "AH", "R"],
    "jogar": ["ZH", "OW", "G", "AH", "R"],
    "trabalhar": ["T", "R", "AH", "B", "AH", "L", "Y", "AH", "R"],
    "estudar": ["EH", "S", "T", "UW", "D", "AH", "R"],
    "aprender": ["AH", "P", "R", "EH", "N", "D", "EH", "R"],
    "falar": ["F", "AH", "L", "AH", "R"],
    "ouvir": ["OW", "V", "IY", "R"],
    "ver": ["V", "EH", "R"],
    "olhar": ["OW", "L", "Y", "AH", "R"],
    "tocar": ["T", "OW", "K", "AH", "R"],
    "sentir": ["S", "EH", "N", "T", "IY", "R"],
    "cheirar": ["SH", "AY", "R", "AH", "R"],
    "provar": ["P", "R", "OW", "V", "AH", "R"],
    "tomar": ["T", "OW", "M", "AH", "R"],
    "dar": ["D", "AH", "R"],
    "receber": ["R", "EH", "S", "EH", "B", "EH", "R"],
    "comprar": ["K", "OW", "M", "P", "R", "AH", "R"],
    "vender": ["V", "EH", "N", "D", "EH", "R"],
    "pagar": ["P", "AH", "G", "AH", "R"],
    "contar": ["K", "OW", "N", "T", "AH", "R"],
    "medir": ["M", "EH", "D", "IY", "R"],
    "cortar": ["K", "OW", "R", "T", "AH", "R"],
    "quebrar": ["K", "EH", "B", "R", "AH", "R"],
    "arrumar": ["AH", "R", "R", "UW", "M", "AH", "R"],
    "limpar": ["L", "IY", "M", "P", "AH", "R"],
    "cozinhar": ["K", "OW", "Z", "IY", "NH", "AH", "R"],
    "ferver": ["F", "EH", "R", "V", "EH", "R"],
    "fritar": ["F", "R", "IY", "T", "AH", "R"],
    "congelar": ["K", "OW", "N", "ZH", "EH", "L", "AH", "R"],
    "descongelar": ["D", "EH", "S", "K", "OW", "N", "ZH", "EH", "L", "AH", "R"],
    "fechar": ["F", "EH", "SH", "AH", "R"],
    "abrir": ["AH", "B", "R", "IY", "R"],
    "entrar": ["EH", "N", "T", "R", "AH", "R"],
    "sair": ["S", "AH", "IY", "R"],
    "ir": ["IY", "R"],
    "vir": ["V", "IY", "R"],
    "voltar": ["V", "OW", "L", "T", "AH", "R"],
    "retornar": ["R", "EH", "T", "OW", "R", "N", "AH", "R"],
    "subir": ["S", "UW", "B", "IY", "R"],
    "descer": ["D", "EH", "S", "EH", "R"],
    "cair": ["K", "AH", "IY", "R"],
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

    def score_pronunciation(self, target: str, spoken: str) -> dict:
        """Score how well a spoken word matches the target pronunciation.

        Uses phoneme-level comparison to evaluate pronunciation accuracy.
        """
        target_ids = self.encode(target).flatten()
        spoken_ids = self.encode(spoken).flatten()

        target_phonemes = [PORTUGUESE_ID_TO_PHONEME.get(i, "?") for i in target_ids if i not in (BOS, EOS, PAD)]
        spoken_phonemes = [PORTUGUESE_ID_TO_PHONEME.get(i, "?") for i in spoken_ids if i not in (BOS, EOS, PAD)]

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
        return NUM_PORTUGUESE_PHONEMES
