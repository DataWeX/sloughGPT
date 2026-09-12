"""
Unified Phoneme Encoder — multi-language phoneme encoding.

Provides a single interface for encoding text in multiple languages.
Supports English, German, French, Spanish, Italian, and Portuguese with automatic language detection.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from domain.voice._internal.phoneme_encoder import PhonemeEncoder, BOS as EN_BOS, EOS as EN_EOS, PAD as EN_PAD
from domain.voice._internal.german_phoneme_encoder import (
    GermanPhonemeEncoder, BOS as DE_BOS, EOS as DE_EOS, PAD as DE_PAD,
)
from domain.voice._internal.french_phoneme_encoder import FrenchPhonemeEncoder
from domain.voice._internal.spanish_phoneme_encoder import SpanishPhonemeEncoder
from domain.voice._internal.italian_phoneme_encoder import ItalianPhonemeEncoder
from domain.voice._internal.portuguese_phoneme_encoder import PortuguesePhonemeEncoder


# Language detection patterns
_LANG_PATTERNS = {
    "de": ["ich", "du", "er", "sie", "wir", "ihr", "bin", "ist", "hat", "haben",
           "nicht", "ja", "nein", "danke", "guten", "morgen", "tag", "abend",
           "nacht", "welt", "hund", "katze", "auto", "haus", "schule", "buch",
           "gut", "gross", "klein", "neu", "alt", "heiss", "kalt", "schnell",
           "langsam", "bitte", "tschuss", "eins", "zwei", "drei", "vier",
           "fuenf", "sechs", "sieben", "acht", "neun", "zehn"],
    "fr": ["je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
           "suis", "es", "est", "sommes", "etes", "ai", "as", "avons", "ont",
           "pas", "ne", "oui", "non", "merci", "bonjour", "bonsoir", "salut",
           "au revoir", "s'il vous plait", "pardon", "excusez-moi",
           "bon", "mauvais", "grand", "petit", "nouveau", "vieux",
           "maison", "ecole", "livre", "chien", "chat", "eau", "pain", "vin",
           "fromage", "comment", "allez", "tres", "bien", "aussi", "peut-etre",
           "voir", "faire", "dire", "venir", "prendre", "mettre", "parler",
           "manger", "dormir", "lire", "ecrire", "vivre",
           "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
           "jour", "nuit", "heure", "matin", "soir",
           "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
    "es": ["yo", "tu", "el", "ella", "nosotros", "vosotros", "ellos", "ellas",
           "soy", "eres", "es", "somos", "sois", "son",
           "tengo", "tienes", "tiene", "tenemos", "teneis", "tienen",
           "hola", "adios", "gracias", "por favor", "si", "no",
           "bueno", "malo", "grande", "pequeno", "nuevo", "viejo",
           "casa", "escuela", "libro", "perro", "gato", "dia", "noche",
           "tiempo", "hora", "lunes", "martes", "miercoles", "jueves",
           "viernes", "sabado", "domingo",
           "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
           "ocho", "nueve", "diez"],
    "it": ["io", "tu", "lui", "lei", "noi", "voi", "loro",
           "sono", "sei", "e", "siamo", "siete", "hanno", "ho", "hai",
           "non", "si", "no", "ciao", "buongiorno", "buonasera", "grazie",
           "prego", "per favore", "bene", "male", "grande", "piccolo",
           "nuovo", "vecchio", "casa", "scuola", "libro", "gatto",
           "giorno", "notte", "tempo", "ora",
           "lunedi", "martedi", "mercoledi", "giovedi", "venerdi",
           "sabato", "domenica",
           "uno", "due", "tre", "quattro", "cinque", "sei", "sette",
           "otto", "nove", "dieci"],
    "pt": ["eu", "ele", "ela", "nos", "voce", "eles", "elas",
           "sou", "eres", "e", "somos", "sao",
           "tenho", "ten", "mae", "nao", "sim", "oi", "ola",
           "obrigado", "obrigada", "por favor", "bem", "mal",
           "grande", "pequeno", "novo", "velho",
           "casa", "escola", "livro", "gato", "dia", "noite",
           "tempo", "hora", "segunda", "terca", "quarta", "quinta",
           "sexta", "sabado", "domingo",
           "um", "dois", "tres", "quatro", "cinco", "seis", "sette",
           "oito", "nove", "dez"],
}


def detect_language(text: str) -> str:
    """Detect the language of input text.

    Simple heuristic based on common words in each language.
    Defaults to English if no matches are found.

    Args:
        text: Input text string
    Returns:
        Language code ("en", "de", "fr", "es", "it", or "pt")
    """
    words = text.lower().split()
    for word in words:
        for lang, patterns in _LANG_PATTERNS.items():
            if word in patterns:
                return lang
    return "en"


class UnifiedPhonemeEncoder:
    """Multi-language phoneme encoder.

    Automatically detects the language and uses the appropriate encoder.
    Supports English, German, French, Spanish, Italian, and Portuguese.
    """

    def __init__(self):
        self._encoders = {
            "en": PhonemeEncoder(),
            "de": GermanPhonemeEncoder(),
            "fr": FrenchPhonemeEncoder(),
            "es": SpanishPhonemeEncoder(),
            "it": ItalianPhonemeEncoder(),
            "pt": PortuguesePhonemeEncoder(),
        }
        self._current_language = "en"

    def encode(self, text: str, language: Optional[str] = None) -> np.ndarray:
        """Convert text to phoneme ID array.

        Args:
            text: Input text string
            language: Optional language code ("en", "de", "fr", "es", or "it"). If None, auto-detects.
        Returns:
            (1, seq_len) int32 array of phoneme IDs
        """
        lang = language or detect_language(text)
        if lang not in self._encoders:
            raise ValueError(f"Unsupported language: {lang}")
        self._current_language = lang
        return self._encoders[lang].encode(text)

    def decode(self, ids: np.ndarray, language: Optional[str] = None) -> str:
        """Convert phoneme ID array back to text string.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
            language: Optional language code. If None, uses last encoded language.
        Returns:
            Decoded text string
        """
        lang = language or self._current_language
        if lang not in self._encoders:
            raise ValueError(f"Unsupported language: {lang}")
        return self._encoders[lang].decode(ids)

    def decode_phonemes(self, ids: np.ndarray, language: Optional[str] = None) -> list[str]:
        """Convert phoneme ID array to phoneme string list.

        Args:
            ids: (1, seq_len) int32 array of phoneme IDs
            language: Optional language code. If None, uses last encoded language.
        Returns:
            List of phoneme strings
        """
        lang = language or self._current_language
        if lang not in self._encoders:
            raise ValueError(f"Unsupported language: {lang}")
        return self._encoders[lang].decode_phonemes(ids)

    def visualize(self, text: str, language: Optional[str] = None) -> str:
        """Visualize the encoding process for debugging.

        Args:
            text: Input text string
            language: Optional language code. If None, auto-detects.
        Returns:
            Multi-line string showing the encoding pipeline
        """
        lang = language or detect_language(text)
        if lang not in self._encoders:
            raise ValueError(f"Unsupported language: {lang}")
        self._current_language = lang
        return self._encoders[lang].visualize(text)

    def score_pronunciation(self, target: str, spoken: str,
                           language: Optional[str] = None) -> dict:
        """Score how well a spoken word matches the target pronunciation.

        Args:
            target: The correct word/phrase
            spoken: The spoken word/phrase to evaluate
            language: Optional language code. If None, auto-detects.
        Returns:
            Dictionary with score and detailed comparison
        """
        lang = language or detect_language(target)
        if lang not in self._encoders:
            raise ValueError(f"Unsupported language: {lang}")
        return self._encoders[lang].score_pronunciation(target, spoken)

    def encode_batch(self, texts: list[str], language: Optional[str] = None) -> list[dict]:
        """Batch encode multiple texts.

        Args:
            texts: List of input text strings
            language: Optional language code. If None, auto-detects for each text.
        Returns:
            List of dictionaries with encoding results
        """
        results = []
        for text in texts:
            if language:
                ids = self.encode(text, language=language)
                lang = language
            else:
                ids = self.encode(text)
                lang = self.current_language

            phonemes = self.decode_phonemes(ids, language=lang)
            decoded = self.decode(ids, language=lang)

            results.append({
                "text": text,
                "language": lang,
                "phonemes": phonemes,
                "ids": ids.flatten().tolist(),
                "decoded": decoded,
            })

        return results

    def detect_language(self, text: str) -> str:
        """Detect the language of input text.

        Args:
            text: Input text string
        Returns:
            Language code ("en", "de", "fr", "es", "it", or "pt")
        """
        return detect_language(text)

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        return list(self._encoders.keys())

    @property
    def current_language(self) -> str:
        """Return the last used language code."""
        return self._current_language
