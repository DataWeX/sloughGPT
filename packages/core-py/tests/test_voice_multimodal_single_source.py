"""domain.multimodal's audio modules are re-export shims of domain.voice.

Card d6e498ad — the voice/multimodal fork reconciliation. voice is canonical
for audio/phoneme (precedent: this package's speech.py already re-exported;
verified content-equivalence: 6 byte-identical encoders, tts/audio_filter/
unified_phoneme_encoder AST-equal modulo docstrings and import strategy).
These identity assertions are the ratchet that stops the fork re-growing:
a future edit must land in voice, where both packages see it at once.
"""

import importlib

import pytest

SHIMMED = {
    "audio_filter": [
        "AudioFilterConfig",
        "FilterMode",
        "FilterResult",
        "apply_agc",
        "apply_audio_filter",
        "apply_noise_gate",
        "detect_voice_activity",
        "normalize_loudness",
    ],
    "french_phoneme_encoder": [
        "BOS",
        "EOS",
        "FRENCH_CONSONANTS",
        "FRENCH_ID_TO_PHONEME",
        "FRENCH_VOWELS",
        "NUM_FRENCH_PHONEMES",
        "PAD",
        "SILENCE",
        "SPACE",
        "FrenchPhonemeEncoder",
        "french_text_to_phonemes",
    ],
    "german_phoneme_encoder": [
        "BOS",
        "EOS",
        "GERMAN_CONSONANTS",
        "GERMAN_ID_TO_PHONEME",
        "GERMAN_VOWELS",
        "NUM_GERMAN_PHONEMES",
        "PAD",
        "SILENCE",
        "SPACE",
        "GermanPhonemeEncoder",
        "german_text_to_phonemes",
    ],
    "italian_phoneme_encoder": [
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
    ],
    "phoneme_encoder": [
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
    ],
    "portuguese_phoneme_encoder": [
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
    ],
    "spanish_phoneme_encoder": [
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
    ],
    "tts": [
        "GriffinLimVocoder",
        "SpectrogramDecoder",
        "TTSEngine",
        "parse_ssml",
    ],
    "unified_phoneme_encoder": [
        "UnifiedPhonemeEncoder",
        "detect_language",
    ],
}


@pytest.mark.parametrize("mod", sorted(SHIMMED))
def test_multimodal_audio_module_reexports_voice_objects(mod):
    voice_mod = importlib.import_module(f"domain.voice._internal.{mod}")
    mm_mod = importlib.import_module(f"domain.multimodal._internal.{mod}")
    for name in SHIMMED[mod]:
        assert hasattr(voice_mod, name), f"voice._internal.{mod} lost {name}"
        assert getattr(mm_mod, name) is getattr(voice_mod, name), (
            f"multimodal._internal.{mod}.{name} is not voice's object — "
            "the fork is re-growing; fix belongs in domain.voice"
        )
