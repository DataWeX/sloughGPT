"""No voice domain (card d6e498ad ruling): voice is a product surface, not a domain.

The audio/phoneme stack (TTS, speech recognition, phoneme encoders, audio
filters, engines) lives in domain.multimodal — the modality taxonomy. The
former domain.voice package was absorbed whole; routers/voice.py and
routers/phoneme.py are HTTP surfaces over domain.multimodal, like every
other router.

These assertions are the ratchet for the user's ruling: domain/voice must
not come back, and the public audio API must stay importable from
domain.multimodal where every consumer now points.
"""

import importlib.util

import pytest


def test_voice_domain_is_gone():
    assert importlib.util.find_spec("domain.voice") is None, (
        "domain.voice re-appeared — voice is a product surface served by "
        "routers/voice.py + routers/phoneme.py, not a domain package; "
        "audio/phoneme code belongs in domain.multimodal (card d6e498ad)"
    )


def test_audio_public_api_lives_on_multimodal():
    import domain.multimodal as mm

    expected = [
        # engines
        "VoiceEngine",
        "get_voice_engine",
        "PhonemeEngine",
        "get_phoneme_engine",
        # TTS
        "TTSEngine",
        "GriffinLimVocoder",
        "SpectrogramDecoder",
        "parse_ssml",
        # speech recognition
        "SpeechRecognizer",
        "get_speech_recognizer",
        # phoneme encoding
        "PhonemeEncoder",
        "UnifiedPhonemeEncoder",
        "NUM_PHONEMES",
        "detect_language",
        "text_to_phonemes",
        # audio filters
        "FilterMode",
        "AudioFilterConfig",
        "FilterResult",
        "apply_noise_gate",
        "apply_agc",
        "normalize_loudness",
        "detect_voice_activity",
        "apply_audio_filter",
    ]
    missing = [n for n in expected if not hasattr(mm, n)]
    assert not missing, f"domain.multimodal lost the absorbed audio API: {missing}"


@pytest.mark.parametrize(
    ("router", "factory"),
    [
        ("apps/api/server/routers/voice.py", "get_voice_engine"),
        ("apps/api/server/routers/phoneme.py", "get_phoneme_engine"),
    ],
)
def test_routers_source_engines_from_multimodal(router, factory):
    src = open(router).read()
    assert "domain.voice" not in src, f"{router} still references the dead domain.voice"
    assert f"from domain.multimodal import {factory}" in src
