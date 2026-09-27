"""
Regression: TTSEngine must construct against the real slonet primitives.

Bug fixed: TTSEngine.__init__ unpacked _get_slonet() off-by-one — position 3
is SloLSTM, not SloAdam — so SloAdam(lr=1e-3) raised TypeError and every
server TTS request silently fell back to browser TTS (backend="browser-fallback",
empty audio). Router tests mock the engine, so only a real construction
catches this class of drift.
"""

import numpy as np

from domain.voice._internal.tts import TTSEngine, _get_slonet


def test_slonet_tuple_order():
    names = [cls.__name__ for cls in _get_slonet()]
    assert names == ["Tensor", "SloEmbedding", "SloLSTM", "SloLinear", "SloLayerNorm", "SloAdam"]


def test_tts_engine_constructs_with_real_optimizer():
    engine = TTSEngine()
    assert type(engine.optimizer).__name__ == "SloAdam"


def test_text_to_waveform_returns_audio():
    engine = TTSEngine()
    waveform = engine.text_to_waveform("Hello journey test.")
    assert isinstance(waveform, np.ndarray)
    assert len(waveform) > 0
    assert np.isfinite(waveform).all()
