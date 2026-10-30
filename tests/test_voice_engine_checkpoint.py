"""VoiceEngine TTS checkpoint wiring (voice slice 3).

Contract: if a trained checkpoint exists (env override or the default
models/tts path), first synthesize loads it and reports the path; any
missing/corrupt checkpoint silently falls back to untrained weights so
the speak journey never breaks.
"""

from __future__ import annotations

import numpy as np

from domain.voice._internal.tts import TTSEngine
from domain.voice.engine import VoiceEngine

ENV = "SLOUGHGPT_TTS_CHECKPOINT"
DEFAULT_PATH = "models/tts/espeak_distilled.npz"


def _offset_checkpoint(path) -> str:
    """Save a checkpoint whose weights differ from a fresh engine."""
    tts = TTSEngine()
    for p in tts.parameters():
        p.data[...] += 0.123
    return tts.save(str(path))


def test_env_checkpoint_loads_and_is_reported(monkeypatch, tmp_path):
    ckpt = _offset_checkpoint(tmp_path / "trained.npz")
    monkeypatch.setenv(ENV, ckpt)

    engine = VoiceEngine()
    result = engine.synthesize("Hello world")

    assert result.success, result.error
    assert engine.status()["tts_checkpoint"] == ckpt

    saved = np.load(ckpt)
    for name, p in engine._get_tts().decoder.named_parameters():
        assert np.allclose(p.data, saved[name]), f"weight not loaded: {name}"


def test_missing_checkpoint_falls_back_to_untrained(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "does_not_exist.npz"))

    engine = VoiceEngine()
    result = engine.synthesize("Hello world")

    assert result.success, result.error
    assert engine.status()["tts_checkpoint"] is None


def test_corrupt_checkpoint_falls_back_to_untrained(monkeypatch, tmp_path):
    bad = tmp_path / "corrupt.npz"
    bad.write_bytes(b"this is not an npz archive")
    monkeypatch.setenv(ENV, str(bad))

    engine = VoiceEngine()
    result = engine.synthesize("Hello world")

    assert result.success, result.error
    assert engine.status()["tts_checkpoint"] is None


def test_default_checkpoint_path_used_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    default = tmp_path / DEFAULT_PATH
    default.parent.mkdir(parents=True)
    _offset_checkpoint(default)
    monkeypatch.chdir(tmp_path)

    engine = VoiceEngine()
    result = engine.synthesize("Hello world")

    assert result.success, result.error
    assert engine.status()["tts_checkpoint"] == DEFAULT_PATH


def _rewrite_checkpoint(path, offset: float):
    """Overwrite a checkpoint with different weights + force an mtime change."""
    import os

    tts = TTSEngine()
    for p in tts.parameters():
        p.data[...] += offset
    tts.save(str(path))
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns + 1_000_000_000, st.st_mtime_ns + 1_000_000_000))


def test_synthesize_picks_up_rewritten_checkpoint(monkeypatch, tmp_path):
    ckpt = _offset_checkpoint(tmp_path / "gen1.npz")
    monkeypatch.setenv(ENV, ckpt)

    engine = VoiceEngine()
    assert engine.synthesize("First generation.").success
    assert engine.status()["tts_checkpoint"] == ckpt
    gen1 = [p.data.copy() for p in engine._get_tts().parameters()]

    _rewrite_checkpoint(ckpt, offset=-0.75)

    assert engine.synthesize("Second generation.").success
    for p, old in zip(engine._get_tts().parameters(), gen1):
        assert not np.allclose(p.data, old), "engine served stale weights after rewrite"


def test_corrupt_rewrite_keeps_serving_old_weights(monkeypatch, tmp_path):
    ckpt = _offset_checkpoint(tmp_path / "gen1.npz")
    monkeypatch.setenv(ENV, ckpt)

    engine = VoiceEngine()
    assert engine.synthesize("First generation.").success
    gen1 = [p.data.copy() for p in engine._get_tts().parameters()]

    import os

    orig_load = TTSEngine.load
    attempts: list[str] = []

    def spy_load(self, path):
        attempts.append(str(path))
        return orig_load(self, path)

    monkeypatch.setattr(TTSEngine, "load", spy_load)

    ckpt_path = tmp_path / "gen1.npz"
    ckpt_path.write_bytes(b"truncated mid-write garbage")
    st = os.stat(ckpt_path)
    os.utime(ckpt_path, ns=(st.st_atime_ns + 1_000_000_000, st.st_mtime_ns + 1_000_000_000))

    result = engine.synthesize("Serving continues through a bad reload.")
    assert result.success, result.error
    assert len(attempts) == 1, "engine must attempt the reload before falling back to old weights"
    for p, old in zip(engine._get_tts().parameters(), gen1):
        assert np.allclose(p.data, old), "corrupt rewrite must not clobber serving weights"
