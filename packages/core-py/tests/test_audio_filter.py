"""Tests for multimodal.audio_filter — audio filtering pipeline."""

from __future__ import annotations

import pytest
import numpy as np

from domains.multimodal.audio_filter import (
    FilterMode,
    AudioFilterConfig,
    FilterResult,
    _db_to_linear,
    _linear_to_db,
    _rms_db,
    _frame_energy_db,
    apply_noise_gate,
    apply_agc,
    normalize_loudness,
    detect_voice_activity,
    apply_audio_filter,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_speech_audio(
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    amplitude: float = 10000.0,
    freq: float = 200.0,
) -> np.ndarray:
    """Generate a synthetic speech-like signal (sinusoid)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def _make_silence_audio(
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    noise_level: float = 10.0,
) -> np.ndarray:
    """Generate near-silence audio."""
    n = int(sample_rate * duration_s)
    return np.random.RandomState(42).randn(n).astype(np.float32) * noise_level


# ── Conversion helpers ──────────────────────────────────────────────────────


class TestConversionHelpers:

    def test_db_to_linear_0db(self):
        assert abs(_db_to_linear(0.0) - 1.0) < 1e-6

    def test_db_to_linear_negative(self):
        assert _db_to_linear(-20.0) == pytest.approx(0.1, rel=1e-3)

    def test_db_to_linear_positive(self):
        assert _db_to_linear(20.0) == pytest.approx(10.0, rel=1e-3)

    def test_linear_to_db_1(self):
        assert _linear_to_db(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_linear_to_db_0(self):
        assert _linear_to_db(0.0) == -120.0

    def test_linear_to_db_small(self):
        assert _linear_to_db(0.1) == pytest.approx(-20.0, rel=1e-3)

    def test_rms_db_silence(self):
        audio = np.zeros(1000, dtype=np.int16)
        assert _rms_db(audio) == -120.0

    def test_rms_db_sine(self):
        audio = _make_speech_audio(duration_s=0.1, amplitude=10000)
        db = _rms_db(audio)
        # amplitude 10000/32768 ≈ 0.305, RMS of sine ≈ 0.216, dB ≈ -13.3
        assert -20.0 < db < -5.0

    def test_frame_energy_db(self):
        audio = _make_speech_audio(duration_s=0.1, amplitude=10000)
        frame_size = 320  # 20ms at 16kHz
        energy = _frame_energy_db(audio, frame_size)
        assert energy.shape[0] > 0
        assert all(e > -120.0 for e in energy)


# ── Noise Gate ──────────────────────────────────────────────────────────────


class TestNoiseGate:

    def test_gate_passes_loud_audio(self):
        audio = _make_speech_audio(amplitude=10000)
        config = AudioFilterConfig(noise_gate_threshold_db=-40.0)
        result, open_count, total = apply_noise_gate(audio, config)
        assert result.dtype == audio.dtype
        assert len(result) == len(audio)
        assert open_count > 0

    def test_gate_blocks_silence(self):
        audio = _make_silence_audio(noise_level=0.001)
        config = AudioFilterConfig(noise_gate_threshold_db=-40.0)
        result, open_count, total = apply_noise_gate(audio, config)
        assert open_count == 0

    def test_gate_empty(self):
        audio = np.array([], dtype=np.int16)
        config = AudioFilterConfig()
        result, open_count, total = apply_noise_gate(audio, config)
        assert len(result) == 0


# ── AGC ─────────────────────────────────────────────────────────────────────


class TestAGC:

    def test_agc_boosts_quiet(self):
        audio = _make_speech_audio(amplitude=100)
        config = AudioFilterConfig(agc_target_db=-20.0)
        result, gain_db = apply_agc(audio, config)
        assert gain_db > 0
        assert np.max(np.abs(result)) > np.max(np.abs(audio))

    def test_agc_reduces_loud(self):
        audio = _make_speech_audio(amplitude=30000)
        config = AudioFilterConfig(agc_target_db=-20.0)
        result, gain_db = apply_agc(audio, config)
        assert gain_db < 0

    def test_agc_empty(self):
        audio = np.array([], dtype=np.int16)
        config = AudioFilterConfig()
        result, gain_db = apply_agc(audio, config)
        assert len(result) == 0


# ── Loudness Normalization ─────────────────────────────────────────────────


class TestLoudnessNormalization:

    def test_normalize(self):
        audio = _make_speech_audio(amplitude=5000)
        config = AudioFilterConfig(target_lufs=-16.0)
        result = normalize_loudness(audio, config)
        assert result.dtype == np.int16
        peak = np.max(np.abs(result))
        # target_peak = 0.158, so peak should be ~5179 (0.158 * 32768)
        assert 4000 < peak < 6000

    def test_normalize_already_loud(self):
        audio = _make_speech_audio(amplitude=30000)
        config = AudioFilterConfig(target_lufs=-16.0)
        result = normalize_loudness(audio, config)
        peak = np.max(np.abs(result))
        assert peak <= 32767

    def test_normalize_empty(self):
        audio = np.array([], dtype=np.int16)
        result = normalize_loudness(audio, AudioFilterConfig())
        assert len(result) == 0

    def test_normalize_silence(self):
        audio = np.zeros(1000, dtype=np.int16)
        result = normalize_loudness(audio, AudioFilterConfig())
        assert np.all(result == 0)


# ── VAD ─────────────────────────────────────────────────────────────────────


class TestVAD:

    def test_speech_detected(self):
        audio = _make_speech_audio(amplitude=10000, duration_s=0.5)
        config = AudioFilterConfig()
        detected, silence_ratio = detect_voice_activity(audio, config)
        assert detected is True
        assert silence_ratio < 0.5

    def test_silence_not_detected(self):
        audio = _make_silence_audio(noise_level=0.001)
        config = AudioFilterConfig()
        detected, silence_ratio = detect_voice_activity(audio, config)
        assert detected == False
        assert silence_ratio > 0.5

    def test_empty_audio(self):
        audio = np.array([], dtype=np.int16)
        config = AudioFilterConfig()
        detected, silence_ratio = detect_voice_activity(audio, config)
        assert detected == False
        assert silence_ratio == 1.0


# ── Full Pipeline ───────────────────────────────────────────────────────────


class TestApplyAudioFilter:

    def test_full_pipeline_speech(self):
        audio = _make_speech_audio(amplitude=8000, duration_s=0.5)
        result = apply_audio_filter(audio)
        assert isinstance(result, FilterResult)
        assert result.audio.dtype == audio.dtype
        assert len(result.audio) == len(audio)

    def test_full_pipeline_silence(self):
        audio = _make_silence_audio(noise_level=0.001)
        result = apply_audio_filter(audio)
        assert result.speech_detected == False

    def test_full_pipeline_empty(self):
        audio = np.array([], dtype=np.int16)
        result = apply_audio_filter(audio)
        assert result.speech_detected == False

    def test_full_pipeline_none_config(self):
        audio = _make_speech_audio(amplitude=5000)
        result = apply_audio_filter(audio, config=None)
        assert isinstance(result, FilterResult)

    def test_pipeline_noise_gate_only(self):
        audio = _make_speech_audio(amplitude=10000)
        config = AudioFilterConfig(mode=FilterMode.NOISE_GATE)
        result = apply_audio_filter(audio, config)
        assert result.frames_total > 0

    def test_pipeline_agc_only(self):
        audio = _make_speech_audio(amplitude=100)
        config = AudioFilterConfig(mode=FilterMode.AGC)
        result = apply_audio_filter(audio, config)
        assert result.gain_applied_db != 0.0

    def test_pipeline_vad_only(self):
        audio = _make_silence_audio(noise_level=0.001)
        config = AudioFilterConfig(mode=FilterMode.VAD)
        result = apply_audio_filter(audio, config)
        assert result.speech_detected == False

    def test_pipeline_normalize_only(self):
        audio = _make_speech_audio(amplitude=5000)
        config = AudioFilterConfig(mode=FilterMode.NORMALIZE)
        result = apply_audio_filter(audio, config)
        assert len(result.audio) == len(audio)

    def test_pipeline_all_disabled(self):
        audio = _make_speech_audio(amplitude=5000)
        config = AudioFilterConfig(mode=FilterMode.NONE)
        result = apply_audio_filter(audio, config)
        assert result.gain_applied_db == 0.0
        assert result.frames_gate_open == 0


# ── FilterMode ──────────────────────────────────────────────────────────────


class TestFilterMode:

    def test_all_combines_flags(self):
        assert FilterMode.ALL.value == (
            FilterMode.NOISE_GATE.value
            | FilterMode.AGC.value
            | FilterMode.NORMALIZE.value
            | FilterMode.VAD.value
        )

    def test_individual_flags(self):
        assert FilterMode.NOISE_GATE.value != FilterMode.AGC.value
        assert FilterMode.NORMALIZE.value != FilterMode.VAD.value


# ── FilterResult ────────────────────────────────────────────────────────────


class TestFilterResult:

    def test_defaults(self):
        r = FilterResult(audio=np.zeros(10, dtype=np.int16))
        assert r.speech_detected is True
        assert r.gain_applied_db == 0.0
        assert r.frames_gate_open == 0
        assert r.frames_total == 0

    def test_custom(self):
        r = FilterResult(
            audio=np.zeros(10, dtype=np.int16),
            speech_detected=False,
            gain_applied_db=-5.0,
            frames_gate_open=3,
            frames_total=10,
        )
        assert r.speech_detected is False
        assert r.gain_applied_db == -5.0
