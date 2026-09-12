"""
Audio Filter — voice sensitivity, noise gate, AGC, normalization.

Pure NumPy implementation for pre-processing microphone audio
before speech recognition. No external dependencies beyond NumPy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger("slo.multimodal.audio_filter")


class FilterMode(Enum):
    """Bitmask-style filter mode flags."""
    NONE = 0
    NOISE_GATE = 1 << 0
    AGC = 1 << 1
    NORMALIZE = 1 << 2
    VAD = 1 << 3
    ALL = NOISE_GATE | AGC | NORMALIZE | VAD

    def __and__(self, other: object) -> bool:
        if isinstance(other, FilterMode):
            return bool(self.value & other.value)
        return NotImplemented

    def __or__(self, other: object) -> int:
        if isinstance(other, FilterMode):
            return self.value | other.value
        return NotImplemented


@dataclass
class AudioFilterConfig:
    """Configuration for audio filtering pipeline."""
    sample_rate: int = 16000
    mode: FilterMode = FilterMode.ALL

    # Noise gate
    noise_gate_threshold_db: float = -40.0
    noise_gate_attack_ms: float = 5.0
    noise_gate_release_ms: float = 100.0

    # Automatic Gain Control
    agc_target_db: float = -20.0
    agc_max_gain_db: float = 30.0
    agc_frame_ms: float = 20.0

    # Loudness normalization
    target_lufs: float = -16.0

    # Voice Activity Detection
    vad_energy_threshold_db: float = -35.0
    vad_min_speech_ms: float = 250.0
    vad_silence_ratio: float = 0.6


@dataclass
class FilterResult:
    """Result from audio filtering."""
    audio: np.ndarray
    speech_detected: bool = True
    gain_applied_db: float = 0.0
    frames_gate_open: int = 0
    frames_total: int = 0


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    if linear <= 0:
        return -120.0
    return 20.0 * np.log10(linear)


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    work = audio.astype(np.float64)
    if np.max(np.abs(work)) > 1.0:
        work = work / 32768.0
    rms = np.sqrt(np.mean(work ** 2))
    return _linear_to_db(rms)


def _frame_energy_db(audio: np.ndarray, frame_size: int) -> np.ndarray:
    """Compute per-frame RMS energy in dB (normalizes int16 to float [-1,1])."""
    n_frames = max(1, len(audio) // frame_size)
    padded_len = n_frames * frame_size
    padded = np.zeros(padded_len, dtype=np.float64)
    padded[:len(audio)] = audio.astype(np.float64)
    if np.max(np.abs(padded)) > 1.0:
        padded = padded / 32768.0
    frames = padded[:padded_len].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms = np.maximum(rms, 1e-10)
    return 20.0 * np.log10(rms)


def _apply_envelope(
    audio: np.ndarray,
    gain_curve: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Apply a per-frame gain curve to audio."""
    n_frames = len(gain_curve)
    result = audio.astype(np.float64).copy()
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(audio))
        if start >= len(audio):
            break
        result[start:end] *= gain_curve[i]
    return result


def apply_noise_gate(
    audio: np.ndarray,
    config: AudioFilterConfig,
) -> tuple[np.ndarray, int, int]:
    """
    Noise gate: suppress audio below threshold.

    Uses a simple RMS-based gate with attack/release smoothing.

    Returns:
        (filtered_audio, open_frame_count, total_frames)
    """
    frame_size = int(config.sample_rate * config.agc_frame_ms / 1000)
    energy_db = _frame_energy_db(audio, frame_size)

    threshold = config.noise_gate_threshold_db
    attack_frames = max(1, int(config.noise_gate_attack_ms / config.agc_frame_ms))
    release_frames = max(1, int(config.noise_gate_release_ms / config.agc_frame_ms))

    n_frames = len(energy_db)
    gate_open = np.zeros(n_frames, dtype=np.float64)

    state = False
    counter = 0

    for i in range(n_frames):
        if energy_db[i] > threshold:
            if not state:
                counter += 1
                if counter >= attack_frames:
                    state = True
                    counter = 0
            else:
                counter = 0
            gate_open[i] = 1.0 if state else 0.0
        else:
            if state:
                counter += 1
                if counter >= release_frames:
                    state = False
                    counter = 0
            else:
                counter = 0
            gate_open[i] = 1.0 if state else 0.0

    smoothed = np.convolve(gate_open, np.ones(3) / 3, mode="same")
    smoothed = np.clip(smoothed, 0.0, 1.0)

    filtered = _apply_envelope(audio, smoothed, frame_size)
    open_count = int(np.sum(gate_open > 0.5))
    return filtered.astype(audio.dtype), open_count, n_frames


def apply_agc(
    audio: np.ndarray,
    config: AudioFilterConfig,
) -> tuple[np.ndarray, float]:
    """
    Automatic Gain Control: normalize audio to target loudness.

    Returns:
        (filtered_audio, gain_applied_db)
    """
    frame_size = int(config.sample_rate * config.agc_frame_ms / 1000)
    energy_db = _frame_energy_db(audio, frame_size)

    target = config.agc_target_db
    max_gain = _db_to_linear(config.agc_max_gain_db)

    gains_db = target - energy_db
    gains_linear = np.clip(
        np.vectorize(_db_to_linear)(gains_db),
        0.0,
        max_gain,
    )

    kernel_size = max(3, int(50 / config.agc_frame_ms))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    gains_linear = np.convolve(gains_linear, kernel, mode="same")

    filtered = _apply_envelope(audio, gains_linear, frame_size)
    avg_gain_db = float(np.mean(gains_db))
    return filtered.astype(audio.dtype), avg_gain_db


def normalize_loudness(
    audio: np.ndarray,
    config: AudioFilterConfig,
) -> np.ndarray:
    """
    Normalize audio to target LUFS using peak normalization.
    """
    if audio.size == 0:
        return audio
    work = audio.astype(np.float64)
    peak = float(np.max(np.abs(work)))
    if peak < 1e-10:
        return audio
    work = work / peak
    target_peak = _db_to_linear(config.target_lufs)
    work = work * target_peak
    work = work * 32768.0
    return work.clip(-32768, 32767).astype(np.int16)


def detect_voice_activity(
    audio: np.ndarray,
    config: AudioFilterConfig,
) -> tuple[bool, float]:
    """
    Detect whether audio contains speech.

    Returns:
        (speech_detected, silence_ratio)
    """
    frame_size = int(config.sample_rate * config.agc_frame_ms / 1000)
    energy_db = _frame_energy_db(audio, frame_size)

    speech_frames = np.sum(energy_db > config.vad_energy_threshold_db)
    total_frames = max(1, len(energy_db))
    silence_ratio = 1.0 - (speech_frames / total_frames)

    min_speech_frames = max(1, int(config.vad_min_speech_ms / config.agc_frame_ms))
    has_speech = speech_frames >= min_speech_frames
    dominated_by_silence = silence_ratio > config.vad_silence_ratio

    detected = has_speech and not dominated_by_silence
    return detected, float(silence_ratio)


def apply_audio_filter(
    audio: np.ndarray,
    config: AudioFilterConfig | None = None,
) -> FilterResult:
    """
    Full audio filter pipeline.

    Applies (in order): noise gate -> AGC -> normalization -> VAD.

    Args:
        audio: Raw 16-bit PCM audio (int16 or float32)
        config: Filter configuration

    Returns:
        FilterResult with filtered audio and metadata
    """
    if config is None:
        config = AudioFilterConfig()

    if audio.size == 0:
        return FilterResult(audio=audio, speech_detected=False)

    work = audio.astype(np.float64)

    open_count = 0
    total_frames = 0

    if config.mode & FilterMode.NOISE_GATE:
        work, open_count, total_frames = apply_noise_gate(work, config)

    if config.mode & FilterMode.AGC:
        work, gain_db = apply_agc(work, config)
    else:
        gain_db = 0.0

    if config.mode & FilterMode.NORMALIZE:
        work = normalize_loudness(work, config)

    speech_detected = True
    if config.mode & FilterMode.VAD:
        speech_detected, _ = detect_voice_activity(work, config)

    result = work.astype(np.float32 if audio.dtype == np.float32 else np.int16)
    return FilterResult(
        audio=result,
        speech_detected=speech_detected,
        gain_applied_db=gain_db,
        frames_gate_open=open_count,
        frames_total=total_frames,
    )


__all__ = [
    "FilterMode",
    "AudioFilterConfig",
    "FilterResult",
    "apply_noise_gate",
    "apply_agc",
    "normalize_loudness",
    "detect_voice_activity",
    "apply_audio_filter",
]
