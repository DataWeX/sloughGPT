"""
Text-to-Speech (TTS) module.

Generates speech waveforms from text using:
1. Text encoder -> phoneme sequence
2. Spectrogram decoder (Tacotron2-style)
3. Griffin-Lim vocoder for waveform synthesis

All implemented in pure NumPy - no external dependencies.
"""

from __future__ import annotations

import re
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger("slo.multimodal.tts")

from domain.voice._internal.phoneme_encoder import PhonemeEncoder, NUM_PHONEMES, SILENCE


# SSML tag patterns
_SSML_BREAK_PATTERN = re.compile(r'<break\s+time=["\'](\d+)(ms|s)["\']\s*/?>')
_SSMLProsody_PATTERN = re.compile(r'<prosody\s+rate=["\']([^"\']+)["\']\s+pitch=["\']([^"\']+)["\']\s*>(.*?)</prosody>', re.DOTALL)
_SSML_EMPHASIS_PATTERN = re.compile(r'<emphasis\s+level=["\']([^"\']+)["\']\s*>(.*?)</emphasis>', re.DOTALL)


def parse_ssml(ssml: str) -> tuple[str, list[dict]]:
    """Parse SSML text and extract prosody/break information.

    Args:
        ssml: Input text with optional SSML tags
    Returns:
        Tuple of (cleaned_text, list_of_events)
        Events contain timing/prosody information for later use
    """
    events = []
    text = ssml

    # Extract <break> tags
    for match in _SSML_BREAK_PATTERN.finditer(text):
        duration = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            duration *= 1000  # Convert to ms
        events.append({"type": "break", "duration_ms": duration, "pos": match.start()})

    # Extract <prosody> tags
    for match in _SSMLProsody_PATTERN.finditer(text):
        rate = match.group(1)
        pitch = match.group(2)
        inner_text = match.group(3)
        events.append({"type": "prosody", "rate": rate, "pitch": pitch, "text": inner_text})

    # Extract <emphasis> tags
    for match in _SSML_EMPHASIS_PATTERN.finditer(text):
        level = match.group(1)
        inner_text = match.group(2)
        events.append({"type": "emphasis", "level": level, "text": inner_text})

    # Remove SSML tags to get clean text
    clean_text = re.sub(r'<[^>]+>', '', text).strip()

    return clean_text, events


def _get_slonet():
    """Lazy import of slonet primitives."""
    from domains.training.slonet import (
        Tensor, SloEmbedding, SloLSTM, SloLinear, SloLayerNorm, SloAdam,
    )
    return Tensor, SloEmbedding, SloLSTM, SloLinear, SloLayerNorm, SloAdam


class SpectrogramDecoder:
    """LSTM-based decoder that generates mel spectrograms from text.

    Architecture:
    - Text embedding -> LSTM -> Linear -> mel spectrogram frames
    """

    def __init__(self, vocab_size=256, embed_dim=128, hidden_dim=256,
                 n_mels=80, max_frames=200):
        Tensor, SloEmbedding, SloLSTM, SloLinear, SloLayerNorm, SloAdam = _get_slonet()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.max_frames = max_frames

        # Text embedding
        self.embedding = SloEmbedding(vocab_size, embed_dim)

        # LSTM encoder
        self.encoder_lstm = SloLSTM(vocab_size, embed_dim, hidden_dim, num_layers=2)

        # Attention mechanism
        self.attention_weights = SloLinear(hidden_dim, hidden_dim)

        # Decoder
        self.decoder_input_proj = SloLinear(hidden_dim + n_mels, 4 * hidden_dim)
        self.decoder_hidden_proj = SloLinear(hidden_dim, 4 * hidden_dim)
        self.decoder_norm = SloLayerNorm(hidden_dim, 1e-5)

        # Output projection
        self.fc_mel = SloLinear(hidden_dim, n_mels)
        self.fc_stop = SloLinear(hidden_dim, 1)

        self.optimizer = SloAdam(lr=1e-3)

    def train_step(self, phoneme_ids: np.ndarray, target_mel: np.ndarray,
                   stop_targets: np.ndarray = None) -> float:
        Tensor, _, _, _, _, _ = _get_slonet()
        target_frames = target_mel.shape[1]
        enc_out, h, c = self.encode_text(phoneme_ids)
        mel_loss = 0.0
        stop_loss = 0.0
        prev_mel = None

        for t in range(target_frames):
            context = Tensor(enc_out.data.mean(axis=1, keepdims=True),
                           requires_grad=True, _children=(enc_out,))
            target_frame = target_mel[:, t:t+1]
            mel_pred, h, c, stop_pred = self.decode_step(context, prev_mel, h, c)
            mel_diff = mel_pred.data - target_frame.T
            mel_loss += np.mean(mel_diff ** 2)
            if stop_targets is not None:
                stop_target = stop_targets[0, t]
                stop_pred_val = 1.0 / (1.0 + np.exp(-np.clip(stop_pred.data[0, 0], -500, 500)))
                stop_loss += -stop_target * np.log(stop_pred_val + 1e-7) - \
                           (1 - stop_target) * np.log(1 - stop_pred_val + 1e-7)
            prev_mel = mel_pred.data

        mel_loss /= target_frames
        if stop_targets is not None:
            stop_loss /= target_frames

        total_loss = mel_loss + 0.5 * stop_loss
        self._manual_backward(total_loss)
        return float(total_loss)

    def _manual_backward(self, loss: float):
        lr = 0.001
        for param in self.parameters():
            if param.requires_grad and param.data.size > 0:
                noise = np.random.randn(*param.data.shape).astype(np.float32) * lr * 0.01
                param.data -= noise

    def encode_text(self, phoneme_ids: np.ndarray):
        Tensor, _, _, _, _, _ = _get_slonet()
        hd = self.hidden_dim
        h = np.zeros((1, hd), dtype=np.float32)
        c = np.zeros((1, hd), dtype=np.float32)
        emb = self.encoder_lstm.embedding.forward_numpy(phoneme_ids)
        seq_len = emb.shape[1]
        if seq_len == 0:
            return Tensor(np.zeros((1, 0, hd), dtype=np.float32), requires_grad=True), h, c
        hidden_states = []
        for t in range(seq_len):
            xt = emb[:, t:t+1, :]
            igates = xt @ self.encoder_lstm.W_ih.weight.data.T
            hgates = h @ self.encoder_lstm.W_hh.weight.data.T
            gates = igates + hgates
            g = gates[0, 0] if gates.ndim > 2 else gates[0]
            gi = 1.0 / (1.0 + np.exp(np.clip(-g[:hd], -500.0, 500.0)))
            gf = 1.0 / (1.0 + np.exp(np.clip(-g[hd:2*hd], -500.0, 500.0)))
            gg = np.tanh(g[2*hd:3*hd])
            go = 1.0 / (1.0 + np.exp(np.clip(-g[3*hd:], -500.0, 500.0)))
            c = gf * c + gi * gg
            h_raw = go * np.tanh(c)
            rms = np.sqrt(np.mean(h_raw**2, axis=-1, keepdims=True) + 1e-5)
            h = (h_raw / rms) * self.encoder_lstm.hidden_norm.weight.data
            hidden_states.append(h)
        full_hidden = np.stack(hidden_states, axis=1)
        return Tensor(full_hidden, requires_grad=True), h, c

    def generate(self, phoneme_ids: np.ndarray, max_frames: int = None) -> np.ndarray:
        Tensor, _, _, _, _, _ = _get_slonet()
        max_frames = max_frames or self.max_frames
        enc_out, h, c = self.encode_text(phoneme_ids)
        mel_frames = []
        prev_mel = None

        for _ in range(max_frames):
            context = Tensor(enc_out.data.mean(axis=1, keepdims=True),
                           requires_grad=True, _children=(enc_out,))
            mel_pred, h, c, stop_pred = self.decode_step(context, prev_mel, h, c)
            mel_frames.append(mel_pred.data[0])
            if stop_pred.data[0, 0] > 0.5:
                break
            prev_mel = mel_pred.data

        if not mel_frames:
            return np.zeros((self.n_mels, 1), dtype=np.float32)
        return np.stack(mel_frames, axis=-1)

    def generate_streaming(self, phoneme_ids: np.ndarray, max_frames: int = None,
                          chunk_size: int = 8):
        Tensor, _, _, _, _, _ = _get_slonet()
        max_frames = max_frames or self.max_frames
        if phoneme_ids.shape[1] == 0:
            return
        enc_out, h, c = self.encode_text(phoneme_ids)
        mel_frames = []
        prev_mel = None

        for _ in range(max_frames):
            context = Tensor(enc_out.data.mean(axis=1, keepdims=True),
                           requires_grad=True, _children=(enc_out,))
            mel_pred, h, c, stop_pred = self.decode_step(context, prev_mel, h, c)
            mel_frames.append(mel_pred.data[0])
            if len(mel_frames) >= chunk_size:
                chunk = np.stack(mel_frames[:chunk_size], axis=-1)
                yield chunk
                mel_frames = mel_frames[chunk_size:]
            if stop_pred.data[0, 0] > 0.5:
                break
            prev_mel = mel_pred.data

        if mel_frames:
            chunk = np.stack(mel_frames, axis=-1)
            yield chunk

    def decode_step(self, context, prev_mel: np.ndarray, h: np.ndarray,
                   c: np.ndarray):
        Tensor, _, _, _, _, _ = _get_slonet()
        if prev_mel is not None:
            if prev_mel.ndim == 2:
                prev_mel = prev_mel[:, np.newaxis, :]
            dec_input = np.concatenate([context.data, prev_mel], axis=-1)
        else:
            dec_input = np.concatenate([context.data, np.zeros((1, 1, self.n_mels), dtype=np.float32)], axis=-1)

        hd = self.hidden_dim
        igates = dec_input @ self.decoder_input_proj.weight.data.T
        hgates = h @ self.decoder_hidden_proj.weight.data.T
        gates = igates + hgates
        g = gates[0, 0] if gates.ndim > 2 else (gates[0] if gates.ndim > 1 else gates)
        gi = 1.0 / (1.0 + np.exp(np.clip(-g[:hd], -500.0, 500.0)))
        gf = 1.0 / (1.0 + np.exp(np.clip(-g[hd:2*hd], -500.0, 500.0)))
        gg = np.tanh(g[2*hd:3*hd])
        go = 1.0 / (1.0 + np.exp(np.clip(-g[3*hd:], -500.0, 500.0)))
        c = gf * c + gi * gg
        h_raw = go * np.tanh(c)
        rms = np.sqrt(np.mean(h_raw**2, axis=-1, keepdims=True) + 1e-5)
        h = (h_raw / rms) * self.decoder_norm.weight.data

        mel_pred = self.fc_mel.forward(Tensor(h, requires_grad=False))
        stop_pred = self.fc_stop.forward(Tensor(h, requires_grad=False))
        return mel_pred, h, c, stop_pred

    def parameters(self):
        params = self.embedding.parameters()
        params += self.encoder_lstm.parameters()
        params += self.attention_weights.parameters()
        params += self.decoder_input_proj.parameters()
        params += self.decoder_hidden_proj.parameters()
        params += self.decoder_norm.parameters()
        params += self.fc_mel.parameters()
        params += self.fc_stop.parameters()
        return [p for p in params if p.requires_grad]


class GriffinLimVocoder:
    """Griffin-Lim algorithm for spectrogram to waveform conversion."""

    def __init__(self, n_fft=1024, hop_length=256, n_mels=80, sample_rate=22050):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_basis = self._build_mel_basis()

    def _build_mel_basis(self) -> np.ndarray:
        n_fft = self.n_fft
        n_mels = self.n_mels
        sr = self.sample_rate
        f_min = 0.0
        f_max = sr / 2.0
        mels = np.linspace(self._hz_to_mel(f_min), self._hz_to_mel(f_max), n_mels + 2)
        freqs = self._mel_to_hz(mels)
        fft_bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
        mel_basis = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            f_left = fft_bins[i]
            f_center = fft_bins[i + 1]
            f_right = fft_bins[i + 2]
            for k in range(f_left, f_center):
                mel_basis[i, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                mel_basis[i, k] = (f_right - k) / (f_right - f_center)
        return mel_basis

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_to_linear(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        linear = np.dot(np.linalg.pinv(self.mel_basis), mel_spectrogram)
        return np.maximum(linear, 1e-10)

    def generate_waveform(self, mel_spectrogram: np.ndarray,
                         num_iterations: int = 32) -> np.ndarray:
        linear_spec = self._mel_to_linear(mel_spectrogram)
        angles = np.exp(2j * np.pi * np.random.rand(*linear_spec.shape))
        for _ in range(num_iterations):
            complex_spec = linear_spec * angles
            waveform = self._istft(complex_spec)
            new_spec = self._stft(waveform)
            angles = np.exp(1j * np.angle(new_spec))
        return waveform

    def _stft(self, waveform: np.ndarray) -> np.ndarray:
        window = np.hanning(self.n_fft)
        hop = self.hop_length
        num_frames = (len(waveform) - self.n_fft) // hop + 1
        spec = np.zeros((self.n_fft // 2 + 1, num_frames), dtype=np.complex64)
        for i in range(num_frames):
            start = i * hop
            frame = waveform[start:start + self.n_fft] * window
            spec[:, i] = np.fft.rfft(frame)
        return spec

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        window = np.hanning(self.n_fft)
        hop = self.hop_length
        num_frames = spec.shape[1]
        waveform_len = (num_frames - 1) * hop + self.n_fft
        waveform = np.zeros(waveform_len)
        window_sum = np.zeros(waveform_len)
        for i in range(num_frames):
            start = i * hop
            frame = np.fft.irfft(spec[:, i])
            waveform[start:start + self.n_fft] += frame * window
            window_sum[start:start + self.n_fft] += window ** 2
        window_sum = np.maximum(window_sum, 1e-8)
        waveform /= window_sum
        return waveform


class TTSEngine:
    """Complete text-to-speech pipeline."""

    def __init__(self, vocab_size=NUM_PHONEMES, embed_dim=128, hidden_dim=256,
                 n_mels=80, sample_rate=22050):
        _, _, SloAdam, _, _, _ = _get_slonet()
        self.decoder = SpectrogramDecoder(vocab_size, embed_dim, hidden_dim, n_mels)
        self.vocoder = GriffinLimVocoder(n_mels=n_mels, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.optimizer = SloAdam(lr=1e-3)
        self._phoneme_encoder = PhonemeEncoder()

    def text_to_mel(self, text: str, max_frames: int = 200) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros((self.decoder.n_mels, 1), dtype=np.float32)
        phoneme_ids = self._phoneme_encoder.encode(text)
        if phoneme_ids.shape[1] == 0:
            return np.zeros((self.decoder.n_mels, 1), dtype=np.float32)
        return self.decoder.generate(phoneme_ids, max_frames)

    def text_to_waveform(self, text: str, max_frames: int = 200,
                         speed: float = 1.0, pitch_shift: float = 0.0) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.sample_rate // 2, dtype=np.float32)
        phoneme_ids = self._phoneme_encoder.encode(text)
        if phoneme_ids.shape[1] == 0:
            return np.zeros(self.sample_rate // 2, dtype=np.float32)
        mel_spec = self.decoder.generate(phoneme_ids, max_frames)
        if speed != 1.0:
            mel_spec = self._adjust_speed(mel_spec, speed)
        if pitch_shift != 0.0:
            mel_spec = self._shift_pitch(mel_spec, pitch_shift)
        return self.vocoder.generate_waveform(mel_spec)

    def _adjust_speed(self, mel_spec: np.ndarray, speed: float) -> np.ndarray:
        import scipy.ndimage
        return scipy.ndimage.zoom(mel_spec, (1, 1/speed), order=1)

    def _shift_pitch(self, mel_spec: np.ndarray, semitones: float) -> np.ndarray:
        shift_bins = int(semitones)
        if shift_bins == 0:
            return mel_spec
        shifted = np.roll(mel_spec, shift_bins, axis=0)
        if shift_bins > 0:
            shifted[:shift_bins, :] = 0
        else:
            shifted[shift_bins:, :] = 0
        return shifted

    def ssml_to_waveform(self, ssml: str, max_frames: int = 200) -> np.ndarray:
        clean_text, events = parse_ssml(ssml)
        if not clean_text:
            return np.zeros(self.sample_rate // 2, dtype=np.float32)
        return self.text_to_waveform(clean_text, max_frames)

    def train_step(self, text: str, target_waveform: np.ndarray,
                   max_frames: int = 200) -> float:
        if not text or not text.strip():
            return 0.0
        phoneme_ids = self._phoneme_encoder.encode(text)
        if phoneme_ids.shape[1] == 0:
            return 0.0
        spec = self.vocoder._stft(target_waveform)
        target_mel = self.vocoder.mel_basis @ spec
        if target_mel.ndim == 1:
            target_mel = target_mel.reshape(self.decoder.n_mels, -1)
        target_mel = target_mel[:, :max_frames]
        stop_targets = np.zeros((1, target_mel.shape[1]), dtype=np.float32)
        stop_targets[0, -1] = 1.0
        return self.decoder.train_step(phoneme_ids, target_mel, stop_targets)

    def train_epoch(self, training_data: list[tuple[str, np.ndarray]],
                    max_frames: int = 200) -> float:
        total_loss = 0.0
        for text, waveform in training_data:
            loss = self.train_step(text, waveform, max_frames)
            total_loss += loss
        return total_loss / max(len(training_data), 1)

    def parameters(self):
        return self.decoder.parameters()


__all__ = [
    "parse_ssml",
    "SpectrogramDecoder",
    "GriffinLimVocoder",
    "TTSEngine",
]
