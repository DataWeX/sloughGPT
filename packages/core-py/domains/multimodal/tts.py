"""
Text-to-Speech (TTS) module.

Generates speech waveforms from text using:
1. Text encoder -> phoneme sequence
2. Spectrogram decoder (Tacotron2-style)
3. Griffin-Lim vocoder for waveform synthesis

All implemented in pure NumPy - no external dependencies.
"""

from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger("slo.multimodal.tts")

from domains.training.slonet import (
    Tensor, SloNet, SloEmbedding, SloLSTM, SloLinear, SloLayerNorm,
    SloAdam, relu as _relu, sigmoid as _sigmoid,
    tensor as _tensor,
)


class SpectrogramDecoder:
    """LSTM-based decoder that generates mel spectrograms from text.

    Architecture:
    - Text embedding -> LSTM -> Linear -> mel spectrogram frames
    """

    def __init__(self, vocab_size=256, embed_dim=128, hidden_dim=256,
                 n_mels=80, max_frames=200):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        self.max_frames = max_frames

        # Text embedding
        self.embedding = SloEmbedding(vocab_size, embed_dim)

        # LSTM encoder: SloLSTM(vocab_size, embed_dim, hidden_dim)
        self.encoder_lstm = SloLSTM(vocab_size, embed_dim, hidden_dim, num_layers=2)

        # Attention mechanism
        self.attention_weights = SloLinear(hidden_dim, hidden_dim)

        # Decoder — use raw SloLinear layers instead of SloLSTM (which expects token IDs)
        self.decoder_input_proj = SloLinear(hidden_dim + n_mels, 4 * hidden_dim)
        self.decoder_hidden_proj = SloLinear(hidden_dim, 4 * hidden_dim)
        self.decoder_norm = SloLayerNorm(hidden_dim, 1e-5)

        # Output projection
        self.fc_mel = SloLinear(hidden_dim, n_mels)
        self.fc_stop = SloLinear(hidden_dim, 1)  # Stop token prediction

        self.optimizer = SloAdam(lr=1e-3)

    def encode_text(self, phoneme_ids: np.ndarray) -> Tensor:
        """Encode phoneme sequence to hidden states."""
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
        """
        Generate mel spectrogram from phonemes.

        Args:
            phoneme_ids: (1, seq_len) phoneme IDs
            max_frames: Maximum number of frames to generate
        Returns:
            mel_spectrogram: (n_mels, num_frames)
        """
        max_frames = max_frames or self.max_frames

        # Encode text
        enc_out, h, c = self.encode_text(phoneme_ids)

        # Decode autoregressively
        mel_frames = []
        prev_mel = None

        for _ in range(max_frames):
            # Simple attention: average encoder outputs
            context = Tensor(enc_out.data.mean(axis=1, keepdims=True),
                           requires_grad=True, _children=(enc_out,))

            mel_pred, h, c, stop_pred = self.decode_step(context, prev_mel, h, c)

            mel_frames.append(mel_pred.data[0])  # Remove batch dim

            # Check stop condition
            if stop_pred.data[0, 0] > 0.5:
                break

            prev_mel = mel_pred.data

        if not mel_frames:
            return np.zeros((self.n_mels, 1), dtype=np.float32)

        mel_spectrogram = np.stack(mel_frames, axis=-1)
        return mel_spectrogram

    def decode_step(self, context: Tensor, prev_mel: np.ndarray, h: np.ndarray,
                   c: np.ndarray) -> Tuple[Tensor, np.ndarray, np.ndarray, Tensor]:
        """Single decoding step with raw LSTM cell."""
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
    """Griffin-Lim algorithm for spectrogram to waveform conversion.

    Iteratively reconstructs phase from magnitude spectrogram.
    """

    def __init__(self, n_fft=1024, hop_length=256, n_mels=80, sample_rate=22050):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Mel filterbank
        self.mel_basis = self._build_mel_basis()

    def _build_mel_basis(self) -> np.ndarray:
        """Build mel filterbank matrix."""
        n_fft = self.n_fft
        n_mels = self.n_mels
        sr = self.sample_rate

        # Frequency bins
        f_min = 0.0
        f_max = sr / 2.0
        mels = np.linspace(
            self._hz_to_mel(f_min),
            self._hz_to_mel(f_max),
            n_mels + 2
        )
        freqs = self._mel_to_hz(mels)

        # FFT bin indices
        fft_bins = np.floor((n_fft + 1) * freqs / sr).astype(int)

        # Build filterbank
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
        """Convert mel spectrogram to linear spectrogram."""
        # Inverse mel filterbank (pseudo-inverse)
        linear = np.dot(np.linalg.pinv(self.mel_basis), mel_spectrogram)
        return np.maximum(linear, 1e-10)

    def generate_waveform(self, mel_spectrogram: np.ndarray,
                         num_iterations: int = 32) -> np.ndarray:
        """
        Convert mel spectrogram to waveform using Griffin-Lim.

        Args:
            mel_spectrogram: (n_mels, num_frames)
            num_iterations: Number of GL iterations
        Returns:
            waveform: (num_samples,) audio waveform
        """
        # Convert to linear spectrogram
        linear_spec = self._mel_to_linear(mel_spectrogram)

        # Initialize random phase
        angles = np.exp(2j * np.pi * np.random.rand(*linear_spec.shape))

        # Griffin-Lim iterations
        for _ in range(num_iterations):
            # Reconstruct complex spectrogram
            complex_spec = linear_spec * angles

            # Inverse STFT
            waveform = self._istft(complex_spec)

            # Forward STFT to get new phase
            new_spec = self._stft(waveform)
            angles = np.exp(1j * np.angle(new_spec))

        return waveform

    def _stft(self, waveform: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform."""
        # Simple STFT with Hann window
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
        """Inverse short-time Fourier transform."""
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

        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-8)
        waveform /= window_sum

        return waveform


class TTSEngine:
    """Complete text-to-speech pipeline.

    Text -> Phonemes -> Spectrogram -> Waveform
    """

    def __init__(self, vocab_size=256, embed_dim=128, hidden_dim=256,
                 n_mels=80, sample_rate=22050):
        self.decoder = SpectrogramDecoder(vocab_size, embed_dim, hidden_dim, n_mels)
        self.vocoder = GriffinLimVocoder(n_mels=n_mels, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.optimizer = SloAdam(lr=1e-3)

    def text_to_waveform(self, text: str, max_frames: int = 200) -> np.ndarray:
        """
        Convert text to speech waveform.

        Args:
            text: Input text string
            max_frames: Maximum spectrogram frames
        Returns:
            waveform: (num_samples,) audio waveform
        """
        # Simple character-level encoding (no phonemizer)
        phoneme_ids = np.array([[ord(c) % 256 for c in text]], dtype=np.int32)

        if phoneme_ids.shape[1] == 0:
            return np.zeros(self.sample_rate // 2, dtype=np.float32)

        # Generate mel spectrogram
        mel_spec = self.decoder.generate(phoneme_ids, max_frames)

        # Convert to waveform
        waveform = self.vocoder.generate_waveform(mel_spec)

        return waveform

    def parameters(self):
        return self.decoder.parameters()
