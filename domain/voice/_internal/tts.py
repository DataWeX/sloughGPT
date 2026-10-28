"""
Text-to-Speech (TTS) module.

Generates speech waveforms from text using:
1. Text encoder -> phoneme sequence
2. Spectrogram decoder (Tacotron2-style)
3. Griffin-Lim vocoder for waveform synthesis

All implemented in pure NumPy - no external dependencies.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger("slo.multimodal.tts")

from domain.voice._internal.phoneme_encoder import NUM_PHONEMES, PhonemeEncoder

# SSML tag patterns
_SSML_BREAK_PATTERN = re.compile(r'<break\s+time=["\'](\d+)(ms|s)["\']\s*/?>')
_SSMLProsody_PATTERN = re.compile(
    r'<prosody\s+rate=["\']([^"\']+)["\']\s+pitch=["\']([^"\']+)["\']\s*>(.*?)</prosody>', re.DOTALL
)
_SSML_EMPHASIS_PATTERN = re.compile(
    r'<emphasis\s+level=["\']([^"\']+)["\']\s*>(.*?)</emphasis>', re.DOTALL
)


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
    clean_text = re.sub(r"<[^>]+>", "", text).strip()

    return clean_text, events


def _get_slonet():
    """Lazy import of slonet primitives."""
    from domain.training._internal.slonet import (
        SloAdam,
        SloEmbedding,
        SloLayerNorm,
        SloLinear,
        SloLSTM,
        Tensor,
    )

    return Tensor, SloEmbedding, SloLSTM, SloLinear, SloLayerNorm, SloAdam


def _concat_last(a, b):
    """Differentiable concat along the last axis.

    slonet's public concatenate() is non-differentiable (requires_grad=False,
    no children), so training needs its own broadcast-correct splitter.
    """
    from domain.training._internal import slonet as _sn

    split = a.data.shape[-1]
    out = _sn.Tensor(
        np.concatenate([a.data, b.data], axis=-1),
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
    )

    def bk(g):
        ga, gb = np.split(np.asarray(g, dtype=np.float32), [split], axis=-1)
        for src, gg in ((a, ga), (b, gb)):
            if src.requires_grad:
                if src.grad is None:
                    src.grad = _sn.Tensor(gg, _copy=False)
                else:
                    src.grad.data += gg

    out._backward_fn = bk
    return out


class SpectrogramDecoder:
    """LSTM-based decoder that generates mel spectrograms from text.

    Architecture:
    - Text embedding -> LSTM -> Linear -> mel spectrogram frames
    """

    def __init__(
        self, vocab_size=256, embed_dim=128, hidden_dim=256, n_mels=80, max_frames=200, lr=1e-3
    ):
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

        self.optimizer = SloAdam(lr=lr)

    def train_step(
        self,
        phoneme_ids: np.ndarray,
        target_mel: np.ndarray,
        stop_targets: np.ndarray = None,
        update: bool = True,
    ) -> float:
        """One REAL gradient step: Tensor-graph forward -> backward -> Adam.

        Parity contract with the numpy inference path (encode_text/decode_step/
        generate): same weights, sequence-mean context, bias-free recurrent
        gates, bias-ful output heads, identical RMS math via _rmsnorm — so
        trained weights directly improve numpy generate(). Replaces the old
        _manual_backward, which subtracted random noise and never populated
        .grad.

        Stop loss is MSE on the sigmoid probability (slonet has no graph-wired
        log op for BCE); the old BCE ran on detached numpy anyway.
        """
        from domain.training._internal import slonet as _sn

        target_frames = target_mel.shape[1]
        if target_frames == 0:
            return 0.0
        hd = self.hidden_dim
        seq_len = phoneme_ids.shape[1] if phoneme_ids.ndim > 1 else phoneme_ids.shape[0]
        if seq_len == 0:
            return 0.0

        emb = self.encoder_lstm.embedding.forward(
            _sn.Tensor(phoneme_ids.astype(np.int64), requires_grad=False)
        )
        h = _sn.Tensor(np.zeros((1, hd), dtype=np.float32))
        c = _sn.Tensor(np.zeros((1, hd), dtype=np.float32))

        # Encoder — operation-for-operation mirror of encode_text() (layer-1
        # weights only, exactly like the numpy path; layer-2 stays unused).
        W_ih_T = self.encoder_lstm.W_ih.weight.T()
        W_hh_T = self.encoder_lstm.W_hh.weight.T()
        all_igates = _sn._matmul(emb, W_ih_T)
        enc_sum = None
        for t in range(seq_len):
            igates = _sn._slice(all_igates, (slice(None), t, slice(None)))
            gates = _sn._add(igates, _sn._matmul(h, W_hh_T))
            gi = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(0, hd))))
            gf = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(hd, 2 * hd))))
            gg = _sn.tanh(_sn._slice(gates, (slice(None), slice(2 * hd, 3 * hd))))
            go = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(3 * hd, 4 * hd))))
            c = _sn._add(_sn._mul(gf, c), _sn._mul(gi, gg))
            h_raw = _sn._mul(go, _sn.tanh(c))
            h = _sn._rmsnorm(h_raw, self.encoder_lstm.hidden_norm.weight, 1e-5)
            enc_sum = h if enc_sum is None else _sn._add(enc_sum, h)
        # Sequence mean without a stack op: mean(h_t) == sum(h_t)/S — matches
        # the numpy context (enc_out.data.mean(axis=1)) element-for-element.
        context = enc_sum * (1.0 / float(seq_len))

        # Decoder rollout over target frames — mirror of decode_step().
        prev_mel = _sn.Tensor(np.zeros((1, self.n_mels), dtype=np.float32))
        dec_in_T = self.decoder_input_proj.weight.T()
        dec_hid_T = self.decoder_hidden_proj.weight.T()
        mel_acc = None
        stop_acc = None
        for t in range(target_frames):
            dec_input = _concat_last(context, prev_mel)
            gates = _sn._add(
                _sn._matmul(dec_input, dec_in_T), _sn._matmul(h, dec_hid_T)
            )
            gi = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(0, hd))))
            gf = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(hd, 2 * hd))))
            gg = _sn.tanh(_sn._slice(gates, (slice(None), slice(2 * hd, 3 * hd))))
            go = _sn.sigmoid(_sn._slice(gates, (slice(None), slice(3 * hd, 4 * hd))))
            c = _sn._add(_sn._mul(gf, c), _sn._mul(gi, gg))
            h_raw = _sn._mul(go, _sn.tanh(c))
            h = _sn._rmsnorm(h_raw, self.decoder_norm.weight, 1e-5)
            mel_pred = self.fc_mel.forward(h)
            diff = _sn._sub(
                mel_pred,
                _sn.Tensor(target_mel[:, t : t + 1].T.copy(), requires_grad=False),
            )
            frame_mse = _sn._mean(_sn._mul(diff, diff))
            mel_acc = frame_mse if mel_acc is None else _sn._add(mel_acc, frame_mse)
            if stop_targets is not None:
                p = _sn.sigmoid(self.fc_stop.forward(h))
                st = _sn.Tensor(
                    np.array([[stop_targets[0, t]]], dtype=np.float32),
                    requires_grad=False,
                )
                sd = _sn._sub(p, st)
                stop_mse = _sn._mean(_sn._mul(sd, sd))
                stop_acc = stop_mse if stop_acc is None else _sn._add(stop_acc, stop_mse)
            prev_mel = mel_pred

        total = mel_acc * (1.0 / float(target_frames))
        if stop_acc is not None:
            total = total + stop_acc * (0.5 / float(target_frames))
        if update:
            total.backward()
            self.optimizer.step([p for p in self.parameters() if p.grad is not None])
        return float(np.asarray(total.data).real)

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
            xt = emb[:, t : t + 1, :]
            igates = xt @ self.encoder_lstm.W_ih.weight.data.T
            hgates = h @ self.encoder_lstm.W_hh.weight.data.T
            gates = igates + hgates
            g = gates[0, 0] if gates.ndim > 2 else gates[0]
            gi = 1.0 / (1.0 + np.exp(np.clip(-g[:hd], -500.0, 500.0)))
            gf = 1.0 / (1.0 + np.exp(np.clip(-g[hd : 2 * hd], -500.0, 500.0)))
            gg = np.tanh(g[2 * hd : 3 * hd])
            go = 1.0 / (1.0 + np.exp(np.clip(-g[3 * hd :], -500.0, 500.0)))
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
            context = Tensor(
                enc_out.data.mean(axis=1, keepdims=True), requires_grad=True, _children=(enc_out,)
            )
            mel_pred, h, c, stop_pred = self.decode_step(context, prev_mel, h, c)
            mel_frames.append(mel_pred.data[0])
            if stop_pred.data[0, 0] > 0.5:
                break
            prev_mel = mel_pred.data

        if not mel_frames:
            return np.zeros((self.n_mels, 1), dtype=np.float32)
        return np.stack(mel_frames, axis=-1)

    def generate_streaming(
        self, phoneme_ids: np.ndarray, max_frames: int = None, chunk_size: int = 8
    ):
        Tensor, _, _, _, _, _ = _get_slonet()
        max_frames = max_frames or self.max_frames
        if phoneme_ids.shape[1] == 0:
            return
        enc_out, h, c = self.encode_text(phoneme_ids)
        mel_frames = []
        prev_mel = None

        for _ in range(max_frames):
            context = Tensor(
                enc_out.data.mean(axis=1, keepdims=True), requires_grad=True, _children=(enc_out,)
            )
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

    def decode_step(self, context, prev_mel: np.ndarray, h: np.ndarray, c: np.ndarray):
        Tensor, _, _, _, _, _ = _get_slonet()
        if prev_mel is not None:
            if prev_mel.ndim == 2:
                prev_mel = prev_mel[:, np.newaxis, :]
            dec_input = np.concatenate([context.data, prev_mel], axis=-1)
        else:
            dec_input = np.concatenate(
                [context.data, np.zeros((1, 1, self.n_mels), dtype=np.float32)], axis=-1
            )

        hd = self.hidden_dim
        igates = dec_input @ self.decoder_input_proj.weight.data.T
        hgates = h @ self.decoder_hidden_proj.weight.data.T
        gates = igates + hgates
        g = gates[0, 0] if gates.ndim > 2 else (gates[0] if gates.ndim > 1 else gates)
        gi = 1.0 / (1.0 + np.exp(np.clip(-g[:hd], -500.0, 500.0)))
        gf = 1.0 / (1.0 + np.exp(np.clip(-g[hd : 2 * hd], -500.0, 500.0)))
        gg = np.tanh(g[2 * hd : 3 * hd])
        go = 1.0 / (1.0 + np.exp(np.clip(-g[3 * hd :], -500.0, 500.0)))
        c = gf * c + gi * gg
        h_raw = go * np.tanh(c)
        rms = np.sqrt(np.mean(h_raw**2, axis=-1, keepdims=True) + 1e-5)
        h = (h_raw / rms) * self.decoder_norm.weight.data

        mel_pred = self.fc_mel.forward(Tensor(h, requires_grad=False))
        stop_pred = self.fc_stop.forward(Tensor(h, requires_grad=False))
        return mel_pred, h, c, stop_pred

    def parameters(self):
        """Forward-relevant trainable weights (deterministic order).

        Exactly the tensors the numpy inference path reads (encode_text /
        decode_step). Excluded as dead weight — none can ever receive a
        gradient: the decoder's unused self.embedding, encoder layer-2
        (numpy runs layer-1 only), the attention head (never called), the
        LM head fc_out (SloLSTM.parameters() would add it), and biases the
        bias-free gates / RMS norms never read (numpy parity). hidden_norm
        stays: without it the encoder norm never trains.
        """
        lstm = self.encoder_lstm
        params = [
            lstm.embedding.weight,
            lstm.W_ih.weight,
            lstm.W_hh.weight,
            lstm.hidden_norm.weight,
            self.decoder_input_proj.weight,
            self.decoder_hidden_proj.weight,
            self.decoder_norm.weight,
        ]
        params += self.fc_mel.parameters()
        params += self.fc_stop.parameters()
        return [p for p in params if p.requires_grad]

    def named_parameters(self) -> list[tuple[str, object]]:
        """Checkpoint keys — the SAME live set as parameters(), same order.

        save() writes exactly what the optimizer trains; load() ignores
        unknown extra keys (older checkpoints stay loadable) and rejects
        missing ones (a real shape change must not silently half-load).
        """
        out: list[tuple[str, object]] = [
            ("enc.embedding.weight", self.encoder_lstm.embedding.weight),
            ("enc.W_ih.weight", self.encoder_lstm.W_ih.weight),
            ("enc.W_hh.weight", self.encoder_lstm.W_hh.weight),
            ("enc.hidden_norm.weight", self.encoder_lstm.hidden_norm.weight),
            ("dec_in.weight", self.decoder_input_proj.weight),
            ("dec_hid.weight", self.decoder_hidden_proj.weight),
            ("dec_norm.weight", self.decoder_norm.weight),
        ]
        for prefix, lin in (("fc_mel", self.fc_mel), ("fc_stop", self.fc_stop)):
            out.append((f"{prefix}.weight", lin.weight))
            if lin.use_bias:
                out.append((f"{prefix}.bias", lin.bias))
        return out

    def save(self, path: str) -> str:
        """Persist all weights to an .npz checkpoint."""
        target = path if path.endswith(".npz") else f"{path}.npz"
        np.savez(target, **{name: np.asarray(p.data) for name, p in self.named_parameters()})
        return target

    def load(self, path: str) -> None:
        """Load weights from an .npz checkpoint written by save()."""
        with np.load(path) as z:
            names = {name for name, _ in self.named_parameters()}
            missing = names - set(z.files)
            if missing:
                raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
            for name, p in self.named_parameters():
                arr = z[name]
                if arr.shape != p.data.shape:
                    raise ValueError(
                        f"shape mismatch for {name}: ckpt {arr.shape} vs model {p.data.shape}"
                    )
                p.data[...] = arr


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

    def generate_waveform(
        self, mel_spectrogram: np.ndarray, num_iterations: int = 32
    ) -> np.ndarray:
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
            frame = waveform[start : start + self.n_fft] * window
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
            waveform[start : start + self.n_fft] += frame * window
            window_sum[start : start + self.n_fft] += window**2
        window_sum = np.maximum(window_sum, 1e-8)
        waveform /= window_sum
        return waveform


class TTSEngine:
    """Complete text-to-speech pipeline."""

    def __init__(
        self,
        vocab_size=NUM_PHONEMES,
        embed_dim=128,
        hidden_dim=256,
        n_mels=80,
        sample_rate=22050,
        lr=1e-3,
    ):
        _, _, _, _, _, SloAdam = _get_slonet()
        self.decoder = SpectrogramDecoder(vocab_size, embed_dim, hidden_dim, n_mels, lr=lr)
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

    def text_to_waveform(
        self, text: str, max_frames: int = 200, speed: float = 1.0, pitch_shift: float = 0.0
    ) -> np.ndarray:
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

        return scipy.ndimage.zoom(mel_spec, (1, 1 / speed), order=1)

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

    def train_step(
        self, text: str, target_waveform: np.ndarray, max_frames: int = 200, update: bool = True
    ) -> float:
        if not text or not text.strip():
            return 0.0
        phoneme_ids = self._phoneme_encoder.encode(text)
        if phoneme_ids.shape[1] == 0:
            return 0.0
        # Magnitude spectrogram: _stft returns complex64, and training against
        # its real part (the old behavior) is unsupervised noise — mel filters
        # act on |spec|. Inference never reads this path (griffin-lim takes the
        # decoder's mel directly), so this changes training targets only.
        spec = np.abs(self.vocoder._stft(target_waveform))
        target_mel = self.vocoder.mel_basis @ spec
        if target_mel.ndim == 1:
            target_mel = target_mel.reshape(self.decoder.n_mels, -1)
        target_mel = np.ascontiguousarray(target_mel[:, :max_frames], dtype=np.float32)
        stop_targets = np.zeros((1, target_mel.shape[1]), dtype=np.float32)
        stop_targets[0, -1] = 1.0
        return self.decoder.train_step(phoneme_ids, target_mel, stop_targets, update=update)

    def train_epoch(
        self, training_data: list[tuple[str, np.ndarray]], max_frames: int = 200
    ) -> float:
        total_loss = 0.0
        for text, waveform in training_data:
            loss = self.train_step(text, waveform, max_frames)
            total_loss += loss
        return total_loss / max(len(training_data), 1)

    def eval_loss(self, pairs: list[tuple[str, np.ndarray]], max_frames: int = 200) -> float:
        """Mean forward loss over `pairs` WITHOUT optimizer updates (see update=).

        The held-out/eval twin of train_epoch — same forward, no backward,
        no step, no grad clearing — so a benchmark can watch generalization
        without training on what it measures.
        """
        if not pairs:
            return 0.0
        total = 0.0
        for text, waveform in pairs:
            total += self.train_step(text, waveform, max_frames, update=False)
        return total / len(pairs)

    def parameters(self):
        return self.decoder.parameters()

    def save(self, path: str) -> str:
        """Persist decoder weights to an .npz checkpoint (returns final path)."""
        return self.decoder.save(path)

    def load(self, path: str) -> None:
        """Load decoder weights from an .npz checkpoint written by save()."""
        self.decoder.load(path)


__all__ = [
    "parse_ssml",
    "SpectrogramDecoder",
    "GriffinLimVocoder",
    "TTSEngine",
]
