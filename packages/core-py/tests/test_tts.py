"""Tests for the multimodal TTS module (SpectrogramDecoder, GriffinLimVocoder, TTSEngine)."""

import numpy as np
import pytest

from domains.training.slonet import Tensor
from domains.multimodal.tts import (
    GriffinLimVocoder,
    SpectrogramDecoder,
    TTSEngine,
)


def make_decoder(**kw):
    defaults = dict(vocab_size=40, embed_dim=16, hidden_dim=24, n_mels=20, max_frames=8)
    defaults.update(kw)
    return SpectrogramDecoder(**defaults)


def force_stop(decoder, fire=True):
    decoder.fc_stop.weight.data[:] = 0.0
    decoder.fc_stop.bias.data[:] = 1000.0 if fire else -1000.0


# ---------------------------------------------------------------------------
# SpectrogramDecoder
# ---------------------------------------------------------------------------

class TestDecoderInit:
    def test_attributes(self):
        d = make_decoder()
        assert d.vocab_size == 40
        assert d.embed_dim == 16
        assert d.hidden_dim == 24
        assert d.n_mels == 20
        assert d.max_frames == 8
        for attr in [
            "embedding",
            "encoder_lstm",
            "attention_weights",
            "decoder_input_proj",
            "decoder_hidden_proj",
            "decoder_norm",
            "fc_mel",
            "fc_stop",
            "optimizer",
        ]:
            assert hasattr(d, attr), attr

    def test_output_dimensions(self):
        d = make_decoder()
        assert d.fc_mel.out_features == 20
        assert d.fc_stop.out_features == 1


class TestEncodeText:
    def test_returns_tensor_and_state(self):
        d = make_decoder()
        enc, h, c = d.encode_text(np.array([[1, 2, 3]], dtype=np.int32))
        assert isinstance(enc, Tensor)
        assert enc.data.shape == (1, 3, 24)
        assert h.shape == (1, 24)
        assert c.shape == (1, 24)

    def test_state_dtype(self):
        d = make_decoder()
        _, h, c = d.encode_text(np.array([[5]], dtype=np.int32))
        assert h.dtype == np.float32
        assert c.dtype == np.float32

    def test_empty_sequence(self):
        d = make_decoder()
        enc, h, c = d.encode_text(np.empty((1, 0), dtype=np.int32))
        assert enc.data.shape == (1, 0, 24)
        assert h.shape == (1, 24)
        assert c.shape == (1, 24)


class TestDecodeStep:
    def test_prev_mel_none(self):
        d = make_decoder()
        enc, h, c = d.encode_text(np.array([[1, 2]], dtype=np.int32))
        context = Tensor(enc.data.mean(axis=1, keepdims=True), requires_grad=True)
        mel, h2, c2, stop = d.decode_step(context, None, h, c)
        assert mel.data.shape == (1, 20)
        assert stop.data.shape == (1, 1)
        assert h2.shape == (1, 24)
        assert c2.shape == (1, 24)

    def test_prev_mel_2d(self):
        d = make_decoder()
        enc, h, c = d.encode_text(np.array([[1, 2]], dtype=np.int32))
        context = Tensor(enc.data.mean(axis=1, keepdims=True), requires_grad=True)
        mel, _, _, _ = d.decode_step(context, np.zeros((1, 20), dtype=np.float32), h, c)
        assert mel.data.shape == (1, 20)

    def test_prev_mel_3d(self):
        d = make_decoder()
        enc, h, c = d.encode_text(np.array([[1, 2]], dtype=np.int32))
        context = Tensor(enc.data.mean(axis=1, keepdims=True), requires_grad=True)
        mel, _, _, _ = d.decode_step(context, np.zeros((1, 1, 20), dtype=np.float32), h, c)
        assert mel.data.shape == (1, 20)


class TestGenerate:
    def test_returns_mel_spectrogram(self):
        d = make_decoder()
        mel = d.generate(np.array([[1, 2, 3]], dtype=np.int32), max_frames=6)
        assert mel.ndim == 2
        assert mel.shape[0] == 20
        assert 1 <= mel.shape[1] <= 6

    def test_stop_fires_early(self):
        d = make_decoder()
        force_stop(d, fire=True)
        mel = d.generate(np.array([[1, 2]], dtype=np.int32), max_frames=20)
        assert mel.shape[1] == 1

    def test_no_stop_generates_max_frames(self):
        d = make_decoder()
        force_stop(d, fire=False)
        mel = d.generate(np.array([[1, 2]], dtype=np.int32), max_frames=5)
        assert mel.shape[1] == 5

    def test_default_max_frames_from_config(self):
        d = make_decoder(max_frames=4)
        force_stop(d, fire=False)
        mel = d.generate(np.array([[1]], dtype=np.int32))
        assert mel.shape[1] == 4

    def test_negative_max_frames_returns_zero_mel(self):
        d = make_decoder()
        mel = d.generate(np.array([[1, 2]], dtype=np.int32), max_frames=-1)
        assert mel.shape == (20, 1)
        assert (mel == 0.0).all()


class TestParameters:
    def test_returns_requires_grad_params(self):
        d = make_decoder()
        params = d.parameters()
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# GriffinLimVocoder
# ---------------------------------------------------------------------------

class TestMelScale:
    def test_hz_to_mel_zero(self):
        v = GriffinLimVocoder()
        assert v._hz_to_mel(0.0) == 0.0

    def test_hz_to_mel_monotonic(self):
        v = GriffinLimVocoder()
        assert v._hz_to_mel(1000.0) < v._hz_to_mel(2000.0)

    def test_mel_to_hz_roundtrip(self):
        v = GriffinLimVocoder()
        for hz in [0.0, 100.0, 1000.0, 5000.0]:
            assert abs(v._mel_to_hz(v._hz_to_mel(hz)) - hz) < 1e-6

    def test_mel_basis_shape_and_range(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20)
        assert v.mel_basis.shape == (20, 65)
        assert v.mel_basis.min() >= 0.0
        assert v.mel_basis.max() <= 1.0

    def test_attributes(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20, sample_rate=8000)
        assert v.n_fft == 128
        assert v.hop_length == 32
        assert v.n_mels == 20
        assert v.sample_rate == 8000


class TestMelToLinear:
    def test_shape(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20)
        out = v._mel_to_linear(np.ones((20, 4)))
        assert out.shape == (65, 4)

    def test_non_negative(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20)
        out = v._mel_to_linear(np.ones((20, 4)))
        assert (out >= 1e-10).all()


class TestSTFT:
    def test_stft_shape(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32)
        wav = np.random.randn(512)
        spec = v._stft(wav)
        assert spec.shape == (65, (512 - 128) // 32 + 1)
        assert np.iscomplexobj(spec)

    def test_istft_reconstructs_length(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32)
        wav = v._istft(np.ones((65, 5), dtype=np.complex64))
        assert len(wav) == (5 - 1) * 32 + 128

    def test_generate_waveform(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20)
        wav = v.generate_waveform(np.ones((20, 4)), num_iterations=2)
        assert wav.ndim == 1
        assert len(wav) == (4 - 1) * 32 + 128
        assert np.isfinite(wav).all()

    def test_generate_waveform_single_iteration(self):
        v = GriffinLimVocoder(n_fft=128, hop_length=32, n_mels=20)
        wav = v.generate_waveform(np.ones((20, 4)), num_iterations=1)
        assert wav.ndim == 1
        assert len(wav) > 0


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

def make_engine(**kw):
    defaults = dict(vocab_size=40, embed_dim=16, hidden_dim=24, n_mels=20, sample_rate=8000)
    defaults.update(kw)
    return TTSEngine(**defaults)


class TestEngineInit:
    def test_attributes(self):
        eng = make_engine()
        assert eng.sample_rate == 8000
        assert isinstance(eng.decoder, SpectrogramDecoder)
        assert isinstance(eng.vocoder, GriffinLimVocoder)
        assert eng.vocoder.n_mels == 20
        assert eng.vocoder.sample_rate == 8000
        assert eng.optimizer is not None


class TestTextToWaveform:
    def test_returns_waveform(self):
        eng = make_engine()
        wav = eng.text_to_waveform("hi", max_frames=4)
        assert wav.ndim == 1
        assert len(wav) > 0
        assert np.isfinite(wav).all()

    def test_empty_text_returns_silence(self):
        eng = make_engine()
        wav = eng.text_to_waveform("", max_frames=4)
        assert wav.ndim == 1
        assert len(wav) == eng.sample_rate // 2
        assert (wav == 0).all()

    def test_parameters_delegates_to_decoder(self):
        eng = make_engine()
        assert len(eng.parameters()) == len(eng.decoder.parameters())
