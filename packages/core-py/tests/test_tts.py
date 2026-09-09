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


# ---------------------------------------------------------------------------
# SSML Parsing
# ---------------------------------------------------------------------------

class TestSSMLParsing:
    def test_parse_simple_text(self):
        from domains.multimodal.tts import parse_ssml
        text, events = parse_ssml("hello world")
        assert text == "hello world"
        assert events == []

    def test_parse_break_tag(self):
        from domains.multimodal.tts import parse_ssml
        text, events = parse_ssml("hello<break time='500ms'/>world")
        assert text == "helloworld"
        assert len(events) == 1
        assert events[0]["type"] == "break"
        assert events[0]["duration_ms"] == 500

    def test_parse_break_tag_seconds(self):
        from domains.multimodal.tts import parse_ssml
        text, events = parse_ssml("hello<break time='1s'/>world")
        assert text == "helloworld"
        assert events[0]["duration_ms"] == 1000

    def test_parse_prosody_tag(self):
        from domains.multimodal.tts import parse_ssml
        text, events = parse_ssml("<prosody rate='slow' pitch='low'>hello</prosody>")
        assert text == "hello"
        assert len(events) == 1
        assert events[0]["type"] == "prosody"
        assert events[0]["rate"] == "slow"
        assert events[0]["pitch"] == "low"

    def test_parse_emphasis_tag(self):
        from domains.multimodal.tts import parse_ssml
        text, events = parse_ssml("<emphasis level='strong'>hello</emphasis>")
        assert text == "hello"
        assert len(events) == 1
        assert events[0]["type"] == "emphasis"
        assert events[0]["level"] == "strong"

    def test_parse_multiple_tags(self):
        from domains.multimodal.tts import parse_ssml
        ssml = "hello<break time='200ms'/><emphasis level='strong'>world</emphasis>"
        text, events = parse_ssml(ssml)
        assert text == "helloworld"
        assert len(events) == 2


# ---------------------------------------------------------------------------
# SSML to Waveform
# ---------------------------------------------------------------------------

class TestSSMLToWaveform:
    def test_returns_waveform(self):
        eng = make_engine()
        wav = eng.ssml_to_waveform("hello", max_frames=4)
        assert wav.ndim == 1
        assert len(wav) > 0
        assert np.isfinite(wav).all()

    def test_empty_ssml_returns_silence(self):
        eng = make_engine()
        wav = eng.ssml_to_waveform("", max_frames=4)
        assert wav.ndim == 1
        assert len(wav) == eng.sample_rate // 2
        assert (wav == 0).all()

    def test_ssml_with_break(self):
        eng = make_engine()
        wav = eng.ssml_to_waveform("hello<break time='500ms'/>world", max_frames=8)
        assert wav.ndim == 1
        assert len(wav) > 0

    def test_ssml_with_prosody(self):
        eng = make_engine()
        wav = eng.ssml_to_waveform("<prosody rate='slow' pitch='low'>hello</prosody>", max_frames=4)
        assert wav.ndim == 1
        assert len(wav) > 0


# ---------------------------------------------------------------------------
# Streaming Generation
# ---------------------------------------------------------------------------

class TestStreamingGeneration:
    def test_yields_chunks(self):
        d = make_decoder()
        chunks = list(d.generate_streaming(np.array([[1, 2, 3]], dtype=np.int32), max_frames=6, chunk_size=3))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.ndim == 2
            assert chunk.shape[0] == 20  # n_mels

    def test_chunk_size(self):
        d = make_decoder()
        chunks = list(d.generate_streaming(np.array([[1, 2]], dtype=np.int32), max_frames=8, chunk_size=4))
        assert len(chunks) > 0
        # First chunks should have chunk_size frames
        for chunk in chunks[:-1]:
            assert chunk.shape[1] == 4

    def test_stop_early(self):
        d = make_decoder()
        force_stop(d, fire=True)
        chunks = list(d.generate_streaming(np.array([[1, 2]], dtype=np.int32), max_frames=20, chunk_size=4))
        # Should stop early due to stop token
        total_frames = sum(c.shape[1] for c in chunks)
        assert total_frames < 20

    def test_empty_sequence(self):
        d = make_decoder()
        chunks = list(d.generate_streaming(np.empty((1, 0), dtype=np.int32), max_frames=4, chunk_size=2))
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestDecoderTrainStep:
    def test_returns_loss(self):
        d = make_decoder()
        phoneme_ids = np.array([[1, 2, 3]], dtype=np.int32)
        target_mel = np.random.randn(20, 4).astype(np.float32)
        loss = d.train_step(phoneme_ids, target_mel)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_loss_decreases(self):
        d = make_decoder()
        phoneme_ids = np.array([[1, 2, 3]], dtype=np.int32)
        target_mel = np.random.randn(20, 4).astype(np.float32)

        losses = []
        for _ in range(5):
            loss = d.train_step(phoneme_ids, target_mel)
            losses.append(loss)

        # Loss should generally decrease or stay stable
        assert losses[-1] <= losses[0] + 0.1

    def test_with_stop_targets(self):
        d = make_decoder()
        phoneme_ids = np.array([[1, 2, 3]], dtype=np.int32)
        target_mel = np.random.randn(20, 4).astype(np.float32)
        stop_targets = np.zeros((1, 4), dtype=np.float32)
        stop_targets[0, -1] = 1.0
        loss = d.train_step(phoneme_ids, target_mel, stop_targets)
        assert isinstance(loss, float)


class TestEngineTrainStep:
    def test_returns_loss(self):
        eng = make_engine()
        wav = np.random.randn(8000).astype(np.float32)
        loss = eng.train_step("hello", wav)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_empty_text_returns_zero(self):
        eng = make_engine()
        wav = np.random.randn(8000).astype(np.float32)
        loss = eng.train_step("", wav)
        assert loss == 0.0


class TestEngineTrainEpoch:
    def test_returns_avg_loss(self):
        eng = make_engine()
        training_data = [
            ("hello", np.random.randn(8000).astype(np.float32)),
            ("world", np.random.randn(8000).astype(np.float32)),
        ]
        loss = eng.train_epoch(training_data)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_empty_data_returns_zero(self):
        eng = make_engine()
        loss = eng.train_epoch([])
        assert loss == 0.0
