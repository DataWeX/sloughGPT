"""Tests for domains.multimodal.engine — pure-logic unit tests.

Covers: MultimodalOutput, TextDecoder, VisionEncoder, AudioEncoder,
SloTransformerDecoderBlock, SloTransformerDecoder, MultimodalEngine,
ReplayBuffer, augment_image, contrastive_loss, _causal_mask,
get_multimodal_engine.

No external API mocks. All tests use small dims for speed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.multimodal.engine import (
    MultimodalOutput,
    TextDecoder,
    VisionEncoder,
    AudioEncoder,
    SloTransformerDecoderBlock,
    SloTransformerDecoder,
    MultimodalEngine,
    ReplayBuffer,
    augment_image,
    contrastive_loss,
    _causal_mask,
    get_multimodal_engine,
)
from domains.training.slonet import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_engine(**kw) -> MultimodalEngine:
    defaults = dict(embed_dim=64, hidden_dim=128, n_vit_layers=1,
                    n_heads=2, n_decoder_layers=1, n_audio_layers=1)
    defaults.update(kw)
    return MultimodalEngine(**defaults)


def _rand_img(B=1) -> np.ndarray:
    return np.random.rand(B, 224, 224, 3).astype(np.float32)


def _rand_tokens(B=1, L=5) -> np.ndarray:
    return np.random.randint(2, 10, size=(B, L), dtype=np.int64)


# ---------------------------------------------------------------------------
# MultimodalOutput
# ---------------------------------------------------------------------------

class TestMultimodalOutput:
    def test_creation(self):
        o = MultimodalOutput(text="a cat", confidence=0.9)
        assert o.text == "a cat"
        assert o.confidence == 0.9

    def test_defaults_are_settable(self):
        o = MultimodalOutput(text="", confidence=0.0)
        assert o.text == ""
        assert o.confidence == 0.0

    def test_high_confidence(self):
        o = MultimodalOutput(text="test", confidence=1.0)
        assert o.confidence == 1.0

    def test_negative_confidence_allowed(self):
        o = MultimodalOutput(text="x", confidence=-0.5)
        assert o.confidence == -0.5


# ---------------------------------------------------------------------------
# TextDecoder
# ---------------------------------------------------------------------------

class TestTextDecoder:
    def test_build_vocab(self):
        td = TextDecoder()
        td.build_vocab(["hello world", "test data"])
        assert td.vocab_size > 0

    def test_encode_decode_roundtrip(self):
        td = TextDecoder()
        td.build_vocab(["hello"])
        ids = td.encode("hello")
        text = td.decode(ids)
        assert text == "hello"

    def test_vocab_size_includes_special_tokens(self):
        td = TextDecoder()
        td.build_vocab(["ab"])
        # At least BOS, EOS, PAD, UNK + 'a', 'b' + printable ASCII
        assert td.vocab_size >= 6

    def test_encode_empty_string(self):
        td = TextDecoder()
        td.build_vocab(["test"])
        ids = td.encode("")
        # Empty text => BOS + EOS only
        assert ids == [0, 1]

    def test_encode_unknown_char(self):
        td = TextDecoder()
        td.build_vocab(["a"])
        ids = td.encode("\x00")  # control char not in vocab
        # Should get UNK token (3)
        assert 3 in ids

    def test_decode_strips_special_tokens(self):
        td = TextDecoder()
        td.build_vocab(["ab"])
        ids = td.encode("ab")
        decoded = td.decode(ids)
        assert decoded == "ab"
        assert "<BOS>" not in decoded
        assert "<EOS>" not in decoded

    def test_embed_dim_stored(self):
        td = TextDecoder(embed_dim=128, hidden_dim=256)
        assert td.embed_dim == 128
        assert td.hidden_dim == 256


# ---------------------------------------------------------------------------
# VisionEncoder
# ---------------------------------------------------------------------------

class TestVisionEncoder:
    def test_extract_patches_shape(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        img = _rand_img(1)
        patches = ve.extract_patches(img)
        # (B, num_patches, patch_dim) where patch_dim = 3*32*32 = 3072
        assert patches.shape == (1, 49, 3072)

    def test_extract_patches_batch(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        img = _rand_img(2)
        patches = ve.extract_patches(img)
        assert patches.shape == (2, 49, 3072)

    def test_forward_shape(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        img = _rand_img(1)
        out = ve.forward(img)
        # (B, 1, embed_dim) — CLS token
        assert out.data.shape == (1, 1, 64)

    def test_get_patch_embeddings_shape(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        img = _rand_img(1)
        out = ve.get_patch_embeddings(img)
        # (B, num_patches+1, embed_dim)
        assert out.data.shape == (1, 50, 64)

    def test_forward_batch(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        img = _rand_img(3)
        out = ve.forward(img)
        assert out.data.shape == (3, 1, 64)

    def test_parameters_count(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=2)
        params = ve.parameters()
        # cls_token, pos_embed + patch_proj(2) + norm(2) + 2 blocks
        assert len(params) > 5

    def test_train_eval_mode(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        ve.train(False)
        ve.train(True)
        ve.eval()

    def test_image_size_constants(self):
        assert VisionEncoder.PATCH_SIZE == 32
        assert VisionEncoder.IMAGE_SIZE == 224
        assert VisionEncoder.IMAGE_SIZE % VisionEncoder.PATCH_SIZE == 0

    def test_patch_projection(self):
        ve = VisionEncoder(embed_dim=64, n_heads=2, n_layers=1)
        # patch_proj should map patch_dim -> embed_dim
        assert ve.patch_proj.weight.data.shape == (64, 3072)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class TestAudioEncoder:
    def test_mel_spectrogram_shape(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        mel = ae._mel_spectrogram(wf)
        assert mel.ndim == 2
        assert mel.shape[0] == 80  # N_MELS

    def test_mel_short_audio(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(100).astype(np.float32)
        mel = ae._mel_spectrogram(wf)
        assert mel.shape[0] == 80

    def test_mel_log_scale(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        mel = ae._mel_spectrogram(wf)
        # Log scale means all values should be finite
        assert np.all(np.isfinite(mel))

    def test_extract_patches_shape(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        patches = ae.extract_patches(wf)
        assert patches.ndim == 3
        assert patches.shape[0] == 1  # batch
        assert patches.shape[2] == ae.input_dim

    def test_extract_patches_1d_waveform(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        patches = ae.extract_patches(wf)
        # 1D should be reshaped to (1, T)
        assert patches.shape[0] == 1

    def test_extract_patches_batch(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(2, 16000).astype(np.float32)
        patches = ae.extract_patches(wf)
        assert patches.shape[0] == 2

    def test_forward_shape(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        out = ae.forward(wf)
        # (B, 1, embed_dim) — CLS token
        assert out.data.shape == (1, 1, 64)

    def test_get_patch_embeddings_shape(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        wf = np.random.randn(16000).astype(np.float32)
        out = ae.get_patch_embeddings(wf)
        assert out.data.shape[0] == 1
        assert out.data.shape[2] == 64

    def test_embed_patches(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        raw = ae.extract_patches(np.random.randn(16000).astype(np.float32))
        out = ae._embed_patches(raw)
        assert out.data.shape[2] == 64
        assert out.data.shape[0] == 1

    def test_constants(self):
        assert AudioEncoder.SAMPLE_RATE == 16000
        assert AudioEncoder.N_MELS == 80
        assert AudioEncoder.N_FFT == 512
        assert AudioEncoder.HOP_LENGTH == 160

    def test_parameters_count(self):
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        params = ae.parameters()
        assert len(params) > 5


# ---------------------------------------------------------------------------
# SloTransformerDecoderBlock
# ---------------------------------------------------------------------------

class TestSloTransformerDecoderBlock:
    def test_forward_with_context(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.0, name="test")
        x = Tensor(np.random.randn(1, 4, 64).astype(np.float32), requires_grad=True)
        ctx = Tensor(np.random.randn(1, 10, 64).astype(np.float32), requires_grad=True)
        out, cache = block.forward(x, ctx)
        assert out.data.shape == (1, 4, 64)
        assert cache is not None

    def test_forward_without_context(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.0, name="test")
        x = Tensor(np.random.randn(1, 4, 64).astype(np.float32), requires_grad=True)
        out, cache = block.forward(x)
        assert out.data.shape == (1, 4, 64)

    def test_forward_with_cache(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.0, name="test")
        x = Tensor(np.random.randn(1, 4, 64).astype(np.float32), requires_grad=True)
        _, cache = block.forward(x)
        # Incremental decoding: single new token
        x2 = Tensor(np.random.randn(1, 1, 64).astype(np.float32), requires_grad=True)
        out2, cache2 = block.forward(x2, kv_cache=cache, start_pos=4)
        assert out2.data.shape == (1, 1, 64)

    def test_parameters(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.0, name="test")
        params = block.parameters()
        assert len(params) > 10

    def test_train_mode(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.1, name="test")
        block.train(True)
        block.train(False)

    def test_residual_connection(self):
        block = SloTransformerDecoderBlock(64, 2, dropout=0.0, name="test")
        x = Tensor(np.ones((1, 2, 64), dtype=np.float32), requires_grad=True)
        out, _ = block.forward(x)
        # Output should be different from input (not identity)
        assert not np.allclose(out.data, x.data, atol=1e-5)

    def test_head_dim_divisible(self):
        block = SloTransformerDecoderBlock(64, 4, dropout=0.0, name="test")
        assert block.head_dim == 64 // 4


# ---------------------------------------------------------------------------
# SloTransformerDecoder
# ---------------------------------------------------------------------------

class TestSloTransformerDecoder:
    def test_forward_shape(self):
        dec = SloTransformerDecoder(vocab_size=100, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        img = Tensor(np.random.randn(1, 1, 64).astype(np.float32), requires_grad=False)
        tokens = Tensor(_rand_tokens(1, 4), requires_grad=False)
        patches = Tensor(np.random.randn(1, 10, 64).astype(np.float32), requires_grad=False)
        logits, last, caches = dec.forward(img, tokens, patches)
        # logits: (B, seq_len, vocab_size)
        assert logits.data.shape == (1, 4, 100)
        assert last.data.shape == (1, 1, 128)
        assert len(caches) == 1

    def test_forward_without_patches(self):
        dec = SloTransformerDecoder(vocab_size=100, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        img = Tensor(np.random.randn(1, 1, 64).astype(np.float32), requires_grad=False)
        tokens = Tensor(_rand_tokens(1, 4), requires_grad=False)
        logits, last, _ = dec.forward(img, tokens)
        assert logits.data.shape == (1, 4, 100)

    def test_forward_with_cache(self):
        dec = SloTransformerDecoder(vocab_size=100, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        img = Tensor(np.random.randn(1, 1, 64).astype(np.float32), requires_grad=False)
        tokens = Tensor(_rand_tokens(1, 3), requires_grad=False)
        _, _, caches = dec.forward(img, tokens)
        # Incremental: single new token
        new_tok = Tensor(np.array([[5]]), requires_grad=False)
        logits, _, new_caches = dec.forward(img, new_tok, kv_cache=caches, start_pos=3)
        assert logits.data.shape == (1, 1, 100)

    def test_1d_token_input(self):
        dec = SloTransformerDecoder(vocab_size=50, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        img = Tensor(np.random.randn(1, 1, 64).astype(np.float32), requires_grad=False)
        tokens_1d = Tensor(np.array([0, 3, 1]), requires_grad=False)
        logits, _, _ = dec.forward(img, tokens_1d)
        assert logits.data.shape[1] == 3

    def test_vocab_size_minimum(self):
        dec = SloTransformerDecoder(vocab_size=0, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        assert dec.vocab_size == 1

    def test_parameters_count(self):
        dec = SloTransformerDecoder(vocab_size=50, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=2)
        params = dec.parameters()
        assert len(params) > 10

    def test_train_eval(self):
        dec = SloTransformerDecoder(vocab_size=50, embed_dim=64, hidden_dim=128,
                                     n_heads=2, n_layers=1)
        dec.train(True)
        dec.eval()


# ---------------------------------------------------------------------------
# _causal_mask
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_shape(self):
        m = _causal_mask(4)
        assert m.data.shape == (1, 1, 4, 4)

    def test_upper_triangular(self):
        m = _causal_mask(4)
        # Diagonal and below should be 0 (no masking)
        for i in range(4):
            for j in range(i + 1):
                assert m.data[0, 0, i, j] == 0.0

    def test_upper_triangle_masked(self):
        m = _causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert m.data[0, 0, i, j] < -1e8

    def test_single_token(self):
        m = _causal_mask(1)
        assert m.data.shape == (1, 1, 1, 1)
        assert m.data[0, 0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# MultimodalEngine
# ---------------------------------------------------------------------------

class TestMultimodalEngine:
    def test_init(self):
        e = _small_engine()
        assert e.embed_dim == 64
        assert e._trained is False

    def test_build_vocab(self):
        e = _small_engine()
        e.build_vocab(["hello", "world"])
        assert e.text.vocab_size > 0

    def test_metadata(self):
        e = _small_engine()
        meta = e.metadata
        assert "vocab_size" in meta
        assert "trained" in meta
        assert "embed_dim" in meta
        assert meta["trained"] is False

    def test_model_id(self):
        e = _small_engine()
        assert e.model_id == "multimodal-v1"

    def test_embed_untrained(self):
        e = _small_engine()
        vec = e.embed("hello")
        assert len(vec) == 128
        assert all(v == 0.0 for v in vec)

    def test_embed_trained(self):
        e = _small_engine()
        e.build_vocab(["hello"])
        e._trained = True
        vec = e.embed("h")
        assert len(vec) == 128
        assert sum(vec) == 1.0  # one-hot

    def test_embed_empty_text(self):
        e = _small_engine()
        e.build_vocab(["test"])
        e._trained = True
        # encode("") returns [BOS, EOS] = [0, 1], not empty list,
        # so embed uses tokens[0] = 0 -> vec[0] = 1.0
        vec = e.embed("")
        assert len(vec) == 128
        assert sum(vec) == 1.0

    def test_train_eval(self):
        e = _small_engine()
        e.train()
        e.eval()

    def test_extract_images_image_url(self):
        e = _small_engine()
        msgs = [{'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}
        ]}]
        imgs = e._extract_images(msgs)
        assert len(imgs) == 1

    def test_extract_images_string_base64(self):
        e = _small_engine()
        msgs = [{'content': 'data:image/jpeg;base64,xyz123'}]
        imgs = e._extract_images(msgs)
        assert len(imgs) == 1

    def test_extract_images_multiple(self):
        e = _small_engine()
        msgs = [{'content': [
            {'type': 'image_url', 'image_url': {'url': 'a'}},
            {'type': 'image_url', 'image_url': {'url': 'b'}},
        ]}]
        imgs = e._extract_images(msgs)
        assert len(imgs) == 2

    def test_extract_images_empty(self):
        e = _small_engine()
        assert e._extract_images([]) == []
        assert e._extract_images([{'content': 'hello'}]) == []

    def test_forward(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        logits, embed = e.forward(img, tokens)
        assert logits.data.shape == (1, 4, e.text.vocab_size)
        assert embed.data.shape == (1, 1, 64)

    def test_forward_with_audio(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        audio = np.random.randn(1, 16000).astype(np.float32)
        logits, embed = e.forward(img, tokens, audio_np=audio)
        assert logits.data.shape[2] > 0

    def test_concat_modalities_no_inputs_raises(self):
        e = _small_engine()
        with pytest.raises(ValueError, match="At least one"):
            e._concat_modalities()

    def test_concat_modalities_image_only(self):
        e = _small_engine()
        img = _rand_img(1)
        embed, patches, opts = e._concat_modalities(images_np=img)
        assert embed.data.shape == (1, 1, 64)
        assert patches.data.shape[2] == 64

    def test_concat_modalities_audio_only(self):
        e = _small_engine()
        audio = np.random.randn(1, 16000).astype(np.float32)
        embed, patches, opts = e._concat_modalities(audio_np=audio)
        assert embed.data.shape[2] == 64

    def test_concat_modalities_both(self):
        e = _small_engine()
        img = _rand_img(1)
        audio = np.random.randn(1, 16000).astype(np.float32)
        embed, patches, opts = e._concat_modalities(images_np=img, audio_np=audio)
        assert embed.data.shape[2] == 64
        # patches should be concatenated along seq dim
        assert patches.data.shape[1] > 49

    def test_concat_modalities_precomputed_audio(self):
        e = _small_engine()
        img = _rand_img(1)
        raw = e.audio.extract_patches(np.random.randn(16000).astype(np.float32))
        embed, patches, opts = e._concat_modalities(images_np=img, audio_patches=raw)
        assert embed.data.shape[2] == 64

    def test_train_step(self):
        e = _small_engine()
        e.build_vocab(["hello"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        loss = e.train_step(images_np=img, text_tokens=tokens, lr=1e-3)
        assert isinstance(loss, float)
        assert loss > 0
        assert e._trained is True

    def test_train_step_no_text_raises(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        with pytest.raises(ValueError, match="text_tokens is required"):
            e.train_step(images_np=img)

    def test_train_step_with_audio(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        audio = np.random.randn(1, 16000).astype(np.float32)
        loss = e.train_step(images_np=img, text_tokens=tokens, audio_np=audio)
        assert loss > 0

    def test_train_step_with_precomputed_audio(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        raw = e.audio.extract_patches(np.random.randn(16000).astype(np.float32))
        loss = e.train_step(images_np=img, text_tokens=tokens, audio_patches=raw)
        assert loss > 0

    def test_train_step_sensitivity(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        loss, sens = e.train_step(images_np=img, text_tokens=tokens, compute_sens=True)
        assert isinstance(sens, dict)
        assert "decoder" in sens
        assert "vision" in sens

    def test_train_step_temperature(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        loss1 = e.train_step(images_np=img, text_tokens=tokens, temperature=1.0)
        loss2 = e.train_step(images_np=img, text_tokens=tokens, temperature=2.0)
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)

    def test_train_batch(self):
        e = _small_engine()
        e.build_vocab(["hello"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        samples = [(img, tokens, None, None), (img, tokens, None, None)]
        loss = e.train_batch(samples, lr=1e-3)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_batch_empty(self):
        e = _small_engine()
        assert e.train_batch([]) == 0.0

    def test_train_batch_all_none_tokens(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        samples = [(img, None, None, None)]
        loss = e.train_batch(samples)
        assert loss == 0.0

    def test_train_batch_with_audio(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        audio = np.random.randn(1, 16000).astype(np.float32)
        samples = [(img, tokens, audio, None)]
        loss = e.train_batch(samples)
        assert loss > 0

    def test_train_batch_sensitivity(self):
        e = _small_engine()
        e.build_vocab(["test"])
        img = _rand_img(1)
        tokens = _rand_tokens(1, 4)
        samples = [(img, tokens, None, None)]
        loss, sens = e.train_batch(samples, compute_sens=True)
        assert isinstance(sens, dict)

    def test_generate_greedy(self):
        e = _small_engine()
        e.build_vocab(["hello world"])
        img = _rand_img(1)
        out = e.generate(image_np=img, max_len=5, beam_width=1)
        assert isinstance(out, MultimodalOutput)
        assert isinstance(out.text, str)
        assert out.confidence >= 0.0

    def test_generate_beam(self):
        e = _small_engine()
        e.build_vocab(["hello world"])
        img = _rand_img(1)
        out = e.generate(image_np=img, max_len=5, beam_width=3)
        assert isinstance(out, MultimodalOutput)

    def test_generate_no_image_raises(self):
        e = _small_engine()
        e.build_vocab(["test"])
        with pytest.raises(ValueError, match="At least one"):
            e.generate(max_len=5)

    def test_generate_top_k(self):
        e = _small_engine()
        e.build_vocab(["hello world"])
        img = _rand_img(1)
        out = e.generate(image_np=img, max_len=5, top_k=3)
        assert isinstance(out, MultimodalOutput)

    def test_generate_temperature(self):
        e = _small_engine()
        e.build_vocab(["hello world"])
        img = _rand_img(1)
        out = e.generate(image_np=img, max_len=5, temperature=0.5)
        assert isinstance(out, MultimodalOutput)

    def test_generate_with_audio(self):
        e = _small_engine()
        e.build_vocab(["hello world"])
        img = _rand_img(1)
        audio = np.random.randn(1, 16000).astype(np.float32)
        out = e.generate(image_np=img, max_len=5, audio_np=audio)
        assert isinstance(out, MultimodalOutput)

    def test_save_load_roundtrip(self):
        e = _small_engine()
        e.build_vocab(["hello world", "test"])
        e._trained = True
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            e.save(path)
            assert os.path.exists(path)
            assert os.path.exists(path + ".json")
            e2 = MultimodalEngine.load(path)
            assert e2.embed_dim == e.embed_dim
            assert e2.text.vocab_size == e.text.vocab_size
            assert e2._trained is True
            # Weights match
            assert np.allclose(e.vision.cls_token.data, e2.vision.cls_token.data)
            assert np.allclose(e.decoder.parameters()[0].data,
                               e2.decoder.parameters()[0].data)

    def test_save_default_path(self):
        e = _small_engine()
        e.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            saved = e.save(path)
            assert saved.endswith("model.npz")
            assert os.path.exists(saved)

    def test_save_extra_meta(self):
        e = _small_engine()
        e.build_vocab(["test"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            e.save(path, extra_meta={"custom_key": 42})
            with open(path + ".json") as f:
                meta = json.load(f)
            assert meta["custom_key"] == 42

    def test_save_meta_content(self):
        e = _small_engine()
        e.build_vocab(["abc"])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.npz")
            e.save(path)
            with open(path + ".json") as f:
                meta = json.load(f)
            assert meta["embed_dim"] == 64
            assert meta["n_vit_layers"] == 1
            assert meta["n_heads"] == 2
            assert meta["n_decoder_layers"] == 1
            assert meta["n_audio_layers"] == 1
            assert isinstance(meta["char_vocab"], list)

    def test_params_for_optimizer(self):
        e = _small_engine()
        ps = e._params_for_optimizer(e.decoder.optimizer, None, None)
        assert len(ps) > 0
        ps_v = e._params_for_optimizer(e.vision.optimizer, None, None)
        assert len(ps_v) > 0
        ps_a = e._params_for_optimizer(e.audio.optimizer, None, None)
        assert len(ps_a) > 0
        # Unknown optimizer
        ps_x = e._params_for_optimizer("unknown", None, None)
        assert ps_x == []

    def test_param_groups(self):
        e = _small_engine()
        groups = e.param_groups()
        assert set(groups.keys()) == {"decoder", "vision", "audio"}

    def test_clip_gradients_no_clip(self):
        e = _small_engine()
        p = Tensor(np.array([1.0, 0.0]), requires_grad=True)
        p.grad = Tensor(np.array([0.5, 0.5]))
        e._clip_gradients([p], max_norm=10.0)
        # Norm is ~0.707, below 10.0, no clipping
        np.testing.assert_allclose(p.grad.data, np.array([0.5, 0.5]), atol=1e-6)

    def test_clip_gradients_clips(self):
        e = _small_engine()
        p = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        p.grad = Tensor(np.array([3.0, 4.0]))
        e._clip_gradients([p], max_norm=1.0)
        norm = np.sqrt(np.sum(p.grad.data ** 2))
        assert abs(norm - 1.0) < 1e-5

    def test_sum_grads(self):
        e = _small_engine()
        p = Tensor(np.array([2.0, 4.0]), requires_grad=True)
        p.grad = Tensor(np.array([1.0, 1.0]))
        e._sum_grads([p], 0.5)
        np.testing.assert_allclose(p.grad.data, np.array([0.5, 0.5]))

    def test_maybe_set_lr(self):
        e = _small_engine()
        opt = e.decoder.optimizer
        old_lr = opt.lr
        e._maybe_set_lr(0.99, [opt])
        assert opt.lr == 0.99
        e._maybe_set_lr(None, [opt])
        assert opt.lr == 0.99  # None does nothing

    def test_maybe_restore_lr(self):
        e = _small_engine()
        opt = e.decoder.optimizer
        old_lrs = {id(opt): 0.001}
        opt.lr = 0.99
        e._maybe_restore_lr(0.99, [opt], old_lrs)
        assert opt.lr == 0.001

    def test_precompute_audio_patches(self):
        e = _small_engine()
        wf = np.random.randn(1, 16000).astype(np.float32)
        raw = e.precompute_audio_patches(wf)
        assert raw.ndim == 3
        assert raw.shape[2] == e.audio.input_dim

    def test_capabilities(self):
        e = _small_engine()
        caps = e.capabilities
        assert caps.chat is True
        assert caps.vision is True
        assert caps.embedding is True


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(np.ones((1, 8, 8, 3)), "caption a")
        assert buf.size == 1

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        assert buf.size == 3
        assert buf.captions[-1] == "cap 4"

    def test_sample_all_when_fewer(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        imgs, caps = buf.sample(10)
        assert len(imgs) == 3
        assert len(caps) == 3

    def test_sample_n(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        imgs, caps = buf.sample(3)
        assert len(imgs) == 3
        assert len(caps) == 3

    def test_sample_preserves_content(self):
        buf = ReplayBuffer(capacity=10)
        img = np.ones((1, 4, 4, 3)) * 0.5
        buf.add(img, "my caption")
        imgs, caps = buf.sample(1)
        assert caps[0] == "my caption"
        np.testing.assert_array_equal(imgs[0], img)

    def test_diverse_sampling(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.add(np.ones((1, 4, 4, 3)), "same caption")
        imgs, caps = buf.sample(5)
        assert len(imgs) == 5

    def test_eviction_updates_counts(self):
        buf = ReplayBuffer(capacity=2)
        buf.add(np.ones((1, 4, 4, 3)), "a")
        buf.add(np.ones((1, 4, 4, 3)), "a")
        buf.add(np.ones((1, 4, 4, 3)), "b")
        assert buf.size == 2
        assert buf._counts.get("a", 0) == 1

    def test_eviction_to_zero(self):
        buf = ReplayBuffer(capacity=1)
        buf.add(np.ones((1, 4, 4, 3)), "x")
        buf.add(np.ones((1, 4, 4, 3)), "y")
        assert buf.size == 1
        assert buf._counts.get("x", 0) == 0

    def test_add_does_not_mutate_original(self):
        buf = ReplayBuffer(capacity=10)
        img = np.ones((1, 4, 4, 3))
        buf.add(img, "cap")
        img.fill(0)
        np.testing.assert_array_equal(buf.images[0], 1.0)

    def test_sample_zero(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(np.ones((1, 4, 4, 3)), "cap")
        imgs, caps = buf.sample(0)
        # sample(0) with size < n returns all (size=1 < 0 is False),
        # so it goes to weighted path and returns 0 items
        assert len(imgs) == 0

    def test_size_property(self):
        buf = ReplayBuffer(capacity=5)
        assert buf.size == 0
        buf.add(np.ones((1, 2, 2, 3)), "a")
        assert buf.size == 1


# ---------------------------------------------------------------------------
# augment_image
# ---------------------------------------------------------------------------

class TestAugmentImage:
    def test_preserves_shape(self):
        img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        result = augment_image(img)
        assert result.shape == img.shape

    def test_output_range(self):
        img = np.random.rand(1, 16, 16, 3).astype(np.float32) * 0.5 + 0.25
        result = augment_image(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_does_not_modify_original(self):
        img = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.5
        original = img.copy()
        augment_image(img)
        np.testing.assert_array_equal(img, original)

    def test_batch_format(self):
        img = np.random.rand(1, 64, 64, 3).astype(np.float32)
        result = augment_image(img)
        assert result.shape == (1, 64, 64, 3)

    def test_color_jitter_clipping(self):
        img = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.99
        result = augment_image(img)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_repeated_augmentation(self):
        img = np.random.rand(1, 16, 16, 3).astype(np.float32)
        for _ in range(10):
            result = augment_image(img)
            assert result.shape == img.shape


# ---------------------------------------------------------------------------
# contrastive_loss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_loss_value(self):
        z1 = Tensor(np.random.randn(1, 64).astype(np.float32), requires_grad=False)
        z2 = Tensor(np.random.randn(1, 64).astype(np.float32), requires_grad=False)
        neg = [Tensor(np.random.randn(1, 64).astype(np.float32), requires_grad=False)]
        loss = contrastive_loss(z1, z2, neg, temperature=0.5)
        assert loss.data > 0

    def test_loss_with_identical_views(self):
        z = Tensor(np.ones((1, 64), dtype=np.float32), requires_grad=False)
        neg = [Tensor(np.zeros((1, 64), dtype=np.float32), requires_grad=False)]
        loss = contrastive_loss(z, z, neg, temperature=0.5)
        # Identical views should have lower loss than random
        assert loss.data >= 0

    def test_loss_with_many_negatives(self):
        z1 = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
        z2 = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
        negs = [Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
                for _ in range(10)]
        loss = contrastive_loss(z1, z2, negs, temperature=0.5)
        assert loss.data > 0

    def test_loss_no_negatives(self):
        z1 = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
        z2 = Tensor(np.random.randn(1, 32).astype(np.float32), requires_grad=False)
        loss = contrastive_loss(z1, z2, [], temperature=0.5)
        # With no negatives, loss = -log(1) = 0
        assert abs(loss.data) < 1e-5


# ---------------------------------------------------------------------------
# get_multimodal_engine
# ---------------------------------------------------------------------------

class TestGetMultimodalEngine:
    def test_default(self):
        e = get_multimodal_engine()
        assert e.embed_dim == 256
        assert e._trained is False

    def test_custom_dims(self):
        e = get_multimodal_engine(embed_dim=128, hidden_dim=256)
        assert e.embed_dim == 128
