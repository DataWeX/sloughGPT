"""Tests for multimodal.engine — MultimodalEngine, TextDecoder, VisionEncoder."""

from __future__ import annotations

import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from domains.multimodal.engine import (
    MultimodalOutput,
    TextDecoder,
    MultimodalEngine,
)


# ── MultimodalOutput ────────────────────────────────────────────────────────


class TestMultimodalOutput:

    def test_init(self):
        out = MultimodalOutput(text="hello", confidence=0.9)
        assert out.text == "hello"
        assert out.confidence == 0.9


# ── TextDecoder ─────────────────────────────────────────────────────────────


class TestTextDecoder:

    def test_init(self):
        td = TextDecoder()
        assert td.embed_dim == 256
        assert td.hidden_dim == 512

    def test_init_custom(self):
        td = TextDecoder(embed_dim=128, hidden_dim=256)
        assert td.embed_dim == 128

    def test_build_vocab(self):
        td = TextDecoder()
        td.build_vocab(["hello world", "foo bar"])
        assert td.vocab_size > 0

    def test_encode_decode(self):
        td = TextDecoder()
        td.build_vocab(["hello"])
        encoded = td.encode("hello")
        assert isinstance(encoded, list)
        decoded = td.decode(encoded)
        assert isinstance(decoded, str)

    def test_vocab_size(self):
        td = TextDecoder()
        td.build_vocab(["abc"])
        assert td.vocab_size > 0


# ── MultimodalEngine ────────────────────────────────────────────────────────


class TestMultimodalEngine:

    def test_init(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        assert engine.vision is not None
        assert engine.audio is not None
        assert engine.text is not None
        assert engine.decoder is not None
        assert engine._trained is False

    def test_model_id(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        assert engine.model_id == "multimodal-v1"

    def test_embed_dim(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        assert engine.embed_dim == 64

    def test_metadata(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        meta = engine.metadata
        assert "vocab_size" in meta
        assert "trained" in meta
        assert meta["trained"] is False

    def test_train_eval(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.train(True)
        engine.eval()
        assert engine._trained is False

    def test_build_vocab(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello world", "test caption"])
        assert engine.text.vocab_size > 0

    def test_embed_untrained(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        result = engine.embed("hello")
        assert result == [0.0] * 128

    def test_embed_trained(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello"])
        engine._trained = True
        result = engine.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 128

    def test_extract_images_empty(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        result = engine._extract_images([{"content": "text only"}])
        assert result == []

    def test_extract_images_base64(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        msg = {"content": "data:image/png;base64,abc123"}
        result = engine._extract_images([msg])
        assert len(result) == 1

    def test_extract_images_list_content(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        msg = {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}
        result = engine._extract_images([msg])
        assert len(result) == 1

    def test_params_for_optimizer(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        params = engine._params_for_optimizer(engine.decoder.optimizer, None, None)
        assert len(params) > 0

    def test_params_for_optimizer_vision(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        params = engine._params_for_optimizer(engine.vision.optimizer, None, None)
        assert len(params) > 0

    def test_params_for_optimizer_audio(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        params = engine._params_for_optimizer(engine.audio.optimizer, None, None)
        assert len(params) > 0

    def test_params_for_optimizer_unknown(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        params = engine._params_for_optimizer(MagicMock(), None, None)
        assert params == []

    def test_param_groups(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        groups = engine.param_groups()
        assert "decoder" in groups
        assert "vision" in groups
        assert "audio" in groups

    def test_train_step_no_tokens(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        with pytest.raises(ValueError, match="text_tokens is required"):
            engine.train_step(images_np=np.random.randn(1, 224, 224, 3).astype(np.float32))

    def test_train_step(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello", "world"])
        images = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = np.array([[0, 1, 2]], dtype=np.int64)
        loss = engine.train_step(images_np=images, text_tokens=tokens)
        assert isinstance(loss, float)
        assert engine._trained is True

    def test_train_step_with_audio(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello"])
        images = np.random.randn(1, 224, 224, 3).astype(np.float32)
        audio = np.random.randn(1, 16000).astype(np.float32)
        tokens = np.array([[0, 1, 2]], dtype=np.int64)
        loss = engine.train_step(images_np=images, text_tokens=tokens, audio_np=audio)
        assert isinstance(loss, float)

    def test_train_batch_empty(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        result = engine.train_batch([])
        assert result == 0.0

    def test_generate(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello", "world"])
        images = np.random.randn(1, 224, 224, 3).astype(np.float32)
        output = engine.generate(image_np=images, max_len=5)
        assert isinstance(output, MultimodalOutput)
        assert isinstance(output.text, str)
        assert output.confidence >= 0.0

    def test_generate_no_image(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine.build_vocab(["hello"])
        audio = np.random.randn(1, 16000).astype(np.float32)
        output = engine.generate(audio_np=audio, max_len=5)
        assert isinstance(output, MultimodalOutput)

    def test_concat_modalities_no_input(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        with pytest.raises(ValueError, match="At least one"):
            engine._concat_modalities(None, None, None)

    def test_concat_modalities_image(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        images = np.random.randn(1, 224, 224, 3).astype(np.float32)
        embed, patches, opts = engine._concat_modalities(images_np=images)
        assert embed is not None
        assert patches is not None
        assert len(opts) > 0

    def test_clip_gradients(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        from domains.training.slonet import Tensor
        p = Tensor(np.ones((2, 2)), requires_grad=True)
        p.grad = Tensor(np.ones((2, 2)) * 10)
        engine._clip_gradients([p], max_norm=1.0)

    def test_zero_grad(self):
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        engine._zero_grad([engine.decoder.optimizer])
