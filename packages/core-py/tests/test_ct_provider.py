"""Tests for ct_provider — thin wrapper making NativeEngine look like SlonetChatProvider."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from domains.inference.ct_provider import CTransformProvider


class _FakeEngine:
    """Minimal mock of NativeEngine."""

    def __init__(self):
        self._config = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
        }

    def generate(self, messages, max_tokens=50, temperature=1.0, top_p=0.9, top_k=50, **kwargs):
        return "fake generated text"

    def _tokenize_simple(self, text):
        return list(range(len(text)))

    def _detokenize_simple(self, token_ids):
        return "".join(chr(c + 65) for c in token_ids[:5])


class TestCTransformProvider:
    def test_init(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine, model_id="test-model")
        assert provider._model_id == "test-model"
        assert provider._model is engine
        assert provider._engine is engine
        assert provider._device == "cpu"

    def test_generate(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine, model_id="test")
        result = provider.generate("Hello", max_tokens=10)
        assert result == "fake generated text"

    def test_generate_passes_params(self):
        engine = _FakeEngine()
        engine.generate = MagicMock(return_value="ok")
        provider = CTransformProvider(engine)
        provider.generate("Hi", max_tokens=20, temperature=0.5, top_p=0.8, top_k=30)
        engine.generate.assert_called_once_with(
            [{"role": "user", "content": "Hi"}],
            max_tokens=20,
            temperature=0.5,
            top_p=0.8,
            top_k=30,
        )

    def test_tokenize_uses_tokenizer_if_available(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        tokens = provider.tokenize("abc")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_fallback_when_no_tokenizer(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        provider._tokenizer = None
        tokens = provider.tokenize("abc")
        assert tokens == [0, 1, 2]

    def test_detokenize_uses_tokenizer_if_available(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        text = provider.detokenize([100, 200])
        assert isinstance(text, str)

    def test_detokenize_fallback_when_no_tokenizer(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        provider._tokenizer = None
        text = provider.detokenize([0, 1, 2])
        assert isinstance(text, str)

    def test_metadata(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine, model_id="my-model")
        meta = provider.metadata()
        assert meta["model_id"] == "my-model"
        assert meta["architecture"] == "NativeEngine"
        assert meta["n_layer"] == 2
        assert meta["n_embed"] == 64
        assert meta["n_head"] == 4
        assert meta["vocab_size"] == 1000
        assert meta["max_seq_len"] == 512
        assert meta["device"] == "cpu"
        assert meta["engine"] == "c"

    def test_metadata_no_config(self):
        engine = _FakeEngine()
        engine._config = None
        provider = CTransformProvider(engine)
        meta = provider.metadata()
        assert meta["n_layer"] == 0
        assert meta["vocab_size"] == 0

    def test_model_lock_none(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        assert provider._model_lock is None

    def test_quant_engine_none(self):
        engine = _FakeEngine()
        provider = CTransformProvider(engine)
        assert provider._quant_engine is None
