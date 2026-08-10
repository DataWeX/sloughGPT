"""Tests for multimodal/text_encoder.py — TextEncoder encode_tokens, encode_text, parameters."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from domains.multimodal.text_encoder import TextEncoder


class TestTextEncoderInit:
    def test_default_init(self):
        enc = TextEncoder(vocab_size=100, embed_dim=32, n_heads=2, n_layers=2, max_seq_len=10)
        assert enc.vocab_size == 100
        assert enc.embed_dim == 32
        assert enc.max_seq_len == 10

    def test_has_tokenizer(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        assert enc.tokenizer is not None

    def test_has_optimizer(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        assert enc.optimizer is not None


class TestTextEncoderParameters:
    def test_parameters_returns_list(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=2, max_seq_len=8)
        params = enc.parameters()
        assert isinstance(params, list)
        assert len(params) > 0

    def test_parameters_count(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        params = enc.parameters()
        # token_embedding, pos_embedding, 1 block (2 ln attn + 2 ff + proj + ln_final) + norm + context_proj
        assert len(params) >= 5


class TestTextEncoderEncodeTokens:
    def test_encode_tokens_output_shape(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 5, 16)

    def test_encode_tokens_different_seq_lens(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        padded = np.zeros((2, 5), dtype=np.int32)
        padded[0, :3] = [1, 2, 3]
        padded[1, :5] = [4, 5, 6, 7, 8]
        result = enc.encode_tokens(padded)
        assert result.shape == (2, 5, 16)

    def test_encode_tokens_no_nan(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert not np.any(np.isnan(result.data))


class TestTextEncoderEncodeText:
    def test_encode_text_output_shape(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello world"])
        assert result.ndim == 3
        assert result.shape[0] == 1
        assert result.shape[2] == 32

    def test_encode_text_auto_trains_tokenizer(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        assert not enc.tokenizer._built
        enc.encode_text(["hello world"])
        assert enc.tokenizer._built

    def test_encode_text_multiple(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello", "world test"])
        assert result.shape[0] == 2


class TestTextEncoderTrainTokenizer:
    def test_train_tokenizer(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        assert not enc.tokenizer._built
        enc.train_tokenizer(["hello world", "foo bar"])
        assert enc.tokenizer._built
