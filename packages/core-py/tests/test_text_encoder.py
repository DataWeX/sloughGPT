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

    def test_encode_text_returns_3d_array(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello"])
        assert result.ndim == 3

    def test_encode_text_single_token(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["a"])
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_encode_tokens_single_token(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[5]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 1, 16)

    def test_encode_tokens_zeros(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.zeros((1, 5), dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 5, 16)
        assert not np.any(np.isnan(result.data))

    def test_encode_tokens_all_same_ids(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.full((1, 4), 7, dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 4, 16)
        assert not np.any(np.isnan(result.data))

    def test_encode_text_single_word(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello"])
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_encode_text_long_string_truncated(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=5)
        result = enc.encode_text(["this is a very long sentence that should be truncated"])
        assert result.shape[1] <= 5

    def test_parameters_returns_all_param_types(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=2, max_seq_len=8)
        params = enc.parameters()
        # Should include token_embedding, pos_embedding, block params, norm params, context_proj
        assert len(params) >= 8

    def test_parameters_includes_token_embedding(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        params = enc.parameters()
        param_ids = [id(p) for p in params]
        assert id(enc.token_embedding) in param_ids or any(
            hasattr(p, 'data') and p.data.shape == enc.token_embedding.weight.data.shape for p in params
        )

    def test_parameters_includes_pos_embedding(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        params = enc.parameters()
        param_ids = [id(p) for p in params]
        assert id(enc.pos_embedding) in param_ids

    def test_encode_tokens_large_batch(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.random.randint(0, 50, (8, 10), dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (8, 10, 16)

    def test_encode_text_multiple_different_lengths(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=20)
        result = enc.encode_text(["hi", "hello world", "a longer sentence here"])
        assert result.shape[0] == 3
        assert result.shape[2] == 32

    def test_tokenizer_builds_on_first_encode_text(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        assert not enc.tokenizer._built
        enc.encode_text(["first call"])
        assert enc.tokenizer._built

    def test_tokenizer_reuses_after_build(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        enc.encode_text(["first"])
        enc.encode_text(["second"])
        assert enc.tokenizer._built

    def test_encode_text_no_inf_values(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello world"])
        assert not np.any(np.isinf(result))

    def test_encode_tokens_no_inf_values(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert not np.any(np.isinf(result.data))

    def test_constructor_stores_max_seq_len(self):
        enc = TextEncoder(vocab_size=100, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=20)
        assert enc.max_seq_len == 20

    def test_constructor_stores_embed_dim(self):
        enc = TextEncoder(vocab_size=100, embed_dim=64, n_heads=2, n_layers=1, max_seq_len=10)
        assert enc.embed_dim == 64

    def test_constructor_creates_blocks(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=3, max_seq_len=8)
        assert len(enc.blocks) == 3

    def test_constructor_single_block(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        assert len(enc.blocks) == 1

    def test_constructor_four_blocks(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=4, max_seq_len=8)
        assert len(enc.blocks) == 4

    def test_encode_text_preserves_batch_dimension(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        for n in [1, 2, 5]:
            texts = [f"word{i}" for i in range(n)]
            result = enc.encode_text(texts)
            assert result.shape[0] == n

    def test_encode_tokens_variable_length_within_batch(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.zeros((3, 10), dtype=np.int32)
        token_ids[0, :2] = [1, 2]
        token_ids[1, :5] = [3, 4, 5, 6, 7]
        token_ids[2, :10] = list(range(10))
        result = enc.encode_tokens(token_ids)
        assert result.shape == (3, 10, 16)

    def test_encode_text_special_characters(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello! @#$%"])
        assert result.ndim == 3

    def test_encode_text_numeric_string(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["12345"])
        assert result.ndim == 3

    def test_encode_text_unicode(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["日本語テスト"])
        assert result.ndim == 3


class TestTextEncoderStructure:
    def test_pos_embedding_shape(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        assert enc.pos_embedding.data.shape == (1, 10, 16)

    def test_pos_embedding_requires_grad(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        assert enc.pos_embedding.requires_grad is True

    def test_blocks_are_list(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=2, max_seq_len=8)
        assert isinstance(enc.blocks, list)

    def test_norm_exists(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        assert enc.norm is not None

    def test_context_proj_exists(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        assert enc.context_proj is not None

    def test_token_embedding_weight_shape(self):
        enc = TextEncoder(vocab_size=100, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        assert enc.token_embedding.weight.data.shape == (100, 32)

    def test_different_n_heads(self):
        enc = TextEncoder(vocab_size=50, embed_dim=32, n_heads=4, n_layers=1, max_seq_len=8)
        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 3, 32)

    def test_different_embed_dim(self):
        enc = TextEncoder(vocab_size=50, embed_dim=64, n_heads=2, n_layers=1, max_seq_len=8)
        token_ids = np.array([[1, 2]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape == (1, 2, 64)

    def test_parameters_count_scales_with_layers(self):
        enc1 = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=8)
        enc2 = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=3, max_seq_len=8)
        assert len(enc2.parameters()) > len(enc1.parameters())

    def test_max_seq_len_boundary(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=5)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert result.shape[1] == 5

    def test_encode_tokens_returns_tensor(self):
        from domains.training.slonet import Tensor
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert isinstance(result, Tensor)

    def test_encode_text_returns_numpy(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        result = enc.encode_text(["hello"])
        assert isinstance(result, np.ndarray)

    def test_encode_tokens_finite_values(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        result = enc.encode_tokens(token_ids)
        assert np.all(np.isfinite(result.data))

    def test_encode_text_empty_string_raises(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        with pytest.raises((ValueError, Exception)):
            enc.encode_text([""])

    def test_sequential_encode_text_calls(self):
        enc = TextEncoder(vocab_size=200, embed_dim=32, n_heads=2, n_layers=1, max_seq_len=10)
        r1 = enc.encode_text(["hello"])
        r2 = enc.encode_text(["hello"])
        assert r1.shape == r2.shape
        assert np.all(np.isfinite(r1))
        assert np.all(np.isfinite(r2))

    def test_encode_tokens_sequential_calls(self):
        enc = TextEncoder(vocab_size=50, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=10)
        ids1 = np.array([[1, 2]], dtype=np.int32)
        ids2 = np.array([[3, 4]], dtype=np.int32)
        r1 = enc.encode_tokens(ids1)
        r2 = enc.encode_tokens(ids2)
        assert r1.shape == r2.shape
