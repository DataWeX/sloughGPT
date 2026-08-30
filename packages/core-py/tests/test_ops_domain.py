"""Tests for domains.ops — fused ops, activations, softmax, attention, embeddings."""

import math
import numpy as np
import pytest

from domains.ops import (
    FusedLayerNorm,
    FusedRMSNorm,
    FusedCrossEntropyLoss,
    FusedAttentionBias,
    ChunkedOperation,
    MemoryEfficientSoftmax,
    FusedScaleBias,
    OptimizedEmbedding,
    fused_swiglu,
    efficient_cross_entropy,
    chunked_matmul,
    ragged_to_padded,
    estimate_attention_memory,
    silu,
    gelu,
)


# ---------------------------------------------------------------------------
# FusedLayerNorm
# ---------------------------------------------------------------------------

class TestFusedLayerNorm:
    def test_output_shape(self):
        ln = FusedLayerNorm(8)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = ln(x)
        assert y.shape == x.shape

    def test_normalized_mean_near_zero(self):
        ln = FusedLayerNorm(16)
        x = np.random.randn(4, 16).astype(np.float32) * 10
        y = ln(x)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)

    def test_normalized_var_near_one(self):
        ln = FusedLayerNorm(16)
        x = np.random.randn(4, 16).astype(np.float32) * 10
        y = ln(x)
        assert np.allclose(y.var(axis=-1), 1.0, atol=1e-2)

    def test_weight_scales(self):
        ln = FusedLayerNorm(8)
        ln.weight = np.full(8, 3.0, dtype=np.float32)
        x = np.random.randn(2, 8).astype(np.float32)
        y = ln(x)
        assert np.allclose(y.var(axis=-1), 9.0, atol=0.5)

    def test_bias_shifts(self):
        ln = FusedLayerNorm(8, bias=True)
        ln.bias = np.full(8, 5.0, dtype=np.float32)
        x = np.random.randn(2, 8).astype(np.float32)
        y = ln(x)
        assert np.allclose(y.mean(axis=-1), 5.0, atol=1e-5)

    def test_no_bias(self):
        ln = FusedLayerNorm(8, bias=False)
        assert ln.bias is None
        x = np.random.randn(2, 8).astype(np.float32)
        y = ln(x)
        assert y.shape == x.shape

    def test_eps_prevents_nan(self):
        ln = FusedLayerNorm(4)
        x = np.zeros((2, 4), dtype=np.float32)
        y = ln(x)
        assert np.all(np.isfinite(y))

    def test_dtype_promotion(self):
        ln = FusedLayerNorm(8)
        x = np.random.randn(2, 8).astype(np.float64)
        y = ln(x)
        assert y.dtype == np.float32

    def test_callable(self):
        ln = FusedLayerNorm(8)
        x = np.random.randn(2, 8).astype(np.float32)
        y1 = ln(x)
        y2 = ln.forward(x)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# FusedRMSNorm
# ---------------------------------------------------------------------------

class TestFusedRMSNorm:
    def test_output_shape(self):
        norm = FusedRMSNorm(8)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = norm(x)
        assert y.shape == x.shape

    def test_weight_scales(self):
        norm = FusedRMSNorm(8)
        norm.weight = np.full(8, 2.0, dtype=np.float32)
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        y = norm(x)
        assert np.allclose(y, 2.0, atol=1e-5)

    def test_eps_prevents_nan(self):
        norm = FusedRMSNorm(4)
        x = np.zeros((2, 4), dtype=np.float32)
        y = norm(x)
        assert np.all(np.isfinite(y))

    def test_dtype_promotion(self):
        norm = FusedRMSNorm(8)
        x = np.random.randn(2, 8).astype(np.float64)
        y = norm(x)
        assert y.dtype == np.float32

    def test_3d_input(self):
        norm = FusedRMSNorm(8)
        x = np.random.randn(2, 3, 8).astype(np.float32)
        y = norm(x)
        assert y.shape == (2, 3, 8)

    def test_callable(self):
        norm = FusedRMSNorm(8)
        x = np.random.randn(2, 8).astype(np.float32)
        np.testing.assert_array_equal(norm(x), norm.forward(x))


# ---------------------------------------------------------------------------
# FusedCrossEntropyLoss
# ---------------------------------------------------------------------------

class TestFusedCrossEntropyLoss:
    def test_basic_loss_positive(self):
        ce = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        targets = np.array([2], dtype=np.int64)
        loss = ce(logits, targets)
        assert loss > 0

    def test_log_softmax_correct(self):
        ce = FusedCrossEntropyLoss()
        logits = np.array([[0.0, 0.0, 100.0]], dtype=np.float32)
        targets = np.array([2], dtype=np.int64)
        loss = ce(logits, targets)
        assert loss < 0.01

    def test_ignore_index(self):
        ce = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        targets = np.array([-100], dtype=np.int64)
        loss = ce(logits, targets)
        assert loss == 0.0

    def test_label_smoothing(self):
        ce_smooth = FusedCrossEntropyLoss(label_smoothing=0.1)
        ce_plain = FusedCrossEntropyLoss(label_smoothing=0.0)
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,))
        loss_smooth = ce_smooth(logits, targets)
        loss_plain = ce_plain(logits, targets)
        assert loss_smooth > 0
        assert loss_smooth != loss_plain

    def test_dtype_promotion(self):
        ce = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        targets = np.array([1], dtype=np.int32)
        loss = ce(logits, targets)
        assert isinstance(loss, float)

    def test_batch_dimension(self):
        ce = FusedCrossEntropyLoss()
        logits = np.random.randn(8, 20).astype(np.float32)
        targets = np.random.randint(0, 20, size=(8,))
        loss = ce(logits, targets)
        assert loss > 0

    def test_callable(self):
        ce = FusedCrossEntropyLoss()
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,))
        assert ce(logits, targets) == ce.forward(logits, targets)


# ---------------------------------------------------------------------------
# FusedAttentionBias
# ---------------------------------------------------------------------------

class TestFusedAttentionBias:
    def test_output_shapes(self):
        attn = FusedAttentionBias(num_heads=4)
        B, N, S, H, E = 2, 5, 7, 4, 8
        q = np.random.randn(B, N, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        out, weights = attn(q, k, v)
        assert out.shape == (B, N, H, E)
        assert weights.shape == (B, H, S, N)

    def test_attention_weights_sum_to_one(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 6, 2, 8).astype(np.float32)
        v = np.random.randn(1, 6, 2, 8).astype(np.float32)
        _, weights = attn(q, k, v)
        sums = weights.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_causal_mask(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        _, weights = attn(q, k, v, causal=True)
        # Upper triangular should be zero
        for b in range(1):
            for h in range(2):
                for i in range(4):
                    for j in range(i + 1, 4):
                        assert weights[b, h, i, j] < 1e-6

    def test_with_bias(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 3, 2, 4).astype(np.float32)
        k = np.random.randn(1, 3, 2, 4).astype(np.float32)
        v = np.random.randn(1, 3, 2, 4).astype(np.float32)
        bias = np.random.randn(1, 2, 3, 3).astype(np.float32)
        out, weights = attn(q, k, v, attn_bias=bias)
        assert out.shape == q.shape
        assert weights.sum(axis=-1).shape == (1, 2, 3)

    def test_scale(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 3, 2, 4).astype(np.float32)
        k = np.random.randn(1, 3, 2, 4).astype(np.float32)
        v = np.random.randn(1, 3, 2, 4).astype(np.float32)
        out1, _ = attn(q, k, v, scale=1.0)
        out2, _ = attn(q, k, v, scale=0.1)
        assert not np.allclose(out1, out2)

    def test_single_head_single_token(self):
        attn = FusedAttentionBias(num_heads=1)
        q = np.random.randn(1, 1, 1, 4).astype(np.float32)
        k = np.random.randn(1, 1, 1, 4).astype(np.float32)
        v = np.random.randn(1, 1, 1, 4).astype(np.float32)
        out, weights = attn(q, k, v)
        assert out.shape == (1, 1, 1, 4)
        assert weights.shape == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# ChunkedOperation
# ---------------------------------------------------------------------------

class TestChunkedOperation:
    def test_output_shapes(self):
        B, S, H, E = 2, 12, 4, 8
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        out, weights = chunked.attention_chunked(q, k, v)
        assert out.shape == (B, S, H, E)

    def test_weights_finite_and_nonneg(self):
        B, S, H, E = 1, 8, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        _, weights = chunked.attention_chunked(q, k, v)
        assert np.all(np.isfinite(weights))
        assert (weights >= 0).all()
        assert np.any(weights > 0)

    def test_output_finite_and_nonzero(self):
        B, S, H, E = 1, 8, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        out, _ = chunked.attention_chunked(q, k, v)
        assert np.all(np.isfinite(out))
        assert np.any(out != 0)

    def test_same_chunk_size_deterministic(self):
        B, S, H, E = 1, 16, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=8)
        out1, _ = chunked.attention_chunked(q, k, v, chunk_size=8)
        out2, _ = chunked.attention_chunked(q, k, v, chunk_size=8)
        np.testing.assert_array_equal(out1, out2)

    def test_small_input_no_chunking_needed(self):
        B, S, H, E = 1, 3, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        out, _ = chunked.attention_chunked(q, k, v)
        assert out.shape == (B, S, H, E)


# ---------------------------------------------------------------------------
# MemoryEfficientSoftmax
# ---------------------------------------------------------------------------

class TestMemoryEfficientSoftmax:
    def test_basic_softmax(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert np.allclose(result.sum(), 1.0, atol=1e-5)
        assert result.argmax() == 2

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert np.all(np.isfinite(result))
        assert np.allclose(result.sum(), 1.0, atol=1e-5)

    def test_unstable_overflow(self):
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=False)
        assert not np.all(np.isfinite(result))

    def test_2d_logits(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1)
        assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-5)

    def test_chunked_softmax(self):
        logits = np.random.randn(2, 64).astype(np.float32)
        result_normal = MemoryEfficientSoftmax.forward(logits, stable=True, chunk_size=0)
        result_chunked = MemoryEfficientSoftmax.forward(logits, stable=True, chunk_size=64)
        np.testing.assert_allclose(result_normal, result_chunked, atol=1e-4)

    def test_chunked_large_input(self):
        logits = np.random.randn(3, 128).astype(np.float32)
        result_normal = MemoryEfficientSoftmax.forward(logits, dim=-1, stable=True, chunk_size=0)
        result_chunked = MemoryEfficientSoftmax.forward(logits, dim=-1, stable=True, chunk_size=128)
        np.testing.assert_allclose(result_normal, result_chunked, atol=1e-4)

    def test_uniform_input(self):
        logits = np.ones(8, dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert np.allclose(result, 1.0 / 8, atol=1e-5)

    def test_single_element(self):
        logits = np.array([42.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert np.allclose(result, 1.0)


# ---------------------------------------------------------------------------
# FusedScaleBias
# ---------------------------------------------------------------------------

class TestFusedScaleBias:
    def test_identity(self):
        sb = FusedScaleBias(8)
        x = np.random.randn(2, 8).astype(np.float32)
        np.testing.assert_array_equal(sb(x), x)

    def test_weight_and_bias(self):
        sb = FusedScaleBias(4)
        sb.weight = np.full(4, 2.0, dtype=np.float32)
        sb.bias = np.full(4, 1.0, dtype=np.float32)
        x = np.ones((3, 4), dtype=np.float32)
        y = sb(x)
        assert np.allclose(y, 3.0)

    def test_callable(self):
        sb = FusedScaleBias(4)
        x = np.random.randn(2, 4).astype(np.float32)
        np.testing.assert_array_equal(sb(x), sb.forward(x))


# ---------------------------------------------------------------------------
# OptimizedEmbedding
# ---------------------------------------------------------------------------

class TestOptimizedEmbedding:
    def test_output_shape(self):
        emb = OptimizedEmbedding(num_embeddings=100, embedding_dim=16)
        idx = np.array([0, 1, 2], dtype=np.int64)
        out = emb(idx)
        assert out.shape == (3, 16)

    def test_2d_index(self):
        emb = OptimizedEmbedding(num_embeddings=50, embedding_dim=8)
        idx = np.array([[0, 1], [2, 3]], dtype=np.int64)
        out = emb(idx)
        assert out.shape == (2, 2, 8)

    def test_out_of_range_clipped(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4)
        idx = np.array([999], dtype=np.int64)
        out = emb(idx)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out[0], emb.weight[9])

    def test_negative_index_clipped_to_zero(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4)
        idx = np.array([-5], dtype=np.int64)
        out = emb(idx)
        assert out.shape == (1, 4)
        np.testing.assert_array_equal(out[0], emb.weight[0])

    def test_quantize_uint8(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4)
        emb.quantize_weight(dtype="uint8")
        assert emb._quantized.dtype == np.uint8
        assert emb._quantized.shape == (10, 4)
        assert emb._scale.shape == (10, 1)

    def test_quantize_int8(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4)
        emb.quantize_weight(dtype="int8")
        assert emb._quantized.dtype == np.int8
        assert emb._quantized.shape == (10, 4)

    def test_callable(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4)
        idx = np.array([0, 1], dtype=np.int64)
        np.testing.assert_array_equal(emb(idx), emb.forward(idx))

    def test_scale_grad_by_freq_stored(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4, scale_grad_by_freq=True)
        assert emb.scale_grad_by_freq is True

    def test_sparse_stored(self):
        emb = OptimizedEmbedding(num_embeddings=10, embedding_dim=4, sparse=True)
        assert emb.sparse is True


# ---------------------------------------------------------------------------
# fused_swiglu
# ---------------------------------------------------------------------------

class TestFusedSwiGLU:
    def test_output_shape(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w1 = np.random.randn(16, 8).astype(np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        w2 = np.random.randn(8, 16).astype(np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        w3 = np.random.randn(16, 8).astype(np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert out.shape == (2, 8)

    def test_zero_weights_gives_zero(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w1 = np.zeros((16, 8), dtype=np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        w2 = np.zeros((8, 16), dtype=np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        w3 = np.zeros((16, 8), dtype=np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert np.allclose(out, 0.0)

    def test_silu_activation_in_swiglu(self):
        x = np.ones((1, 4), dtype=np.float32)
        w1 = np.eye(8, 4, dtype=np.float32)
        b1 = np.zeros(8, dtype=np.float32)
        w2 = np.eye(4, 8, dtype=np.float32)
        b2 = np.zeros(4, dtype=np.float32)
        w3 = np.eye(8, 4, dtype=np.float32)
        b3 = np.zeros(8, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# efficient_cross_entropy
# ---------------------------------------------------------------------------

class TestEfficientCrossEntropy:
    def test_basic(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,))
        loss = efficient_cross_entropy(logits, targets)
        assert loss > 0

    def test_ignore_index(self):
        logits = np.random.randn(2, 10).astype(np.float32)
        targets = np.array([-100, -100], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets, ignore_index=-100)
        assert loss == 0.0

    def test_perfect_prediction(self):
        logits = np.zeros((1, 5), dtype=np.float32)
        logits[0, 3] = 100.0
        targets = np.array([3], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets)
        assert loss < 0.01

    def test_reduction_mean(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,))
        loss = efficient_cross_entropy(logits, targets, reduction="mean")
        assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# chunked_matmul
# ---------------------------------------------------------------------------

class TestChunkedMatmul:
    def test_small_matrix_direct(self):
        a = np.random.randn(100, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=512)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_large_matrix_chunked(self):
        a = np.random.randn(1024, 128).astype(np.float32)
        b = np.random.randn(128, 64).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=256)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_single_row(self):
        a = np.random.randn(1, 32).astype(np.float32)
        b = np.random.randn(32, 16).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=8)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_identity(self):
        a = np.random.randn(4, 4).astype(np.float32)
        b = np.eye(4, dtype=np.float32)
        result = chunked_matmul(a, b, chunk_size=2)
        np.testing.assert_allclose(result, a, atol=1e-5)


# ---------------------------------------------------------------------------
# ragged_to_padded
# ---------------------------------------------------------------------------

class TestRaggedToPadded:
    def test_basic(self):
        tokens = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        np.testing.assert_array_equal(padded, tokens)
        expected_mask = np.array([[True, True, True], [True, True, False]])
        np.testing.assert_array_equal(mask, expected_mask)

    def test_no_padding(self):
        tokens = np.array([[1, 2], [3, 4]], dtype=np.int64)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert mask.all()

    def test_all_padding(self):
        tokens = np.array([[0, 0], [0, 0]], dtype=np.int64)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert not mask.any()


# ---------------------------------------------------------------------------
# estimate_attention_memory
# ---------------------------------------------------------------------------

class TestEstimateAttentionMemory:
    def test_basic_estimate(self):
        mb = estimate_attention_memory(batch_size=1, seq_len=128, num_heads=8, head_dim=64)
        expected_bytes = 1 * 8 * 128 * 128 * 2
        assert mb == pytest.approx(expected_bytes / (1024 ** 2))

    def test_quadrupling_seq_len(self):
        mb1 = estimate_attention_memory(batch_size=1, seq_len=64, num_heads=4, head_dim=32)
        mb2 = estimate_attention_memory(batch_size=1, seq_len=128, num_heads=4, head_dim=32)
        assert mb2 == pytest.approx(mb1 * 4)

    def test_custom_precision(self):
        mb_fp32 = estimate_attention_memory(1, 64, 2, 16, precision_bytes=4)
        mb_fp16 = estimate_attention_memory(1, 64, 2, 16, precision_bytes=2)
        assert mb_fp32 == pytest.approx(mb_fp16 * 2)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------

class TestSilu:
    def test_zero(self):
        assert silu(np.array([0.0])) == pytest.approx(0.0)

    def test_large_positive(self):
        x = np.array([100.0])
        assert silu(x) == pytest.approx(100.0, abs=0.01)

    def test_large_negative(self):
        x = np.array([-100.0])
        assert silu(x) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        x = np.array([2.0])
        assert silu(x) == pytest.approx(x * (1 / (1 + np.exp(-x.item()))))

    def test_array(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = silu(x)
        assert result.shape == x.shape
        assert result[2] > result[1] > result[0]


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

class TestGelu:
    def test_zero(self):
        assert gelu(np.array([0.0])) == pytest.approx(0.0)

    def test_large_positive(self):
        x = np.array([100.0])
        assert gelu(x) == pytest.approx(100.0, abs=0.1)

    def test_large_negative(self):
        x = np.array([-100.0])
        assert gelu(x) == pytest.approx(0.0, abs=1e-5)

    def test_output_shape(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = gelu(x)
        assert result.shape == x.shape

    def test_threshold_behavior(self):
        x = np.array([-10.0, 10.0], dtype=np.float32)
        result = gelu(x)
        assert result[0] < 0.01
        assert result[1] > 9.0

    def test_zero_crossing(self):
        x = np.array([0.0], dtype=np.float32)
        result = gelu(x)
        assert np.allclose(result, 0.0, atol=1e-6)
