"""Tests for domains.ops — pure NumPy operations (no external dependencies)."""

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


# =============================================================================
# FusedLayerNorm
# =============================================================================

class TestFusedLayerNorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        ln = FusedLayerNorm(8)
        y = ln(x)
        assert y.shape == x.shape

    def test_normalized_mean_near_zero(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        ln = FusedLayerNorm(16)
        y = ln(x)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5)

    def test_normalized_var_near_one(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        ln = FusedLayerNorm(16)
        y = ln(x)
        vars_ = y.var(axis=-1)
        assert np.allclose(vars_, 1.0, atol=0.05)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32)
        ln = FusedLayerNorm(8)
        ln.weight = np.full(8, 2.0, dtype=np.float32)
        y = ln(x)
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_bias_shifts(self):
        x = np.ones((2, 8), dtype=np.float32) * 5.0
        ln = FusedLayerNorm(8)
        ln.bias = np.full(8, 3.0, dtype=np.float32)
        y = ln(x)
        assert np.allclose(y, 3.0, atol=1e-5)

    def test_no_bias(self):
        x = np.ones((2, 8), dtype=np.float32)
        ln = FusedLayerNorm(8, bias=False)
        assert ln.bias is None

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        ln = FusedLayerNorm(8)
        y = ln(x)
        assert y.shape == (2, 3, 8)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        ln = FusedLayerNorm(4)
        y = ln(x)
        assert np.all(np.isfinite(y))

    def test_dtype_conversion(self):
        x = np.random.randn(2, 8).astype(np.float64)
        ln = FusedLayerNorm(8)
        y = ln(x)
        assert y.dtype == np.float32

    def test_call_forward_equivalence(self):
        x = np.random.randn(2, 8).astype(np.float32)
        ln = FusedLayerNorm(8)
        assert np.allclose(ln(x), ln.forward(x))


# =============================================================================
# FusedRMSNorm
# =============================================================================

class TestFusedRMSNorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        rms = FusedRMSNorm(8)
        y = rms(x)
        assert y.shape == x.shape

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        rms = FusedRMSNorm(8)
        rms.weight = np.full(8, 2.0, dtype=np.float32)
        y = rms(x)
        assert np.allclose(y, 2.0, atol=1e-5)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        rms = FusedRMSNorm(4)
        y = rms(x)
        assert np.all(np.isfinite(y))

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        rms = FusedRMSNorm(8)
        y = rms(x)
        assert y.shape == (2, 3, 8)

    def test_dtype_conversion(self):
        x = np.random.randn(2, 8).astype(np.float64)
        rms = FusedRMSNorm(8)
        y = rms(x)
        assert y.dtype == np.float32

    def test_call_forward_equivalence(self):
        x = np.random.randn(2, 8).astype(np.float32)
        rms = FusedRMSNorm(8)
        assert np.allclose(rms(x), rms.forward(x))


# =============================================================================
# FusedCrossEntropyLoss
# =============================================================================

class TestFusedCrossEntropyLoss:
    def test_perfect_prediction_low_loss(self):
        logits = np.array([[10.0, 0.1, 0.1], [0.1, 10.0, 0.1]], dtype=np.float32)
        targets = np.array([0, 1], dtype=np.int64)
        ce = FusedCrossEntropyLoss()
        loss = ce(logits, targets)
        assert loss < 0.1

    def test_bad_prediction_high_loss(self):
        logits = np.array([[0.1, 10.0, 0.1], [10.0, 0.1, 0.1]], dtype=np.float32)
        targets = np.array([0, 0], dtype=np.int64)
        ce = FusedCrossEntropyLoss()
        loss = ce(logits, targets)
        assert loss > 4.0

    def test_ignore_index(self):
        logits = np.array([[10.0, 0.1], [0.1, 10.0]], dtype=np.float32)
        targets = np.array([-100, 1], dtype=np.int64)
        ce = FusedCrossEntropyLoss(ignore_index=-100)
        loss = ce(logits, targets)
        assert loss < 10.0

    def test_all_ignored_returns_zero(self):
        logits = np.array([[10.0, 0.1], [0.1, 10.0]], dtype=np.float32)
        targets = np.array([-100, -100], dtype=np.int64)
        ce = FusedCrossEntropyLoss(ignore_index=-100)
        loss = ce(logits, targets)
        assert loss == 0.0

    def test_label_smoothing(self):
        logits = np.array([[10.0, 0.1, 0.1]], dtype=np.float32)
        targets = np.array([0], dtype=np.int64)
        ce_no_smooth = FusedCrossEntropyLoss(label_smoothing=0.0)
        ce_smooth = FusedCrossEntropyLoss(label_smoothing=0.1)
        loss_no = ce_no_smooth(logits, targets)
        loss_smooth = ce_smooth(logits, targets)
        assert loss_smooth != loss_no

    def test_call_forward_equivalence(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,))
        ce = FusedCrossEntropyLoss()
        assert ce(logits, targets) == ce.forward(logits, targets)


# =============================================================================
# FusedAttentionBias
# =============================================================================

class TestFusedAttentionBias:
    def test_output_shape(self):
        B, N, S, H, E = 2, 4, 6, 3, 8
        q = np.random.randn(B, N, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        attn = FusedAttentionBias(num_heads=H)
        out, w = attn(q, k, v)
        assert out.shape == (B, N, H, E)
        assert w.shape == (B, H, N, S)

    def test_weights_sum_to_one(self):
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 6, 2, 8).astype(np.float32)
        v = np.random.randn(1, 6, 2, 8).astype(np.float32)
        attn = FusedAttentionBias(num_heads=2)
        _, w = attn(q, k, v)
        sums = w.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_with_bias(self):
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        bias = np.random.randn(1, 2, 4, 4).astype(np.float32)
        attn = FusedAttentionBias(num_heads=2)
        out, w = attn(q, k, v, attn_bias=bias)
        assert out.shape == q.shape

    def test_causal_mask(self):
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        attn = FusedAttentionBias(num_heads=2)
        _, w_causal = attn(q, k, v, causal=True)
        _, w_no_causal = attn(q, k, v, causal=False)
        assert not np.allclose(w_causal, w_no_causal)

    def test_scale(self):
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        attn = FusedAttentionBias(num_heads=2)
        _, w1 = attn(q, k, v, scale=0.5)
        _, w2 = attn(q, k, v, scale=2.0)
        assert not np.allclose(w1, w2)


# =============================================================================
# ChunkedOperation
# =============================================================================

class TestChunkedOperation:
    def test_output_shape(self):
        B, S, H, E = 1, 8, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        out, w = chunked.attention_chunked(q, k, v)
        assert out.shape == (B, S, H, E)

    def test_weights_sum_to_one(self):
        B, S, H, E = 1, 8, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        _, w = chunked.attention_chunked(q, k, v)
        sums = w.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_chunk_size_parameter(self):
        B, S, H, E = 1, 12, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=512)
        out, w = chunked.attention_chunked(q, k, v, chunk_size=3)
        assert out.shape == (B, S, H, E)

    def test_single_chunk(self):
        B, S, H, E = 1, 4, 2, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        chunked = ChunkedOperation(chunk_size=4)
        out, w = chunked.attention_chunked(q, k, v)
        assert out.shape == (B, S, H, E)


# =============================================================================
# MemoryEfficientSoftmax
# =============================================================================

class TestMemoryEfficientSoftmax:
    def test_sums_to_one(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        sums = result.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_stable_mode(self):
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert np.all(np.isfinite(result))
        assert np.allclose(result.sum(), 1.0, atol=1e-5)

    def test_unstable_mode(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=False)
        assert np.allclose(result.sum(), 1.0, atol=1e-5)

    def test_chunked_mode(self):
        logits = np.random.randn(2, 20).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1, stable=True, chunk_size=5)
        sums = result.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-4)

    def test_large_logits_stable(self):
        logits = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert np.all(np.isfinite(result))

    def test_2d_input(self):
        logits = np.random.randn(3, 8).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert result.shape == logits.shape


# =============================================================================
# FusedScaleBias
# =============================================================================

class TestFusedScaleBias:
    def test_identity(self):
        x = np.random.randn(2, 8).astype(np.float32)
        sb = FusedScaleBias(8)
        y = sb(x)
        assert np.allclose(y, x)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32)
        sb = FusedScaleBias(8)
        sb.weight = np.full(8, 3.0, dtype=np.float32)
        y = sb(x)
        assert np.allclose(y, 3.0)

    def test_bias_shifts(self):
        x = np.zeros((2, 8), dtype=np.float32)
        sb = FusedScaleBias(8)
        sb.bias = np.full(8, 5.0, dtype=np.float32)
        y = sb(x)
        assert np.allclose(y, 5.0)

    def test_combined(self):
        x = np.ones((2, 8), dtype=np.float32)
        sb = FusedScaleBias(8)
        sb.weight = np.full(8, 2.0, dtype=np.float32)
        sb.bias = np.full(8, 1.0, dtype=np.float32)
        y = sb(x)
        assert np.allclose(y, 3.0)

    def test_call_forward_equivalence(self):
        x = np.random.randn(2, 8).astype(np.float32)
        sb = FusedScaleBias(8)
        assert np.allclose(sb(x), sb.forward(x))


# =============================================================================
# OptimizedEmbedding
# =============================================================================

class TestOptimizedEmbedding:
    def test_output_shape(self):
        emb = OptimizedEmbedding(100, 16)
        x = np.array([0, 1, 2], dtype=np.int64)
        y = emb(x)
        assert y.shape == (3, 16)

    def test_2d_input(self):
        emb = OptimizedEmbedding(100, 16)
        x = np.array([[0, 1], [2, 3]], dtype=np.int64)
        y = emb(x)
        assert y.shape == (2, 2, 16)

    def test_clipping(self):
        emb = OptimizedEmbedding(10, 8)
        x = np.array([-1, 0, 9, 10], dtype=np.int64)
        y = emb(x)
        assert y.shape == (4, 8)

    def test_quantize_uint8(self):
        emb = OptimizedEmbedding(100, 16, quantize=True)
        emb.quantize_weight(dtype="uint8")
        assert emb._quantized is not None
        assert emb._quantized.dtype == np.uint8

    def test_quantize_int8(self):
        emb = OptimizedEmbedding(100, 16, quantize=True)
        emb.quantize_weight(dtype="int8")
        assert emb._quantized is not None
        assert emb._quantized.dtype == np.int8

    def test_call_forward_equivalence(self):
        emb = OptimizedEmbedding(100, 16)
        x = np.array([0, 5, 10], dtype=np.int64)
        assert np.allclose(emb(x), emb.forward(x))


# =============================================================================
# fused_swiglu
# =============================================================================

class TestFusedSwiglu:
    def test_output_shape(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w1 = np.random.randn(16, 8).astype(np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        w2 = np.random.randn(8, 16).astype(np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        w3 = np.random.randn(16, 8).astype(np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        y = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert y.shape == (2, 8)

    def test_output_finite(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w1 = np.random.randn(16, 8).astype(np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        w2 = np.random.randn(8, 16).astype(np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        w3 = np.random.randn(16, 8).astype(np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        y = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert np.all(np.isfinite(y))


# =============================================================================
# efficient_cross_entropy
# =============================================================================

class TestEfficientCrossEntropy:
    def test_perfect_prediction_low_loss(self):
        logits = np.array([[10.0, 0.1], [0.1, 10.0]], dtype=np.float32)
        targets = np.array([0, 1], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets)
        assert loss < 0.1

    def test_ignore_index(self):
        logits = np.array([[10.0, 0.1], [0.1, 10.0]], dtype=np.float32)
        targets = np.array([-100, 1], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets, ignore_index=-100)
        assert loss < 0.1

    def test_all_ignored_returns_zero(self):
        logits = np.array([[10.0, 0.1], [0.1, 10.0]], dtype=np.float32)
        targets = np.array([-100, -100], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets, ignore_index=-100)
        assert loss == 0.0


# =============================================================================
# chunked_matmul
# =============================================================================

class TestChunkedMatmul:
    def test_small_matrices(self):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 6).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=512)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)

    def test_large_matrices_chunked(self):
        a = np.random.randn(1024, 64).astype(np.float32)
        b = np.random.randn(64, 128).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=256)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-4)

    def test_identity_matrix(self):
        a = np.random.randn(10, 10).astype(np.float32)
        b = np.eye(10, dtype=np.float32)
        result = chunked_matmul(a, b, chunk_size=5)
        assert np.allclose(result, a, atol=1e-5)


# =============================================================================
# ragged_to_padded
# =============================================================================

class TestRaggedToPadded:
    def test_basic(self):
        tokens = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert padded.shape == tokens.shape
        assert mask.shape == tokens.shape

    def test_mask_values(self):
        tokens = np.array([[1, 2, 3], [4, 0, 0]], dtype=np.int64)
        _, mask = ragged_to_padded(tokens, pad_token_id=0)
        expected_mask = np.array([[True, True, True], [True, False, False]])
        assert np.array_equal(mask, expected_mask)

    def test_no_padding(self):
        tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        _, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert mask.all()


# =============================================================================
# estimate_attention_memory
# =============================================================================

class TestEstimateAttentionMemory:
    def test_basic_estimate(self):
        mem = estimate_attention_memory(batch_size=2, seq_len=128, num_heads=8, head_dim=64)
        expected = 2 * 8 * 128 * 128 * 2 / (1024 ** 2)
        assert mem == pytest.approx(expected)

    def test_zero_batch(self):
        mem = estimate_attention_memory(batch_size=0, seq_len=128, num_heads=8, head_dim=64)
        assert mem == 0.0

    def test_precision_bytes(self):
        mem2 = estimate_attention_memory(batch_size=1, seq_len=64, num_heads=4, head_dim=32, precision_bytes=2)
        mem4 = estimate_attention_memory(batch_size=1, seq_len=64, num_heads=4, head_dim=32, precision_bytes=4)
        assert mem4 == pytest.approx(mem2 * 2)


# =============================================================================
# silu
# =============================================================================

class TestSiLU:
    def test_zero(self):
        assert silu(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        y = silu(x)
        assert np.all(y > 0)

    def test_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        y = silu(x)
        assert np.all(y < 0)

    def test_large_positive(self):
        x = np.array([100.0])
        y = silu(x)
        assert np.allclose(y, 100.0, atol=1e-3)


# =============================================================================
# gelu
# =============================================================================

class TestGELU:
    def test_zero(self):
        assert gelu(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        y = gelu(x)
        assert np.all(y > 0)

    def test_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        y = gelu(x)
        assert np.all(y < 0)

    def test_approximates_relu_for_large_positive(self):
        x = np.array([10.0, 20.0, 30.0])
        y = gelu(x)
        assert np.allclose(y, x, atol=1.0)
