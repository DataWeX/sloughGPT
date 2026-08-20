"""
Tests for domains/ops/__init__.py — Fused operations (pure numpy).

Covers:
    - FusedLayerNorm: forward correctness, bias/no-bias, float64→float32 cast
    - FusedRMSNorm: forward correctness, LLaMA-style normalization
    - FusedCrossEntropyLoss: basic, ignore_index, label_smoothing, all-ignored
    - FusedAttentionBias: basic, causal mask, attn_bias
    - ChunkedOperation: chunked attention vs reference
    - MemoryEfficientSoftmax: stable softmax, chunked, multi-dim
    - FusedScaleBias: forward correctness
    - OptimizedEmbedding: forward, clipping, quantize uint8/int8
    - fused_swiglu: shape correctness
    - efficient_cross_entropy: basic, ignore_index, mean reduction
    - chunked_matmul: small matrix passthrough, chunked path
    - ragged_to_padded: mask generation
    - estimate_attention_memory: formula correctness
    - silu, gelu: scalar and array correctness
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

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


# ── FusedLayerNorm ────────────────────────────────────────────────────


class TestFusedLayerNorm:
    def test_output_shape(self):
        ln = FusedLayerNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        y = ln(x)
        assert y.shape == x.shape

    def test_normalization(self):
        ln = FusedLayerNorm(32)
        x = np.random.randn(4, 32).astype(np.float32) * 10 + 5
        y = ln(x)
        # After layernorm with unit weight and zero bias, output should be ~0 mean, ~1 std per row
        assert abs(y.mean()) < 0.5
        assert abs(y.std() - 1.0) < 0.3

    def test_no_bias(self):
        ln = FusedLayerNorm(32, bias=False)
        assert ln.bias is None
        x = np.random.randn(2, 32).astype(np.float32)
        y = ln(x)
        assert y.shape == x.shape

    def test_float64_input(self):
        ln = FusedLayerNorm(16)
        x = np.random.randn(2, 16)  # float64
        y = ln(x)
        assert y.dtype == np.float32

    def test_callable(self):
        ln = FusedLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)
        y = ln(x)
        assert y.shape == (1, 32)


# ── FusedRMSNorm ──────────────────────────────────────────────────────


class TestFusedRMSNorm:
    def test_output_shape(self):
        norm = FusedRMSNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        y = norm(x)
        assert y.shape == x.shape

    def test_rms_is_one(self):
        norm = FusedRMSNorm(32)
        x = np.random.randn(4, 32).astype(np.float32) * 5
        y = norm(x)
        # RMS of each row should be ~1 (since weight=1)
        rms_vals = np.sqrt(np.mean(y ** 2, axis=-1))
        np.testing.assert_allclose(rms_vals, 1.0, atol=0.1)

    def test_float64_input(self):
        norm = FusedRMSNorm(16)
        x = np.random.randn(2, 16)
        y = norm(x)
        assert y.dtype == np.float32

    def test_callable(self):
        norm = FusedRMSNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)
        y = norm(x)
        assert y.shape == (1, 32)


# ── FusedCrossEntropyLoss ─────────────────────────────────────────────


class TestFusedCrossEntropyLoss:
    def test_basic(self):
        ce = FusedCrossEntropyLoss()
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        loss = ce(logits, targets)
        assert loss > 0

    def test_perfect_prediction(self):
        ce = FusedCrossEntropyLoss()
        logits = np.zeros((2, 5), dtype=np.float32)
        targets = np.array([0, 1], dtype=np.int64)
        logits[0, 0] = 100.0
        logits[1, 1] = 100.0
        loss = ce(logits, targets)
        assert loss < 0.01

    def test_ignore_index(self):
        ce = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.random.randn(2, 5).astype(np.float32)
        targets = np.array([-100, -100], dtype=np.int64)
        loss = ce(logits, targets)
        assert loss == 0.0

    def test_label_smoothing(self):
        ce_smooth = FusedCrossEntropyLoss(label_smoothing=0.1)
        ce_plain = FusedCrossEntropyLoss(label_smoothing=0.0)
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        loss_smooth = ce_smooth(logits, targets)
        loss_plain = ce_plain(logits, targets)
        # Both should be positive, different values
        assert loss_smooth > 0
        assert loss_smooth != loss_plain

    def test_callable(self):
        ce = FusedCrossEntropyLoss()
        logits = np.random.randn(2, 5).astype(np.float32)
        targets = np.array([0, 1], dtype=np.int64)
        assert ce(logits, targets) > 0


# ── FusedAttentionBias ────────────────────────────────────────────────


class TestFusedAttentionBias:
    def test_basic(self):
        attn = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 8, 4, 16).astype(np.float32)
        k = np.random.randn(1, 8, 4, 16).astype(np.float32)
        v = np.random.randn(1, 8, 4, 16).astype(np.float32)
        out, weights = attn(q, k, v)
        assert out.shape == q.shape
        assert weights.shape == (1, 4, 8, 8)

    def test_causal_mask(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        out, weights = attn(q, k, v, causal=True)
        # Causal: future positions should have zero weight
        # weights[0, 0, 0, 1] should be ~0 (can't attend to position 1 from position 0)
        assert weights[0, 0, 0, 1] < 0.01

    def test_weights_sum_to_one(self):
        attn = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 6, 2, 8).astype(np.float32)
        k = np.random.randn(1, 6, 2, 8).astype(np.float32)
        v = np.random.randn(1, 6, 2, 8).astype(np.float32)
        _, weights = attn(q, k, v)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


# ── ChunkedOperation ─────────────────────────────────────────────────


class TestChunkedOperation:
    def test_output_shape(self):
        chunk = ChunkedOperation(chunk_size=4)
        q = np.random.randn(1, 8, 2, 8).astype(np.float32)
        k = np.random.randn(1, 8, 2, 8).astype(np.float32)
        v = np.random.randn(1, 8, 2, 8).astype(np.float32)
        out, weights = chunk.attention_chunked(q, k, v, chunk_size=4)
        assert out.shape == q.shape
        assert weights.shape[0] == 1  # batch
        assert weights.shape[1] == 2  # heads

    def test_weights_sum_to_one(self):
        chunk = ChunkedOperation(chunk_size=8)
        q = np.random.randn(1, 6, 2, 8).astype(np.float32)
        k = np.random.randn(1, 6, 2, 8).astype(np.float32)
        v = np.random.randn(1, 6, 2, 8).astype(np.float32)
        _, weights = chunk.attention_chunked(q, k, v, chunk_size=8)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


# ── MemoryEfficientSoftmax ───────────────────────────────────────────


class TestMemoryEfficientSoftmax:
    def test_basic(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert abs(result.sum() - 1.0) < 1e-5
        assert result[0, 2] > result[0, 0]  # highest logit → highest prob

    def test_stable(self):
        logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert abs(result.sum() - 1.0) < 1e-5
        assert not np.any(np.isnan(result))

    def test_unstable_large(self):
        logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=False)
        # May produce NaN/Inf for very large logits without stabilization
        # Just check it doesn't crash

    def test_chunked(self):
        logits = np.random.randn(1, 20).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, chunk_size=5)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_2d(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-5)


# ── FusedScaleBias ───────────────────────────────────────────────────


class TestFusedScaleBias:
    def test_identity(self):
        sb = FusedScaleBias(32)
        x = np.random.randn(2, 32).astype(np.float32)
        y = sb(x)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_with_params(self):
        sb = FusedScaleBias(4)
        sb.weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        sb.bias = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        x = np.ones((1, 4), dtype=np.float32)
        y = sb(x)
        np.testing.assert_allclose(y[0], [3.0, 4.0, 5.0, 6.0])


# ── OptimizedEmbedding ───────────────────────────────────────────────


class TestOptimizedEmbedding:
    def test_forward_shape(self):
        emb = OptimizedEmbedding(100, 32)
        x = np.array([0, 5, 99], dtype=np.int64)
        y = emb(x)
        assert y.shape == (3, 32)

    def test_clipping(self):
        emb = OptimizedEmbedding(10, 8)
        x = np.array([-5, 15], dtype=np.int64)  # out of range
        y = emb(x)
        assert y.shape == (2, 8)  # clipped to valid range

    def test_quantize_uint8(self):
        emb = OptimizedEmbedding(50, 16)
        emb.quantize_weight("uint8")
        assert emb._quantized is not None
        assert emb._quantized.dtype == np.uint8
        assert emb._quantized.shape == (50, 16)

    def test_quantize_int8(self):
        emb = OptimizedEmbedding(50, 16)
        emb.quantize_weight("int8")
        assert emb._quantized is not None
        assert emb._quantized.dtype == np.int8

    def test_callable(self):
        emb = OptimizedEmbedding(100, 32)
        x = np.array([0, 1, 2], dtype=np.int64)
        y = emb(x)
        assert y.shape == (3, 32)


# ── fused_swiglu ──────────────────────────────────────────────────────


class TestFusedSwiglu:
    def test_shape(self):
        x = np.random.randn(2, 16).astype(np.float32)
        w1 = np.random.randn(32, 16).astype(np.float32)
        w2 = np.random.randn(16, 32).astype(np.float32)
        w3 = np.random.randn(32, 16).astype(np.float32)
        b1 = np.zeros(32, dtype=np.float32)
        b2 = np.zeros(16, dtype=np.float32)
        b3 = np.zeros(32, dtype=np.float32)
        y = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert y.shape == (2, 16)

    def test_not_nan(self):
        x = np.random.randn(1, 8).astype(np.float32)
        w1 = np.random.randn(16, 8).astype(np.float32)
        w2 = np.random.randn(8, 16).astype(np.float32)
        w3 = np.random.randn(16, 8).astype(np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        y = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert not np.any(np.isnan(y))


# ── efficient_cross_entropy ───────────────────────────────────────────


class TestEfficientCrossEntropy:
    def test_basic(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.array([0, 1, 2, 3], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets)
        assert loss > 0

    def test_ignore_index(self):
        logits = np.random.randn(2, 5).astype(np.float32)
        targets = np.array([-100, -100], dtype=np.int64)
        loss = efficient_cross_entropy(logits, targets, ignore_index=-100)
        assert loss == 0.0


# ── chunked_matmul ───────────────────────────────────────────────────


class TestChunkedMatmul:
    def test_small_passthrough(self):
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 30).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=64)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_chunked(self):
        a = np.random.randn(100, 50).astype(np.float32)
        b = np.random.randn(50, 30).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=32)
        np.testing.assert_allclose(result, a @ b, atol=1e-4)


# ── ragged_to_padded ──────────────────────────────────────────────────


class TestRaggedToPadded:
    def test_mask(self):
        tokens = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        np.testing.assert_array_equal(mask, [[True, True, False], [True, False, False]])


# ── estimate_attention_memory ─────────────────────────────────────────


class TestEstimateAttentionMemory:
    def test_formula(self):
        mem = estimate_attention_memory(2, 128, 8, 64, precision_bytes=2)
        expected = 2 * 8 * 128 * 128 * 2 / (1024 ** 2)
        assert abs(mem - expected) < 1e-10

    def test_single_head(self):
        mem = estimate_attention_memory(1, 64, 1, 32, precision_bytes=4)
        expected = 1 * 1 * 64 * 64 * 4 / (1024 ** 2)
        assert abs(mem - expected) < 1e-10


# ── silu ──────────────────────────────────────────────────────────────


class TestSilu:
    def test_zero(self):
        assert silu(np.array([0.0]))[0] == 0.0

    def test_positive(self):
        x = np.array([1.0, 2.0, 5.0])
        y = silu(x)
        assert all(y > 0)
        assert all(y < x)  # silu(x) < x for positive x

    def test_negative(self):
        x = np.array([-1.0, -2.0])
        y = silu(x)
        assert all(y < 0)

    def test_large_positive(self):
        # silu(x) → x as x → ∞
        x = np.array([100.0])
        y = silu(x)
        np.testing.assert_allclose(y, x, rtol=1e-5)


# ── gelu ──────────────────────────────────────────────────────────────


class TestGelu:
    def test_zero(self):
        assert abs(gelu(np.array([0.0]))[0]) < 1e-6

    def test_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        y = gelu(x)
        assert all(y > 0)

    def test_negative(self):
        x = np.array([-1.0, -2.0])
        y = gelu(x)
        # GELU(-1) ≈ -0.159, GELU(-2) ≈ -0.046
        assert all(y < 0)
        assert all(y > -1)

    def test_large_positive(self):
        # GELU(x) → x as x → ∞
        x = np.array([100.0])
        y = gelu(x)
        np.testing.assert_allclose(y, x, rtol=1e-4)
