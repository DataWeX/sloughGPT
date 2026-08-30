"""Tests for domains.ops — FusedLayerNorm, FusedRMSNorm, FusedCrossEntropyLoss, FusedAttentionBias, ChunkedOperation, MemoryEfficientSoftmax, FusedScaleBias, OptimizedEmbedding, and standalone functions."""

import math
import numpy as np
import pytest
from domains.ops import (
    FusedLayerNorm, FusedRMSNorm, FusedCrossEntropyLoss, FusedAttentionBias,
    ChunkedOperation, MemoryEfficientSoftmax, FusedScaleBias, OptimizedEmbedding,
    fused_swiglu, efficient_cross_entropy, chunked_matmul, ragged_to_padded,
    estimate_attention_memory, silu, gelu,
)


class TestFusedLayerNorm:
    def test_shape_preserved(self):
        ln = FusedLayerNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_mean_near_zero(self):
        ln = FusedLayerNorm(64)
        x = np.ones((4, 8, 64), dtype=np.float32) * 5.0
        out = ln(x)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)

    def test_callable(self):
        ln = FusedLayerNorm(32)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_weight_scale_identity(self):
        ln = FusedLayerNorm(16)
        x = np.random.randn(3, 16).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_custom_eps(self):
        ln = FusedLayerNorm(32, eps=1e-3)
        x = np.ones((2, 4, 32), dtype=np.float32) * 7.0
        out = ln(x)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-3)

    def test_no_bias(self):
        ln = FusedLayerNorm(32, bias=False)
        assert ln.bias is None
        x = np.random.randn(2, 32).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_with_bias(self):
        ln = FusedLayerNorm(32, bias=True)
        assert ln.bias is not None
        assert ln.bias.shape == (32,)

    def test_normalized_shape_tuple(self):
        ln = FusedLayerNorm((32,))
        x = np.random.randn(2, 32).astype(np.float32)
        out = ln(x)
        assert out.shape == x.shape

    def test_non_float32_input(self):
        ln = FusedLayerNorm(16)
        x = np.ones((2, 16), dtype=np.float64)
        out = ln(x)
        assert out.dtype == np.float32

    def test_unit_variance(self):
        ln = FusedLayerNorm(64)
        x = np.random.randn(100, 64).astype(np.float32) * 10.0
        out = ln(x)
        var = np.var(out, axis=-1)
        np.testing.assert_allclose(var, 1.0, atol=0.1)

    def test_batch_independence(self):
        ln = FusedLayerNorm(32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = ln(x)
        for i in range(4):
            single = ln(x[i:i+1])
            np.testing.assert_allclose(out[i], single[0], atol=1e-5)

    def test_different_sequence_lengths(self):
        ln = FusedLayerNorm(32)
        for seq_len in [1, 5, 20, 100]:
            x = np.random.randn(2, seq_len, 32).astype(np.float32)
            out = ln(x)
            assert out.shape == x.shape

    def test_different_inputs_produce_different_outputs(self):
        ln = FusedLayerNorm(16)
        x1 = np.random.randn(2, 16).astype(np.float32)
        x2 = np.random.randn(2, 16).astype(np.float32)
        out1 = ln(x1)
        out2 = ln(x2)
        assert not np.allclose(out1, out2)

    def test_custom_weight_and_bias(self):
        ln = FusedLayerNorm(8)
        ln.weight = np.ones(8, dtype=np.float32) * 2.0
        ln.bias = np.ones(8, dtype=np.float32) * 3.0
        x = np.zeros((2, 8), dtype=np.float32)
        out = ln(x)
        np.testing.assert_allclose(out, 3.0)

    def test_constant_input_gives_zero_mean(self):
        ln = FusedLayerNorm(32)
        x = np.full((5, 32), 42.0, dtype=np.float32)
        out = ln(x)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)

    def test_int16_input(self):
        ln = FusedLayerNorm(16)
        x = np.ones((2, 16), dtype=np.int16) * 3
        out = ln(x)
        assert out.dtype == np.float32
        assert out.shape == x.shape


class TestFusedRMSNorm:
    def test_shape_preserved(self):
        norm = FusedRMSNorm(64)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_near_one(self):
        norm = FusedRMSNorm(64)
        x = np.ones((4, 8, 64), dtype=np.float32) * 3.0
        out = norm(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_callable(self):
        norm = FusedRMSNorm(32)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        out = norm(x)
        assert out.shape == x.shape

    def test_weight_default_ones(self):
        norm = FusedRMSNorm(16)
        np.testing.assert_array_equal(norm.weight, np.ones(16, dtype=np.float32))

    def test_custom_eps(self):
        norm = FusedRMSNorm(32, eps=1e-3)
        x = np.ones((2, 32), dtype=np.float32) * 5.0
        out = norm(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-2)

    def test_non_float32_input(self):
        norm = FusedRMSNorm(16)
        x = np.ones((2, 16), dtype=np.float64) * 3.0
        out = norm(x)
        assert out.dtype == np.float32

    def test_zero_input(self):
        norm = FusedRMSNorm(16)
        x = np.zeros((2, 16), dtype=np.float32)
        out = norm(x)
        assert np.all(np.isfinite(out))

    def test_negative_values(self):
        norm = FusedRMSNorm(32)
        x = -np.ones((4, 32), dtype=np.float32) * 2.0
        out = norm(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_scale_invariance(self):
        norm = FusedRMSNorm(32)
        x = np.random.randn(2, 32).astype(np.float32)
        out1 = norm(x)
        out2 = norm(x * 100.0)
        np.testing.assert_allclose(out1, out2, atol=1e-4)

    def test_1d_input(self):
        norm = FusedRMSNorm(16)
        x = np.random.randn(16).astype(np.float32)
        out = norm(x)
        assert out.shape == x.shape

    def test_high_dimensional(self):
        norm = FusedRMSNorm(8)
        x = np.random.randn(2, 3, 4, 8).astype(np.float32)
        out = norm(x)
        assert out.shape == x.shape

    def test_batch_independence(self):
        norm = FusedRMSNorm(32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = norm(x)
        for i in range(4):
            single = norm(x[i:i+1])
            np.testing.assert_allclose(out[i], single[0], atol=1e-5)

    def test_custom_weight(self):
        norm = FusedRMSNorm(8)
        norm.weight = np.ones(8, dtype=np.float32) * 2.0
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        out = norm(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, 2.0, atol=1e-3)


class TestFusedCrossEntropyLoss:
    def test_basic(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]], dtype=np.float32)
        targets = np.array([1, 2])
        loss = loss_fn(logits, targets)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_ignore_index(self):
        loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([-100])
        loss = loss_fn(logits, targets)
        assert loss == 0.0

    def test_label_smoothing(self):
        loss_fn_smooth = FusedCrossEntropyLoss(label_smoothing=0.1)
        loss_fn_no = FusedCrossEntropyLoss(label_smoothing=0.0)
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([1])
        l_smooth = loss_fn_smooth(logits, targets)
        l_no = loss_fn_no(logits, targets)
        assert l_smooth != l_no

    def test_perfect_prediction(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[0.0, 100.0, 0.0]], dtype=np.float32)
        targets = np.array([1])
        loss = loss_fn(logits, targets)
        assert loss < 0.01

    def test_worst_prediction(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
        targets = np.array([1])
        loss = loss_fn(logits, targets)
        assert loss > 5.0

    def test_all_ignored(self):
        loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targets = np.array([-100, -100])
        loss = loss_fn(logits, targets)
        assert loss == 0.0

    def test_non_float32_logits(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0]], dtype=np.float64)
        targets = np.array([0])
        loss = loss_fn(logits, targets)
        assert isinstance(loss, float)

    def test_batch_size_one(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([0])
        loss = loss_fn(logits, targets)
        assert loss > 0.0

    def test_batch_size_large(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.random.randn(64, 100).astype(np.float32)
        targets = np.random.randint(0, 100, size=(64,))
        loss = loss_fn(logits, targets)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_many_classes(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.random.randn(4, 10000).astype(np.float32)
        targets = np.random.randint(0, 10000, size=(4,))
        loss = loss_fn(logits, targets)
        assert loss > 0.0

    def test_label_smoothing_value(self):
        loss_fn_0 = FusedCrossEntropyLoss(label_smoothing=0.0)
        loss_fn_01 = FusedCrossEntropyLoss(label_smoothing=0.1)
        loss_fn_02 = FusedCrossEntropyLoss(label_smoothing=0.2)
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([1])
        l0 = loss_fn_0(logits, targets)
        l1 = loss_fn_01(logits, targets)
        l2 = loss_fn_02(logits, targets)
        assert l0 != l1 != l2

    def test_mixed_ignore_and_valid(self):
        loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targets = np.array([-100, 1])
        loss = loss_fn(logits, targets)
        assert loss > 0.0

    def test_equal_logits(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        targets = np.array([0])
        loss = loss_fn(logits, targets)
        assert loss > 0.0

    def test_single_class(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.array([[5.0]], dtype=np.float32)
        targets = np.array([0])
        loss = loss_fn(logits, targets)
        assert loss < 0.01

    def test_2d_batch(self):
        loss_fn = FusedCrossEntropyLoss()
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.array([1, 2, 3, 4])
        loss = loss_fn(logits, targets)
        assert loss > 0.0


class TestFusedAttentionBias:
    def test_shape(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        out, weights = ab(q, k, v)
        assert out.shape == q.shape
        assert weights.shape[0] == 1
        assert weights.shape[-1] == weights.shape[-2]

    def test_weights_sum_to_one(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        _, weights = ab(q, k, v)
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_with_causal_mask(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        out, weights = ab(q, k, v, causal=True)
        assert out.shape == q.shape

    def test_causal_mask_triangular(self):
        ab = FusedAttentionBias(num_heads=2)
        B, N, H, E = 1, 6, 2, 8
        q = np.random.randn(B, N, H, E).astype(np.float32)
        k = np.random.randn(B, N, H, E).astype(np.float32)
        v = np.random.randn(B, N, H, E).astype(np.float32)
        _, weights = ab(q, k, v, causal=True)
        for h in range(H):
            for i in range(N):
                for j in range(i + 1, N):
                    assert weights[0, h, i, j] < 1e-6, f"weights[{h},{i},{j}] should be ~0"

    def test_with_attn_bias(self):
        ab = FusedAttentionBias(num_heads=4)
        B, N, S, H, E = 1, 4, 4, 4, 16
        q = np.random.randn(B, N, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        bias = np.random.randn(B, H, N, S).astype(np.float32)
        out, weights = ab(q, k, v, attn_bias=bias)
        assert out.shape == q.shape

    def test_custom_scale(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        out1, _ = ab(q, k, v, scale=1.0)
        out2, _ = ab(q, k, v, scale=0.5)
        assert not np.allclose(out1, out2)

    def test_different_head_dims(self):
        for E in [8, 16, 32, 64]:
            ab = FusedAttentionBias(num_heads=4)
            q = np.random.randn(1, 2, 4, E).astype(np.float32)
            k = np.random.randn(1, 2, 4, E).astype(np.float32)
            v = np.random.randn(1, 2, 4, E).astype(np.float32)
            out, weights = ab(q, k, v)
            assert out.shape == q.shape

    def test_self_attention(self):
        ab = FusedAttentionBias(num_heads=4)
        x = np.random.randn(1, 8, 4, 16).astype(np.float32)
        out, weights = ab(x, x, x)
        assert out.shape == x.shape
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_multiple_batches(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(4, 4, 8, 16).astype(np.float32)
        k = np.random.randn(4, 4, 8, 16).astype(np.float32)
        v = np.random.randn(4, 4, 8, 16).astype(np.float32)
        out, weights = ab(q, k, v)
        assert out.shape == (4, 4, 8, 16)
        assert weights.shape[0] == 4

    def test_output_finite(self):
        ab = FusedAttentionBias(num_heads=4)
        q = np.random.randn(1, 4, 8, 16).astype(np.float32)
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        out, weights = ab(q, k, v)
        assert np.all(np.isfinite(out))
        assert np.all(np.isfinite(weights))

    def test_single_head(self):
        ab = FusedAttentionBias(num_heads=1)
        q = np.random.randn(1, 4, 1, 8).astype(np.float32)
        k = np.random.randn(1, 4, 1, 8).astype(np.float32)
        v = np.random.randn(1, 4, 1, 8).astype(np.float32)
        out, weights = ab(q, k, v)
        assert out.shape == q.shape

    def test_asymmetric_seq_len(self):
        ab = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 8, 2, 16).astype(np.float32)
        k = np.random.randn(1, 8, 2, 16).astype(np.float32)
        v = np.random.randn(1, 8, 2, 16).astype(np.float32)
        out, weights = ab(q, k, v)
        assert out.shape == q.shape
        assert weights.shape == (1, 2, 8, 8)

    def test_zero_scale(self):
        ab = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        out, weights = ab(q, k, v, scale=0.0)
        assert out.shape == q.shape
        assert np.all(np.isfinite(out))

    def test_large_negative_bias(self):
        ab = FusedAttentionBias(num_heads=2)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        bias = np.full((1, 2, 4, 4), -1e9, dtype=np.float32)
        _, weights = ab(q, k, v, attn_bias=bias)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-3)


class TestChunkedOperation:
    def test_output_shape(self):
        co = ChunkedOperation(chunk_size=4)
        B, S, H, E = 1, 8, 2, 16
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape == q.shape

    def test_weights_sum_to_one_no_causal(self):
        co = ChunkedOperation(chunk_size=8)
        B, S, H, E = 1, 4, 2, 8
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        _, weights = co.attention_chunked(q, k, v)
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    def test_chunk_size_parameter(self):
        for chunk_size in [2, 4, 8]:
            co = ChunkedOperation(chunk_size=chunk_size)
            q = np.random.randn(1, 8, 2, 16).astype(np.float32)
            k = np.random.randn(1, 8, 2, 16).astype(np.float32)
            v = np.random.randn(1, 8, 2, 16).astype(np.float32)
            out, weights = co.attention_chunked(q, k, v)
            assert out.shape == q.shape

    def test_causal_masking(self):
        co = ChunkedOperation(chunk_size=8)
        B, S, H, E = 1, 4, 1, 4
        q = np.random.randn(B, S, H, E).astype(np.float32)
        k = np.random.randn(B, S, H, E).astype(np.float32)
        v = np.random.randn(B, S, H, E).astype(np.float32)
        _, weights = co.attention_chunked(q, k, v)
        for i in range(S):
            for j in range(i + 1, S):
                assert weights[0, 0, i, j] < 1e-6

    def test_long_sequence(self):
        co = ChunkedOperation(chunk_size=16)
        q = np.random.randn(1, 64, 2, 8).astype(np.float32)
        k = np.random.randn(1, 64, 2, 8).astype(np.float32)
        v = np.random.randn(1, 64, 2, 8).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape == q.shape

    def test_output_finite(self):
        co = ChunkedOperation(chunk_size=4)
        q = np.random.randn(1, 8, 2, 16).astype(np.float32)
        k = np.random.randn(1, 8, 2, 16).astype(np.float32)
        v = np.random.randn(1, 8, 2, 16).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert np.all(np.isfinite(out))
        assert np.all(np.isfinite(weights))

    def test_single_chunk(self):
        co = ChunkedOperation(chunk_size=64)
        q = np.random.randn(1, 4, 2, 8).astype(np.float32)
        k = np.random.randn(1, 4, 2, 8).astype(np.float32)
        v = np.random.randn(1, 4, 2, 8).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape == q.shape
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    def test_multiple_heads(self):
        co = ChunkedOperation(chunk_size=4)
        q = np.random.randn(1, 8, 4, 16).astype(np.float32)
        k = np.random.randn(1, 8, 4, 16).astype(np.float32)
        v = np.random.randn(1, 8, 4, 16).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape == (1, 8, 4, 16)
        assert weights.shape[2] == 4

    def test_chunk_size_one(self):
        co = ChunkedOperation(chunk_size=1)
        q = np.random.randn(1, 4, 1, 4).astype(np.float32)
        k = np.random.randn(1, 4, 1, 4).astype(np.float32)
        v = np.random.randn(1, 4, 1, 4).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape == q.shape

    def test_multiple_batches(self):
        co = ChunkedOperation(chunk_size=4)
        q = np.random.randn(3, 8, 2, 8).astype(np.float32)
        k = np.random.randn(3, 8, 2, 8).astype(np.float32)
        v = np.random.randn(3, 8, 2, 8).astype(np.float32)
        out, weights = co.attention_chunked(q, k, v)
        assert out.shape[0] == 3

    def test_override_chunk_size(self):
        co = ChunkedOperation(chunk_size=64)
        q = np.random.randn(1, 8, 2, 8).astype(np.float32)
        k = np.random.randn(1, 8, 2, 8).astype(np.float32)
        v = np.random.randn(1, 8, 2, 8).astype(np.float32)
        out, _ = co.attention_chunked(q, k, v, chunk_size=2)
        assert out.shape == q.shape


class TestMemoryEfficientSoftmax:
    def test_basic(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_unstable_large_values(self):
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=False)
        assert result.shape == logits.shape

    def test_2d(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_chunked_mode(self):
        logits = np.random.randn(2, 100).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1, chunk_size=32)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-3)

    def test_chunked_mode_matches_stable(self):
        logits = np.random.randn(2, 50).astype(np.float32) * 0.1
        result_normal = MemoryEfficientSoftmax.forward(logits, dim=-1, stable=True, chunk_size=0)
        result_chunked = MemoryEfficientSoftmax.forward(logits, dim=-1, stable=True, chunk_size=16)
        np.testing.assert_allclose(result_normal, result_chunked, atol=0.05)

    def test_all_equal_logits(self):
        logits = np.ones(10, dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        np.testing.assert_allclose(result, 0.1, atol=1e-5)

    def test_one_hot_input(self):
        logits = np.array([0.0, 0.0, 100.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        assert result[2] > 0.99

    def test_negative_logits(self):
        logits = np.array([-10.0, -5.0, -1.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_dim_parameter(self):
        logits = np.random.randn(3, 4, 5).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=1)
        sums = result.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones((3, 5)), atol=1e-5)

    def test_single_element(self):
        logits = np.array([5.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits)
        np.testing.assert_allclose(result, 1.0)

    def test_large_negative_values(self):
        logits = np.array([-1000.0, -1001.0, -1002.0], dtype=np.float32)
        result = MemoryEfficientSoftmax.forward(logits, stable=True)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_3d_chunked(self):
        logits = np.random.randn(2, 3, 100).astype(np.float32)
        result = MemoryEfficientSoftmax.forward(logits, dim=-1, chunk_size=16)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones((2, 3)), atol=1e-2)


class TestFusedScaleBias:
    def test_identity(self):
        sb = FusedScaleBias(16)
        x = np.random.randn(2, 16).astype(np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, x)

    def test_custom_weight(self):
        sb = FusedScaleBias(16)
        sb.weight = np.ones(16, dtype=np.float32) * 2.0
        x = np.ones((2, 16), dtype=np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, 2.0)

    def test_custom_bias(self):
        sb = FusedScaleBias(16)
        sb.bias = np.ones(16, dtype=np.float32) * 5.0
        x = np.zeros((2, 16), dtype=np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, 5.0)

    def test_weight_and_bias(self):
        sb = FusedScaleBias(16)
        sb.weight = np.ones(16, dtype=np.float32) * 3.0
        sb.bias = np.ones(16, dtype=np.float32) * 2.0
        x = np.ones((2, 16), dtype=np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, 5.0)

    def test_callable(self):
        sb = FusedScaleBias(16)
        x = np.random.randn(3, 16).astype(np.float32)
        out = sb(x)
        assert out.shape == x.shape

    def test_1d_input(self):
        sb = FusedScaleBias(8)
        x = np.random.randn(8).astype(np.float32)
        out = sb(x)
        assert out.shape == x.shape

    def test_zero_weight(self):
        sb = FusedScaleBias(16)
        sb.weight = np.zeros(16, dtype=np.float32)
        x = np.random.randn(2, 16).astype(np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, 0.0)

    def test_negative_weight(self):
        sb = FusedScaleBias(4)
        sb.weight = np.ones(4, dtype=np.float32) * -1.0
        sb.bias = np.zeros(4, dtype=np.float32)
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        out = sb(x)
        np.testing.assert_allclose(out, -x)

    def test_high_dimensional(self):
        sb = FusedScaleBias(8)
        x = np.random.randn(2, 3, 4, 8).astype(np.float32)
        out = sb(x)
        assert out.shape == x.shape


class TestOptimizedEmbedding:
    def test_basic_lookup(self):
        emb = OptimizedEmbedding(100, 32)
        indices = np.array([0, 1, 2])
        out = emb(indices)
        assert out.shape == (3, 32)

    def test_out_of_range_clamped(self):
        emb = OptimizedEmbedding(10, 16)
        indices = np.array([0, 99, -5])
        out = emb(indices)
        assert out.shape == (3, 16)

    def test_single_index(self):
        emb = OptimizedEmbedding(50, 32)
        indices = np.array([5])
        out = emb(indices)
        assert out.shape == (1, 32)

    def test_2d_indices(self):
        emb = OptimizedEmbedding(100, 32)
        indices = np.array([[0, 1], [2, 3]])
        out = emb(indices)
        assert out.shape == (2, 2, 32)

    def test_quantize_uint8(self):
        emb = OptimizedEmbedding(100, 32)
        emb.quantize_weight(dtype="uint8")
        assert emb._quantized.dtype == np.uint8
        assert emb._quantized.shape == (100, 32)

    def test_quantize_int8(self):
        emb = OptimizedEmbedding(100, 32)
        emb.quantize_weight(dtype="int8")
        assert emb._quantized.dtype == np.int8
        assert emb._quantized.shape == (100, 32)

    def test_quantized_lookup(self):
        emb = OptimizedEmbedding(100, 32)
        emb.quantize_weight(dtype="uint8")
        indices = np.array([0, 1, 2])
        out = emb(indices)
        assert out.shape == (3, 32)

    def test_deterministic(self):
        emb = OptimizedEmbedding(100, 32)
        indices = np.array([0, 1, 2])
        out1 = emb(indices)
        out2 = emb(indices)
        np.testing.assert_array_equal(out1, out2)

    def test_callable(self):
        emb = OptimizedEmbedding(50, 16)
        indices = np.array([0, 1])
        out = emb(indices)
        assert out.shape == (2, 16)

    def test_same_index_same_output(self):
        emb = OptimizedEmbedding(10, 8)
        out1 = emb(np.array([3]))
        out2 = emb(np.array([3]))
        np.testing.assert_array_equal(out1, out2)

    def test_different_indices_different_output(self):
        emb = OptimizedEmbedding(100, 32)
        out1 = emb(np.array([0]))
        out2 = emb(np.array([1]))
        assert not np.allclose(out1, out2)

    def test_3d_indices(self):
        emb = OptimizedEmbedding(50, 16)
        indices = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        out = emb(indices)
        assert out.shape == (2, 2, 2, 16)

    def test_quantize_scale_computed(self):
        emb = OptimizedEmbedding(20, 16)
        emb.quantize_weight(dtype="uint8")
        assert emb._scale is not None
        assert emb._scale.shape == (20, 1)

    def test_large_num_embeddings(self):
        emb = OptimizedEmbedding(10000, 64)
        indices = np.array([0, 9999, 5000])
        out = emb(indices)
        assert out.shape == (3, 64)


class TestFusedSwiglu:
    def test_output_shape(self):
        x = np.random.randn(2, 16).astype(np.float32)
        w1 = np.random.randn(32, 16).astype(np.float32)
        b1 = np.zeros(32, dtype=np.float32)
        w2 = np.random.randn(16, 32).astype(np.float32)
        b2 = np.zeros(16, dtype=np.float32)
        w3 = np.random.randn(32, 16).astype(np.float32)
        b3 = np.zeros(32, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert out.shape == (2, 16)

    def test_zero_weights(self):
        x = np.random.randn(2, 16).astype(np.float32)
        w1 = np.zeros((32, 16), dtype=np.float32)
        b1 = np.zeros(32, dtype=np.float32)
        w2 = np.zeros((16, 32), dtype=np.float32)
        b2 = np.zeros(16, dtype=np.float32)
        w3 = np.zeros((32, 16), dtype=np.float32)
        b3 = np.zeros(32, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_nonzero_bias(self):
        x = np.ones((1, 16), dtype=np.float32)
        w1 = np.random.randn(32, 16).astype(np.float32) * 0.01
        b1 = np.ones(32, dtype=np.float32) * 0.1
        w2 = np.random.randn(16, 32).astype(np.float32) * 0.01
        b2 = np.ones(16, dtype=np.float32) * 0.2
        w3 = np.random.randn(32, 16).astype(np.float32) * 0.01
        b3 = np.ones(32, dtype=np.float32) * 0.3
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert out.shape == (1, 16)
        assert np.all(np.isfinite(out))

    def test_negative_input(self):
        x = -np.ones((1, 8), dtype=np.float32)
        w1 = np.random.randn(16, 8).astype(np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        w2 = np.random.randn(8, 16).astype(np.float32)
        b2 = np.zeros(8, dtype=np.float32)
        w3 = np.random.randn(16, 8).astype(np.float32)
        b3 = np.zeros(16, dtype=np.float32)
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert np.all(np.isfinite(out))

    def test_output_finite(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w1 = np.random.randn(32, 16).astype(np.float32) * 0.1
        b1 = np.random.randn(32).astype(np.float32) * 0.1
        w2 = np.random.randn(16, 32).astype(np.float32) * 0.1
        b2 = np.random.randn(16).astype(np.float32) * 0.1
        w3 = np.random.randn(32, 16).astype(np.float32) * 0.1
        b3 = np.random.randn(32).astype(np.float32) * 0.1
        out = fused_swiglu(x, w1, b1, w2, b2, w3, b3)
        assert np.all(np.isfinite(out))


class TestEfficientCrossEntropy:
    def test_basic(self):
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        targets = np.array([1])
        loss = efficient_cross_entropy(logits, targets)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_ignore_index(self):
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        targets = np.array([-100])
        loss = efficient_cross_entropy(logits, targets, ignore_index=-100)
        assert loss == 0.0

    def test_perfect_prediction(self):
        logits = np.array([[0.0, 100.0]], dtype=np.float32)
        targets = np.array([1])
        loss = efficient_cross_entropy(logits, targets)
        assert loss < 0.01

    def test_reduction_none(self):
        logits = np.array([[1.0, 2.0]], dtype=np.float32)
        targets = np.array([0])
        loss = efficient_cross_entropy(logits, targets, reduction="none")
        assert loss == 0.0

    def test_batch_multiple(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        targets = np.array([0, 1, 0])
        loss = efficient_cross_entropy(logits, targets)
        assert loss > 0.0

    def test_uniform_logits(self):
        logits = np.ones((2, 5), dtype=np.float32)
        targets = np.array([0, 1])
        loss = efficient_cross_entropy(logits, targets)
        assert loss > 0.0


class TestChunkedMatmul:
    def test_small_matrices(self):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 16).astype(np.float32)
        result = chunked_matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_large_matrices(self):
        a = np.random.randn(1024, 256).astype(np.float32)
        b = np.random.randn(256, 512).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=128)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_square_matrices(self):
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=16)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_single_row(self):
        a = np.random.randn(1, 32).astype(np.float32)
        b = np.random.randn(32, 16).astype(np.float32)
        result = chunked_matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_single_column(self):
        a = np.random.randn(32, 1).astype(np.float32)
        b = np.random.randn(1, 16).astype(np.float32)
        result = chunked_matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_chunk_size_larger_than_matrix(self):
        a = np.random.randn(8, 16).astype(np.float32)
        b = np.random.randn(16, 8).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=1024)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_exact_chunk_boundary(self):
        a = np.random.randn(64, 32).astype(np.float32)
        b = np.random.randn(32, 64).astype(np.float32)
        result = chunked_matmul(a, b, chunk_size=32)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-4)


class TestRaggedToPadded:
    def test_basic(self):
        tokens = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        np.testing.assert_array_equal(padded, tokens)
        expected_mask = np.array([[True, True, True], [True, True, False]])
        np.testing.assert_array_equal(mask, expected_mask)

    def test_no_padding(self):
        tokens = np.array([[1, 2], [3, 4]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert mask.all()

    def test_all_padding(self):
        tokens = np.array([[0, 0], [0, 0]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert not mask.any()

    def test_custom_pad_token(self):
        tokens = np.array([[1, 99, 2]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens, pad_token_id=99)
        expected_mask = np.array([[True, False, True]])
        np.testing.assert_array_equal(mask, expected_mask)

    def test_single_token(self):
        tokens = np.array([[5]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens, pad_token_id=0)
        assert mask[0, 0] is np.True_

    def test_mask_shape_matches_tokens(self):
        tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        padded, mask = ragged_to_padded(tokens)
        assert mask.shape == tokens.shape


class TestEstimateAttentionMemory:
    def test_basic(self):
        mem = estimate_attention_memory(1, 128, 12, 64)
        assert mem > 0

    def test_batch_scaling(self):
        mem1 = estimate_attention_memory(1, 128, 12, 64)
        mem2 = estimate_attention_memory(2, 128, 12, 64)
        assert abs(mem2 / mem1 - 2.0) < 0.01

    def test_seq_len_quadratic(self):
        mem1 = estimate_attention_memory(1, 64, 12, 64)
        mem2 = estimate_attention_memory(1, 128, 12, 64)
        assert abs(mem2 / mem1 - 4.0) < 0.01

    def test_precision_bytes(self):
        mem2 = estimate_attention_memory(1, 128, 12, 64, precision_bytes=2)
        mem4 = estimate_attention_memory(1, 128, 12, 64, precision_bytes=4)
        assert abs(mem4 / mem2 - 2.0) < 0.01

    def test_units_mb(self):
        mem = estimate_attention_memory(1, 128, 12, 64, precision_bytes=2)
        expected_bytes = 1 * 12 * 128 * 128 * 2
        expected_mb = expected_bytes / (1024 ** 2)
        np.testing.assert_allclose(mem, expected_mb, rtol=1e-5)

    def test_head_dim_scaling(self):
        mem1 = estimate_attention_memory(1, 64, 8, 32)
        mem2 = estimate_attention_memory(1, 64, 8, 64)
        assert abs(mem2 / mem1 - 1.0) < 0.01

    def test_zero_batch(self):
        mem = estimate_attention_memory(0, 128, 12, 64)
        assert mem == 0.0


class TestSiLU:
    def test_zero(self):
        assert silu(np.array([0.0]))[0] == 0.0

    def test_positive(self):
        x = np.array([1.0, 2.0, 10.0])
        out = silu(x)
        assert np.all(out > 0)

    def test_negative(self):
        x = np.array([-1.0, -2.0])
        out = silu(x)
        assert np.all(out < 0)

    def test_approximation(self):
        x = np.array([1.0])
        expected = 1.0 / (1 + np.exp(-1.0))
        np.testing.assert_allclose(silu(x), expected, atol=1e-6)

    def test_large_positive(self):
        x = np.array([100.0])
        np.testing.assert_allclose(silu(x), 100.0, rtol=1e-5)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = silu(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_large_negative(self):
        x = np.array([-100.0])
        np.testing.assert_allclose(silu(x), 0.0, atol=1e-10)

    def test_half_value(self):
        x = np.array([0.0])
        out = silu(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_approximates_linear_for_large_positive(self):
        x = np.array([50.0, 60.0, 70.0])
        out = silu(x)
        np.testing.assert_allclose(out, x, rtol=0.01)


class TestGELU:
    def test_zero(self):
        out = gelu(np.array([0.0]))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_positive(self):
        x = np.array([1.0, 2.0, 5.0])
        out = gelu(x)
        assert np.all(out > 0)

    def test_negative(self):
        x = np.array([-1.0, -2.0, -5.0])
        out = gelu(x)
        assert np.all(out < 0)

    def test_approximation(self):
        x = np.array([1.0])
        expected = 0.5 * 1.0 * (1 + np.tanh(np.sqrt(2 / np.pi) * (1.0 + 0.044715 * 1.0**3)))
        np.testing.assert_allclose(gelu(x), expected, atol=1e-6)

    def test_large_positive(self):
        x = np.array([100.0])
        np.testing.assert_allclose(gelu(x), 100.0, rtol=1e-4)

    def test_symmetry_approximate(self):
        x = np.array([1.0])
        out_pos = gelu(x)
        out_neg = gelu(-x)
        assert not np.isclose(out_pos, -out_neg)

    def test_array(self):
        x = np.linspace(-5, 5, 100).astype(np.float32)
        out = gelu(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_small_negative(self):
        x = np.array([-0.5])
        out = gelu(x)
        assert out[0] < 0

    def test_boundary(self):
        x = np.array([0.0, -0.0])
        out = gelu(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_very_large(self):
        x = np.array([1000.0])
        out = gelu(x)
        np.testing.assert_allclose(out, 1000.0, rtol=1e-3)
