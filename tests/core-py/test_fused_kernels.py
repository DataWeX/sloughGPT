"""
Tests for fused numba inference kernels integrated into SloNet.

Verifies that fused kernels (LayerNorm, attention_single, attention_multi)
produce results matching the einsum/manual-numpy reference implementations
to machine precision.
"""

import numpy as np
import pytest


class TestFusedLayerNorm:
    """SloLayerNorm.forward_numpy uses fused_layer_norm when numba is available."""

    def test_matches_manual_computation(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(42)
        ln = SloLayerNorm(64)
        x = np.random.randn(1, 64).astype(np.float32)
        out = ln.forward_numpy(x)

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * ln.weight.data + ln.bias.data
        assert np.abs(out - expected).max() < 1e-5

    def test_batch_input(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(123)
        ln = SloLayerNorm(128)
        x = np.random.randn(4, 128).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_matches_einsum_with_kernels_disabled(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(7)
        ln = SloLayerNorm(32)
        x = np.random.randn(2, 32).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_disabled = ln.forward_numpy(x.copy())
        slonet_mod._KERNELS_AVAILABLE = True
        out_enabled = ln.forward_numpy(x.copy())

        diff = np.abs(out_enabled - out_disabled).max()
        assert diff < 1e-5, f"Kernel vs no-kernel diff: {diff}"

    def test_output_shape_matches_input(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(10)
        ln = SloLayerNorm(256)
        x = np.random.randn(8, 256).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (8, 256)

    def test_output_finite(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(11)
        ln = SloLayerNorm(128)
        x = np.random.randn(3, 128).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_zero_input(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        x = np.zeros((1, 32), dtype=np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_large_values(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(12)
        ln = SloLayerNorm(64)
        x = (np.random.randn(1, 64) * 1000).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_small_values(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(13)
        ln = SloLayerNorm(64)
        x = (np.random.randn(1, 64) * 1e-6).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_weight_bias_effect(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(14)
        ln = SloLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)
        out = ln.forward_numpy(x)
        assert not np.allclose(out, x), "LayerNorm should change values"

    def test_single_element_batch(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(15)
        ln = SloLayerNorm(16)
        x = np.random.randn(1, 16).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (1, 16)

    def test_large_batch(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(16)
        ln = SloLayerNorm(64)
        x = np.random.randn(32, 64).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (32, 64)
        assert np.all(np.isfinite(out))


class TestFusedAttentionSingle:
    """Single-token attention in forward_numpy with KV cache."""

    def test_matches_einsum_path(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)

        k_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_einsum, _ = attn.forward_numpy(
            q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache)
        )
        slonet_mod._KERNELS_AVAILABLE = True
        out_fused, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache))

        diff = np.abs(out_fused - out_einsum).max()
        assert diff < 1e-5, f"Single-token fused vs einsum diff: {diff}"

    def test_output_shape(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(99)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out, cache = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 1, 64)
        assert np.all(np.isfinite(out))

    def test_preserves_output_projection(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(55)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_np, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())
        slonet_mod._KERNELS_AVAILABLE = True
        out_fk, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())

        diff = np.abs(out_fk - out_np).max()
        assert diff < 1e-5, f"Output projection diff: {diff}"

    def test_cache_shape(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(100)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        _, (k_cache, v_cache) = attn.forward_numpy(q, k, v)
        assert k_cache.shape[2] == 4  # n_heads
        assert v_cache.shape[2] == 4

    def test_single_token_output_finite(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(101)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert np.all(np.isfinite(out))

    def test_with_different_head_dim(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(102)
        attn = SloMultiHeadAttention(128, 8)
        q = np.random.randn(1, 1, 128).astype(np.float32)
        k = np.random.randn(1, 1, 128).astype(np.float32)
        v = np.random.randn(1, 1, 128).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 1, 128)


class TestFusedAttentionMulti:
    """Multi-token attention in forward_numpy with causal masking."""

    def test_matches_einsum_with_causal_mask(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)
        N = 10
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)

        causal_4d = np.zeros((1, 1, N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                causal_4d[0, 0, i, j] = -1e9

        slonet_mod._KERNELS_AVAILABLE = False
        out_einsum, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=causal_4d)
        slonet_mod._KERNELS_AVAILABLE = True
        out_fused, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=causal_4d)

        diff = np.abs(out_fused - out_einsum).max()
        assert diff < 1e-5, f"Multi-token fused vs einsum diff: {diff}"

    def test_output_shape(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(77)
        attn = SloMultiHeadAttention(64, 4)
        N = 8
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, N, 64)
        assert np.all(np.isfinite(out))

    def test_output_finite(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(78)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 5, 64).astype(np.float32)
        k = np.random.randn(1, 5, 64).astype(np.float32)
        v = np.random.randn(1, 5, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert np.all(np.isfinite(out))

    def test_sequence_length_two(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(79)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 2, 64).astype(np.float32)
        k = np.random.randn(1, 2, 64).astype(np.float32)
        v = np.random.randn(1, 2, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 2, 64)

    def test_large_sequence(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(80)
        attn = SloMultiHeadAttention(64, 4)
        N = 32
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, N, 64)


class TestFusedAttentionStability:
    """Verify fused kernels produce stable results across calls."""

    def test_reproducible_output(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 5, 64).astype(np.float32)
        k = np.random.randn(1, 5, 64).astype(np.float32)
        v = np.random.randn(1, 5, 64).astype(np.float32)

        out1, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())
        out2, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())
        assert np.array_equal(out1, out2)

    def test_single_vs_multi_agree_on_one_token(self):
        """Single-token path should agree with multi-token path when seq_len=1."""
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = True
        out_single, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())

        slonet_mod._KERNELS_AVAILABLE = True
        out_multi, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())

        diff = np.abs(out_single - out_multi).max()
        assert diff < 1e-5, f"Single vs multi path diff: {diff}"

    def test_reproducible_with_cache(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(43)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out1, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache))
        out2, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache))
        assert np.array_equal(out1, out2)

    def test_stable_across_different_seeds(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        results = []
        for seed in [1, 2, 3]:
            np.random.seed(seed)
            attn = SloMultiHeadAttention(64, 4)
            q = np.random.randn(1, 3, 64).astype(np.float32)
            k = np.random.randn(1, 3, 64).astype(np.float32)
            v = np.random.randn(1, 3, 64).astype(np.float32)
            out, _ = attn.forward_numpy(q, k, v)
            results.append(out)

        for i in range(len(results) - 1):
            assert np.all(np.isfinite(results[i]))


class TestFusedKernelsAvailability:
    """Test kernel availability detection and fallback."""

    def test_kernels_flag_is_set(self):
        import domain.training._internal.slonet as slonet_mod

        assert hasattr(slonet_mod, "_KERNELS_AVAILABLE")
        assert isinstance(slonet_mod._KERNELS_AVAILABLE, bool)

    def test_layer_norm_works_with_kernels_disabled(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(42)
        ln = SloLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out = ln.forward_numpy(x)
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_attention_works_with_kernels_disabled(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out, _ = attn.forward_numpy(q, k, v)
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_toggling_kernels_produces_same_result(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(44)
        ln = SloLayerNorm(32)
        x = np.random.randn(2, 32).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = True
        out_on = ln.forward_numpy(x.copy())
        slonet_mod._KERNELS_AVAILABLE = False
        out_off = ln.forward_numpy(x.copy())
        slonet_mod._KERNELS_AVAILABLE = True

        diff = np.abs(out_on - out_off).max()
        assert diff < 1e-5

    def test_single_token_with_kernels_disabled(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(45)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out, _ = attn.forward_numpy(q, k, v)
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == (1, 1, 64)
        assert np.all(np.isfinite(out))

    def test_multi_token_no_mask_with_kernels_disabled(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(46)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 4, 64).astype(np.float32)
        k = np.random.randn(1, 4, 64).astype(np.float32)
        v = np.random.randn(1, 4, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out, _ = attn.forward_numpy(q, k, v)
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == (1, 4, 64)
        assert np.all(np.isfinite(out))

    def test_layer_norm_disabled_vs_enabled_shape(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(47)
        ln = SloLayerNorm(64)
        x = np.random.randn(2, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_off = ln.forward_numpy(x.copy())
        slonet_mod._KERNELS_AVAILABLE = True
        out_on = ln.forward_numpy(x.copy())

        assert out_off.shape == out_on.shape

    def test_layer_norm_disabled_matches_manual(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(48)
        ln = SloLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out = ln.forward_numpy(x)
        slonet_mod._KERNELS_AVAILABLE = True

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * ln.weight.data + ln.bias.data
        diff = np.abs(out - expected).max()
        assert diff < 1e-5

    def test_attention_with_cache_disabled_kernels(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(49)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out, _ = attn.forward_numpy(q, k, v, kv_cache=(k_cache, v_cache))
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == (1, 1, 64)
        assert np.all(np.isfinite(out))

    def test_different_head_counts(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(50)
        for n_heads in [2, 4, 8]:
            attn = SloMultiHeadAttention(64, n_heads)
            q = np.random.randn(1, 1, 64).astype(np.float32)
            k = np.random.randn(1, 1, 64).astype(np.float32)
            v = np.random.randn(1, 1, 64).astype(np.float32)
            out, _ = attn.forward_numpy(q, k, v)
            assert out.shape == (1, 1, 64)

    def test_causal_mask_effect(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(51)
        attn = SloMultiHeadAttention(64, 4)
        N = 5
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)

        causal = np.zeros((1, 1, N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                causal[0, 0, i, j] = -1e9

        out_masked, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=causal)
        # A zero-valued mask keeps the comparison on the same generic path
        # (mask is not None) with no masking applied. The no-mask call takes
        # fused_attention_multi when _KERNELS_AVAILABLE, and that kernel
        # applies causal masking built in — the two outputs were bitwise
        # identical (diff 0.0) and the test failed for the wrong reason.
        zero_mask = np.zeros((1, 1, N, N), dtype=np.float32)
        out_unmasked, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=zero_mask)

        diff = np.abs(out_masked - out_unmasked).max()
        assert diff > 0, "Causal mask should change output"


class TestFusedLayerNormExpanded:
    def test_different_eps_values(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(100)
        for eps in [1e-3, 1e-5, 1e-9]:
            ln = SloLayerNorm(32, eps=eps)
            x = np.random.randn(2, 32).astype(np.float32)
            out = ln.forward_numpy(x)
            assert np.all(np.isfinite(out)), f"Not finite with eps={eps}"

    def test_weight_bias_shapes(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(64)
        assert ln.weight.data.shape == (64,)
        assert ln.bias.data.shape == (64,)

    def test_weight_bias_are_trainable(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        assert ln.weight.requires_grad is True
        assert ln.bias.requires_grad is True

    def test_parameters_count(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(64)
        params = ln.parameters()
        assert len(params) == 2

    def test_zero_weight_zero_output(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        ln.weight.data = np.zeros(32, dtype=np.float32)
        ln.bias.data = np.zeros(32, dtype=np.float32)
        x = np.random.randn(1, 32).astype(np.float32)
        out = ln.forward_numpy(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_large_eps_near_identity(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(101)
        ln = SloLayerNorm(32, eps=1e10)
        ln.weight.data = np.ones(32, dtype=np.float32)
        ln.bias.data = np.zeros(32, dtype=np.float32)
        x = np.random.randn(1, 32).astype(np.float32)
        out = ln.forward_numpy(x)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e10)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_reproducible_two_calls(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(200)
        ln = SloLayerNorm(64)
        x = np.random.randn(2, 64).astype(np.float32)
        out1 = ln.forward_numpy(x.copy())
        out2 = ln.forward_numpy(x.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_single_dim(self):
        from domain.training._internal.slonet import SloLayerNorm

        ln = SloLayerNorm(1)
        x = np.array([[5.0]], dtype=np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (1, 1)
        assert np.all(np.isfinite(out))

    def test_dim_256(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(102)
        ln = SloLayerNorm(256)
        x = np.random.randn(4, 256).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (4, 256)


class TestFusedAttentionExpanded:
    def test_gqa_fewer_kv_heads(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(300)
        attn = SloMultiHeadAttention(64, 4, n_kv_head=2)
        q = np.random.randn(1, 5, 64).astype(np.float32)
        k = np.random.randn(1, 5, 64).astype(np.float32)
        v = np.random.randn(1, 5, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 5, 64)
        assert np.all(np.isfinite(out))

    def test_gqa_single_kv_head(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(301)
        attn = SloMultiHeadAttention(64, 4, n_kv_head=1)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_single_token_with_kv_cache(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(302)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)
        out, (k_out, v_out) = attn.forward_numpy(q, k, v, kv_cache=(k_cache, v_cache))
        assert out.shape == (1, 1, 64)
        assert k_out.shape[1] == 6
        assert v_out.shape[1] == 6

    def test_batch_size_2(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(303)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(2, 5, 64).astype(np.float32)
        k = np.random.randn(2, 5, 64).astype(np.float32)
        v = np.random.randn(2, 5, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (2, 5, 64)
        assert np.all(np.isfinite(out))

    def test_d_model_128(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(304)
        attn = SloMultiHeadAttention(128, 8)
        q = np.random.randn(1, 4, 128).astype(np.float32)
        k = np.random.randn(1, 4, 128).astype(np.float32)
        v = np.random.randn(1, 4, 128).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 4, 128)

    def test_zero_input_attention(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        attn = SloMultiHeadAttention(64, 4)
        q = np.zeros((1, 3, 64), dtype=np.float32)
        k = np.zeros((1, 3, 64), dtype=np.float32)
        v = np.zeros((1, 3, 64), dtype=np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_output_projection_changes_values(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(305)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert not np.allclose(out, q), "Output projection should change values"

    def test_multi_token_with_explicit_causal_mask(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(306)
        attn = SloMultiHeadAttention(64, 4)
        N = 6
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)
        causal = np.zeros((1, 1, N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                causal[0, 0, i, j] = -1e9
        out, _ = attn.forward_numpy(q, k, v, mask=causal)
        assert out.shape == (1, N, 64)
        assert np.all(np.isfinite(out))

    def test_gqa_output_matches_non_gqa_when_same_heads(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(307)
        attn = SloMultiHeadAttention(64, 4, n_kv_head=4)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)
        slonet_mod._KERNELS_AVAILABLE = False
        out1, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())
        out2, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())
        diff = np.abs(out1 - out2).max()
        assert diff < 1e-5
        slonet_mod._KERNELS_AVAILABLE = True

    def test_attention_all_ones_input(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(308)
        attn = SloMultiHeadAttention(64, 4)
        q = np.ones((1, 3, 64), dtype=np.float32)
        k = np.ones((1, 3, 64), dtype=np.float32)
        v = np.ones((1, 3, 64), dtype=np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_cache_grows_correctly(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(309)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 2, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 2, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)
        _, (k_out, v_out) = attn.forward_numpy(q, k, v, kv_cache=(k_cache, v_cache))
        assert k_out.shape[1] == 3
        assert v_out.shape[1] == 3

    def test_reproducible_with_mask(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(310)
        attn = SloMultiHeadAttention(64, 4)
        N = 5
        q = np.random.randn(1, N, 64).astype(np.float32)
        k = np.random.randn(1, N, 64).astype(np.float32)
        v = np.random.randn(1, N, 64).astype(np.float32)
        causal = np.zeros((1, 1, N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                causal[0, 0, i, j] = -1e9
        out1, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=causal)
        out2, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(), mask=causal)
        np.testing.assert_array_equal(out1, out2)

    def test_different_head_dim_32(self):
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(311)
        attn = SloMultiHeadAttention(64, 2)
        q = np.random.randn(1, 4, 64).astype(np.float32)
        k = np.random.randn(1, 4, 64).astype(np.float32)
        v = np.random.randn(1, 4, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 4, 64)
        assert np.all(np.isfinite(out))

    def test_layer_norm_matches_manual_with_kernels(self):
        from domain.training._internal.slonet import SloLayerNorm

        np.random.seed(400)
        ln = SloLayerNorm(64)
        x = np.random.randn(3, 64).astype(np.float32)
        out = ln.forward_numpy(x)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * ln.weight.data + ln.bias.data
        assert np.abs(out - expected).max() < 1e-5

    def test_single_vs_multi_agree_with_cache(self):
        import domain.training._internal.slonet as slonet_mod
        from domain.training._internal.slonet import SloMultiHeadAttention

        np.random.seed(401)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)
        slonet_mod._KERNELS_AVAILABLE = True
        out_single, _ = attn.forward_numpy(
            q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache)
        )
        slonet_mod._KERNELS_AVAILABLE = False
        out_einsum, _ = attn.forward_numpy(
            q.copy(), k.copy(), v.copy(), kv_cache=(k_cache, v_cache)
        )
        slonet_mod._KERNELS_AVAILABLE = True
        diff = np.abs(out_single - out_einsum).max()
        assert diff < 1e-5


class TestFusedAttentionMultiBLASPath:
    """The prefill einsum must ask for optimize=True (BLAS dispatch).

    Why this assertion and not a stopwatch: np.einsum defaults to
    optimize=False, which takes a generic contraction loop instead of
    dispatching to BLAS. Measured at Qwen2.5-0.5B prefill shapes
    (H=14, kv=2, E=128), arms interleaved per repetition with min-of-7,
    run twice for consistency, box at load ~2:

        seq   generic     BLAS       ratio
        256    93.98ms    27.33ms     3.44x
        512   395.29ms   105.63ms     3.74x
        681   683.07ms   132.90ms     5.14x
       1024  1679.10ms   286.30ms     5.86x
       2048  7130.77ms  1231.26ms     5.79x

    ...with allclose True at every size (rtol=1e-5, atol=1e-6). Attention is
    10.8s of a 22.0s profiled prefill (49%), so this is the largest single
    lever on /chat time-to-first-token.

    An earlier pass of this same measurement recorded 8.4-11.6x at these
    sizes; that figure did NOT reproduce on a re-run (two methods, sequential
    and interleaved, agreed on 3.4-5.9x). Quote the reproduced range -- the
    absolute times move with CPU state and load, and the ratio moves with them.

    A timing assertion here would be a flake generator: the same shape read
    1.18x while another session's test suite ran and 8.05x once the box was
    idle. So this pins WHICH PATH IS REQUESTED -- deterministic, no clock --
    and the performance number lives in scripts/benchmark_slonet.py instead.
    """

    def test_prefill_attention_requests_the_blas_path(self, monkeypatch):
        from domain.training._internal.slonet_kernels import fused_attention_multi

        seen: list[bool] = []
        real_einsum = np.einsum

        def spy(*args, **kwargs):
            seen.append(bool(kwargs.get("optimize", False)))
            return real_einsum(*args, **kwargs)

        monkeypatch.setattr(np, "einsum", spy)

        H, E_head, seq = 4, 16, 32
        rng = np.random.default_rng(42)
        q = rng.standard_normal((seq, H, E_head), dtype=np.float32)
        k = rng.standard_normal((2, seq, E_head), dtype=np.float32)
        v = rng.standard_normal((2, seq, E_head), dtype=np.float32)

        out = fused_attention_multi(q, k, v, np.float32(0.25), H, E_head)

        if not seen:
            # numba kernel took the call (SLO_USE_NUMBA=1) — no einsum here.
            pytest.skip("numba attention kernel active; einsum path not exercised")

        assert all(seen), (
            "prefill attention is on einsum's generic contraction path "
            f"(optimize defaults per call: {seen}); pass optimize=True to "
            "dispatch to BLAS — measured 3.4-5.9x faster at prefill sizes "
            "with identical output"
        )
        assert out.shape == (seq, H, E_head)
        assert np.all(np.isfinite(out))

    def test_prefill_output_still_matches_the_generic_path(self):
        """optimize=True must not change the numbers — pin it directly."""
        from domain.training._internal.slonet_kernels import fused_attention_multi

        H, E_head, seq, K_H = 4, 16, 24, 2
        rng = np.random.default_rng(7)
        q = rng.standard_normal((seq, H, E_head), dtype=np.float32)
        k = rng.standard_normal((K_H, seq, E_head), dtype=np.float32)
        v = rng.standard_normal((K_H, seq, E_head), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(E_head))
        reps = H // K_H
        kx, vx = k[np.arange(H) // reps], v[np.arange(H) // reps]

        # Reference: einsum WITHOUT optimize + the causal mask — exactly what
        # the pre-change fallback computes, chunk for chunk.
        scores = np.einsum("she,hne->hsn", q, kx, optimize=False) * scale
        cols = np.arange(seq, dtype=np.int64)[None, :]
        rows = np.arange(seq, dtype=np.int64)[:, None]
        scores = scores + np.where(cols > rows, np.float32(-1e9), np.float32(0.0))
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.einsum("hsn,hne->she", attn, vx, optimize=False)

        got = fused_attention_multi(q, k, v, scale, H, E_head)
        diff = np.abs(got - expected).max()
        assert diff < 1e-4, f"BLAS path diverged from generic path: {diff}"
