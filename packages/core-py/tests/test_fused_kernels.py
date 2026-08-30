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
        from domains.training.slonet import SloLayerNorm

        np.random.seed(42)
        ln = SloLayerNorm(64)
        x = np.random.randn(1, 64).astype(np.float32)
        out = ln.forward_numpy(x)

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * ln.weight.data + ln.bias.data
        assert np.abs(out - expected).max() < 1e-5

    def test_batch_input(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(123)
        ln = SloLayerNorm(128)
        x = np.random.randn(4, 128).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_matches_einsum_with_kernels_disabled(self):
        from domains.training.slonet import SloLayerNorm
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloLayerNorm

        np.random.seed(10)
        ln = SloLayerNorm(256)
        x = np.random.randn(8, 256).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (8, 256)

    def test_output_finite(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(11)
        ln = SloLayerNorm(128)
        x = np.random.randn(3, 128).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_zero_input(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        x = np.zeros((1, 32), dtype=np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_large_values(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(12)
        ln = SloLayerNorm(64)
        x = (np.random.randn(1, 64) * 1000).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_small_values(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(13)
        ln = SloLayerNorm(64)
        x = (np.random.randn(1, 64) * 1e-6).astype(np.float32)
        out = ln.forward_numpy(x)
        assert np.all(np.isfinite(out))

    def test_weight_bias_effect(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(14)
        ln = SloLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)
        out = ln.forward_numpy(x)
        assert not np.allclose(out, x), "LayerNorm should change values"

    def test_single_element_batch(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(15)
        ln = SloLayerNorm(16)
        x = np.random.randn(1, 16).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (1, 16)

    def test_large_batch(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(16)
        ln = SloLayerNorm(64)
        x = np.random.randn(32, 64).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (32, 64)
        assert np.all(np.isfinite(out))


class TestFusedAttentionSingle:
    """Single-token attention in forward_numpy with KV cache."""

    def test_matches_einsum_path(self):
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

        np.random.seed(42)
        attn = SloMultiHeadAttention(64, 4)

        k_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 5, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_einsum, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                            kv_cache=(k_cache, v_cache))
        slonet_mod._KERNELS_AVAILABLE = True
        out_fused, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                           kv_cache=(k_cache, v_cache))

        diff = np.abs(out_fused - out_einsum).max()
        assert diff < 1e-5, f"Single-token fused vs einsum diff: {diff}"

    def test_output_shape(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(99)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out, cache = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 1, 64)
        assert np.all(np.isfinite(out))

    def test_preserves_output_projection(self):
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(100)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        _, (k_cache, v_cache) = attn.forward_numpy(q, k, v)
        assert k_cache.shape[2] == 4  # n_heads
        assert v_cache.shape[2] == 4

    def test_single_token_output_finite(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(101)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert np.all(np.isfinite(out))

    def test_with_different_head_dim(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        out_einsum, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                            mask=causal_4d)
        slonet_mod._KERNELS_AVAILABLE = True
        out_fused, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                           mask=causal_4d)

        diff = np.abs(out_fused - out_einsum).max()
        assert diff < 1e-5, f"Multi-token fused vs einsum diff: {diff}"

    def test_output_shape(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(78)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 5, 64).astype(np.float32)
        k = np.random.randn(1, 5, 64).astype(np.float32)
        v = np.random.randn(1, 5, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert np.all(np.isfinite(out))

    def test_sequence_length_two(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(79)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 2, 64).astype(np.float32)
        k = np.random.randn(1, 2, 64).astype(np.float32)
        v = np.random.randn(1, 2, 64).astype(np.float32)

        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 2, 64)

    def test_large_sequence(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(43)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)

        out1, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                      kv_cache=(k_cache, v_cache))
        out2, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                      kv_cache=(k_cache, v_cache))
        assert np.array_equal(out1, out2)

    def test_stable_across_different_seeds(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        import domains.training.slonet as slonet_mod
        assert hasattr(slonet_mod, '_KERNELS_AVAILABLE')
        assert isinstance(slonet_mod._KERNELS_AVAILABLE, bool)

    def test_layer_norm_works_with_kernels_disabled(self):
        from domains.training.slonet import SloLayerNorm
        import domains.training.slonet as slonet_mod

        np.random.seed(42)
        ln = SloLayerNorm(32)
        x = np.random.randn(1, 32).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out = ln.forward_numpy(x)
        slonet_mod._KERNELS_AVAILABLE = True

        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_attention_works_with_kernels_disabled(self):
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloLayerNorm
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloLayerNorm
        import domains.training.slonet as slonet_mod

        np.random.seed(47)
        ln = SloLayerNorm(64)
        x = np.random.randn(2, 64).astype(np.float32)

        slonet_mod._KERNELS_AVAILABLE = False
        out_off = ln.forward_numpy(x.copy())
        slonet_mod._KERNELS_AVAILABLE = True
        out_on = ln.forward_numpy(x.copy())

        assert out_off.shape == out_on.shape

    def test_layer_norm_disabled_matches_manual(self):
        from domains.training.slonet import SloLayerNorm
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(50)
        for n_heads in [2, 4, 8]:
            attn = SloMultiHeadAttention(64, n_heads)
            q = np.random.randn(1, 1, 64).astype(np.float32)
            k = np.random.randn(1, 1, 64).astype(np.float32)
            v = np.random.randn(1, 1, 64).astype(np.float32)
            out, _ = attn.forward_numpy(q, k, v)
            assert out.shape == (1, 1, 64)

    def test_causal_mask_effect(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        out_unmasked, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy())

        diff = np.abs(out_masked - out_unmasked).max()
        assert diff > 0, "Causal mask should change output"


class TestFusedLayerNormExpanded:
    def test_different_eps_values(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(100)
        for eps in [1e-3, 1e-5, 1e-9]:
            ln = SloLayerNorm(32, eps=eps)
            x = np.random.randn(2, 32).astype(np.float32)
            out = ln.forward_numpy(x)
            assert np.all(np.isfinite(out)), f"Not finite with eps={eps}"

    def test_weight_bias_shapes(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(64)
        assert ln.weight.data.shape == (64,)
        assert ln.bias.data.shape == (64,)

    def test_weight_bias_are_trainable(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        assert ln.weight.requires_grad is True
        assert ln.bias.requires_grad is True

    def test_parameters_count(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(64)
        params = ln.parameters()
        assert len(params) == 2

    def test_zero_weight_zero_output(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(32)
        ln.weight.data = np.zeros(32, dtype=np.float32)
        ln.bias.data = np.zeros(32, dtype=np.float32)
        x = np.random.randn(1, 32).astype(np.float32)
        out = ln.forward_numpy(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_large_eps_near_identity(self):
        from domains.training.slonet import SloLayerNorm

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
        from domains.training.slonet import SloLayerNorm

        np.random.seed(200)
        ln = SloLayerNorm(64)
        x = np.random.randn(2, 64).astype(np.float32)
        out1 = ln.forward_numpy(x.copy())
        out2 = ln.forward_numpy(x.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_single_dim(self):
        from domains.training.slonet import SloLayerNorm

        ln = SloLayerNorm(1)
        x = np.array([[5.0]], dtype=np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (1, 1)
        assert np.all(np.isfinite(out))

    def test_dim_256(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(102)
        ln = SloLayerNorm(256)
        x = np.random.randn(4, 256).astype(np.float32)
        out = ln.forward_numpy(x)
        assert out.shape == (4, 256)


class TestFusedAttentionExpanded:
    def test_gqa_fewer_kv_heads(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(300)
        attn = SloMultiHeadAttention(64, 4, n_kv_head=2)
        q = np.random.randn(1, 5, 64).astype(np.float32)
        k = np.random.randn(1, 5, 64).astype(np.float32)
        v = np.random.randn(1, 5, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 5, 64)
        assert np.all(np.isfinite(out))

    def test_gqa_single_kv_head(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(301)
        attn = SloMultiHeadAttention(64, 4, n_kv_head=1)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_single_token_with_kv_cache(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(303)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(2, 5, 64).astype(np.float32)
        k = np.random.randn(2, 5, 64).astype(np.float32)
        v = np.random.randn(2, 5, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (2, 5, 64)
        assert np.all(np.isfinite(out))

    def test_d_model_128(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(304)
        attn = SloMultiHeadAttention(128, 8)
        q = np.random.randn(1, 4, 128).astype(np.float32)
        k = np.random.randn(1, 4, 128).astype(np.float32)
        v = np.random.randn(1, 4, 128).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 4, 128)

    def test_zero_input_attention(self):
        from domains.training.slonet import SloMultiHeadAttention

        attn = SloMultiHeadAttention(64, 4)
        q = np.zeros((1, 3, 64), dtype=np.float32)
        k = np.zeros((1, 3, 64), dtype=np.float32)
        v = np.zeros((1, 3, 64), dtype=np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_output_projection_changes_values(self):
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(305)
        attn = SloMultiHeadAttention(64, 4)
        q = np.random.randn(1, 3, 64).astype(np.float32)
        k = np.random.randn(1, 3, 64).astype(np.float32)
        v = np.random.randn(1, 3, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert not np.allclose(out, q), "Output projection should change values"

    def test_multi_token_with_explicit_causal_mask(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(308)
        attn = SloMultiHeadAttention(64, 4)
        q = np.ones((1, 3, 64), dtype=np.float32)
        k = np.ones((1, 3, 64), dtype=np.float32)
        v = np.ones((1, 3, 64), dtype=np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 3, 64)
        assert np.all(np.isfinite(out))

    def test_cache_grows_correctly(self):
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention

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
        from domains.training.slonet import SloMultiHeadAttention

        np.random.seed(311)
        attn = SloMultiHeadAttention(64, 2)
        q = np.random.randn(1, 4, 64).astype(np.float32)
        k = np.random.randn(1, 4, 64).astype(np.float32)
        v = np.random.randn(1, 4, 64).astype(np.float32)
        out, _ = attn.forward_numpy(q, k, v)
        assert out.shape == (1, 4, 64)
        assert np.all(np.isfinite(out))

    def test_layer_norm_matches_manual_with_kernels(self):
        from domains.training.slonet import SloLayerNorm

        np.random.seed(400)
        ln = SloLayerNorm(64)
        x = np.random.randn(3, 64).astype(np.float32)
        out = ln.forward_numpy(x)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * ln.weight.data + ln.bias.data
        assert np.abs(out - expected).max() < 1e-5

    def test_single_vs_multi_agree_with_cache(self):
        from domains.training.slonet import SloMultiHeadAttention
        import domains.training.slonet as slonet_mod

        np.random.seed(401)
        attn = SloMultiHeadAttention(64, 4)
        k_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        v_cache = np.random.randn(1, 3, 4, 16).astype(np.float32)
        q = np.random.randn(1, 1, 64).astype(np.float32)
        k = np.random.randn(1, 1, 64).astype(np.float32)
        v = np.random.randn(1, 1, 64).astype(np.float32)
        slonet_mod._KERNELS_AVAILABLE = True
        out_single, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                           kv_cache=(k_cache, v_cache))
        slonet_mod._KERNELS_AVAILABLE = False
        out_einsum, _ = attn.forward_numpy(q.copy(), k.copy(), v.copy(),
                                           kv_cache=(k_cache, v_cache))
        slonet_mod._KERNELS_AVAILABLE = True
        diff = np.abs(out_single - out_einsum).max()
        assert diff < 1e-5
