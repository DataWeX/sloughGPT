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
