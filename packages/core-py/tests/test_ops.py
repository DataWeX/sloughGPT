"""Tests for inference ops — layernorm, rmsnorm, blas (numpy fallback)."""

import numpy as np
import pytest
from domains.inference.ops.layernorm import layernorm
from domains.inference.ops.rmsnorm import rmsnorm
from domains.inference.ops.blas import sgemm, is_available, _load_accelerate, _setup_sgemm


# ── LayerNorm ────────────────────────────────────────────────────────────────

class TestLayernorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == x.shape

    def test_normalized_mean_near_zero(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5)

    def test_normalized_var_near_one(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        vars = y.var(axis=-1)
        assert np.allclose(vars, 1.0, atol=1e-2)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32)
        w = np.full(8, 2.0, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        # Constant input → zero mean, zero var → 0/eps * w + b ≈ 0
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_bias_shifts(self):
        x = np.ones((2, 8), dtype=np.float32) * 5.0
        w = np.ones(8, dtype=np.float32)
        b = np.full(8, 3.0, dtype=np.float32)
        y = layernorm(x, w, b)
        # Constant input normalized to 0, then + bias
        assert np.allclose(y, 3.0, atol=1e-5)

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (2, 3, 8)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b)
        assert np.all(np.isfinite(y))

    def test_1d_input(self):
        x = np.random.randn(8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (8,)

    def test_4d_input(self):
        x = np.random.randn(2, 3, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (2, 3, 4, 8)

    def test_custom_epsilon_large(self):
        x = np.ones((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b, eps=1e-2)
        assert np.all(np.isfinite(y))

    def test_custom_epsilon_tiny(self):
        x = np.random.randn(2, 4).astype(np.float32) * 10
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b, eps=1e-12)
        assert np.all(np.isfinite(y))

    def test_negative_weights(self):
        x = np.ones((2, 8), dtype=np.float32) * 5.0
        w = np.full(8, -1.0, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        # Constant input normalized to 0, so output is 0 regardless of weight sign
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_weight_and_bias_combined(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w = np.full(8, 3.0, dtype=np.float32)
        b = np.full(8, 2.0, dtype=np.float32)
        y = layernorm(x, w, b)
        # After normalization with w=3, b=2: mean should be ~2
        assert np.allclose(y.mean(axis=-1), 2.0, atol=1e-5)

    def test_large_batch(self):
        x = np.random.randn(64, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (64, 32)

    def test_single_element_last_dim(self):
        x = np.random.randn(2, 1).astype(np.float32)
        w = np.ones(1, dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (2, 1)

    def test_high_variance_input(self):
        x = np.random.randn(4, 16).astype(np.float32) * 1000
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-4)

    def test_small_epsilon_all_finite(self):
        x = np.random.randn(10, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b, eps=1e-15)
        assert np.all(np.isfinite(y))

    def test_output_dtype(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.dtype == np.float32

    def test_preserves_batch_independence(self):
        x = np.random.randn(3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        # Each batch element is independently normalized
        for i in range(3):
            assert np.allclose(y[i].mean(), 0.0, atol=1e-5)

    def test_with_varying_scale_per_element(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0],
                       [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)

    def test_custom_eps_boundary(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b, eps=1e-8)
        assert np.all(np.isfinite(y))


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class TestRmsnorm:
    def test_output_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == x.shape

    def test_normalized_rms_near_one(self):
        x = np.random.randn(4, 16).astype(np.float32) * 10
        w = np.ones(16, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)

    def test_weight_scales(self):
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        w = np.full(8, 2.0, dtype=np.float32)
        y = rmsnorm(x, w)
        # Constant input: rms = 3.0, y = 3.0 / 3.0 * 2.0 = 2.0
        assert np.allclose(y, 2.0, atol=1e-5)

    def test_epsilon_prevents_nan(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w)
        assert np.all(np.isfinite(y))

    def test_3d_input(self):
        x = np.random.randn(2, 3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (2, 3, 8)

    def test_1d_input(self):
        x = np.random.randn(8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (8,)

    def test_4d_input(self):
        x = np.random.randn(2, 3, 4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (2, 3, 4, 8)

    def test_zero_weight(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w = np.zeros(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert np.allclose(y, 0.0)

    def test_negative_weight(self):
        x = np.ones((2, 8), dtype=np.float32) * 3.0
        w = np.full(8, -1.0, dtype=np.float32)
        y = rmsnorm(x, w)
        # Constant input rms=3, y = 3/3 * (-1) = -1
        assert np.allclose(y, -1.0, atol=1e-5)

    def test_custom_eps_large(self):
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w, eps=1.0)
        assert np.all(np.isfinite(y))

    def test_custom_eps_tiny(self):
        x = np.random.randn(2, 4).astype(np.float32) * 10
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w, eps=1e-12)
        assert np.all(np.isfinite(y))

    def test_large_batch(self):
        x = np.random.randn(64, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (64, 32)

    def test_output_dtype(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.dtype == np.float32

    def test_high_variance_input(self):
        x = np.random.randn(4, 16).astype(np.float32) * 1000
        w = np.ones(16, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)

    def test_single_element_last_dim(self):
        x = np.random.randn(2, 1).astype(np.float32)
        w = np.ones(1, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (2, 1)

    def test_preserves_direction(self):
        x = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        y = rmsnorm(x, w)
        # Row 2 is 2x row 1, so after normalization they should be equal
        assert np.allclose(y[0], y[1], atol=1e-5)

    def test_scaled_output(self):
        x = np.random.randn(2, 8).astype(np.float32) * 5.0
        w = np.full(8, 3.0, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 3.0, atol=0.1)

    def test_small_eps_all_finite(self):
        x = np.random.randn(10, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w, eps=1e-15)
        assert np.all(np.isfinite(y))

    def test_very_small_values(self):
        x = np.full((2, 4), 1e-7, dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w)
        assert np.all(np.isfinite(y))

    def test_mixed_sign_input(self):
        x = np.array([[-1.0, 1.0, -1.0, 1.0],
                       [1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y = rmsnorm(x, w)
        assert np.all(np.isfinite(y))
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)


# ── BLAS ─────────────────────────────────────────────────────────────────────

class TestBlas:
    def test_numpy_fallback(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-5)

    def test_square_matrices(self):
        a = np.eye(3, dtype=np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, b, atol=1e-5)

    def test_shape_mismatch_raises(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        with pytest.raises(ValueError):
            sgemm(a, b)

    def test_is_available_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_identity_left(self):
        a = np.eye(5, dtype=np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, b, atol=1e-5)

    def test_identity_right(self):
        a = np.random.randn(3, 5).astype(np.float32)
        b = np.eye(5, dtype=np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, a, atol=1e-5)

    def test_1x1_matrix(self):
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[4.0]], dtype=np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, [[12.0]], atol=1e-5)

    def test_very_large_matrices(self):
        a = np.random.randn(100, 64).astype(np.float32)
        b = np.random.randn(64, 100).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-4)

    def test_zero_matrix(self):
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = sgemm(a, b)
        assert np.allclose(c, 0.0, atol=1e-6)

    def test_transpose_equivalent(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        c = sgemm(a, b)
        # c should be (3,3)
        assert c.shape == (3, 3)

    def test_associativity(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        c = np.random.randn(4, 5).astype(np.float32)
        ab_c = sgemm(sgemm(a, b), c)
        a_bc = sgemm(a, sgemm(b, c))
        assert np.allclose(ab_c, a_bc, atol=1e-4)

    def test_negative_values(self):
        a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        b = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-5)

    def test_float64_input_falls_back(self):
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        # float64 is not float32, numpy fallback should still work
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-10)

    def test_output_shape(self):
        a = np.random.randn(7, 3).astype(np.float32)
        b = np.random.randn(3, 11).astype(np.float32)
        c = sgemm(a, b)
        assert c.shape == (7, 11)

    def test_symmetric_positive_definite(self):
        a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        c = sgemm(a, a.T)
        expected = np.matmul(a, a.T)
        assert np.allclose(c, expected, atol=1e-5)

    def test_scaled_result(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 2), dtype=np.float32)
        c = sgemm(a, b)
        # All ones: 2x3 @ 3x2 = all 3s
        assert np.allclose(c, 3.0, atol=1e-5)

    def test_batch_1xN(self):
        a = np.random.randn(1, 4).astype(np.float32)
        b = np.random.randn(4, 1).astype(np.float32)
        c = sgemm(a, b)
        assert c.shape == (1, 1)

    def test_repeated_multiplication(self):
        a = np.random.randn(3, 3).astype(np.float32)
        result = a.copy()
        for _ in range(5):
            result = sgemm(result, a)
        expected = np.linalg.matrix_power(a, 6)
        assert np.allclose(result, expected, atol=1e-4)


# ── BLAS Internal Functions ──────────────────────────────────────────────────

class TestBlasInternals:
    def test_load_accelerate_returns_none_or_lib(self):
        result = _load_accelerate()
        # On Linux, this should return None (no Accelerate)
        if result is None:
            assert result is None
        else:
            assert hasattr(result, "cblas_sgemm")

    def test_load_accelerate_caches(self):
        first = _load_accelerate()
        second = _load_accelerate()
        # Should return same object (cached)
        assert first is second

    def test_load_accelerate_when_unavailable(self):
        import domains.inference.ops.blas as blas_mod
        old = blas_mod._accelerate
        old_unavail = blas_mod._unavailable
        try:
            blas_mod._accelerate = None
            blas_mod._unavailable = True
            result = _load_accelerate()
            assert result is None
        finally:
            blas_mod._accelerate = old
            blas_mod._unavailable = old_unavail

    def test_load_accelerate_cache_hit(self):
        import domains.inference.ops.blas as blas_mod
        old = blas_mod._accelerate
        old_unavail = blas_mod._unavailable
        try:
            blas_mod._unavailable = False
            blas_mod._accelerate = "fake_lib"
            result = _load_accelerate()
            assert result == "fake_lib"
        finally:
            blas_mod._accelerate = old
            blas_mod._unavailable = old_unavail

    def test_sgemm_alpha_beta(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 2), dtype=np.float32)
        # When Accelerate is unavailable, numpy fallback ignores alpha/beta
        c = sgemm(a, b, alpha=2.0, beta=0.0)
        if is_available():
            # Accelerate path: 2.0 * (2x3 @ 3x2) = 2.0 * all-3s = all-6s
            assert np.allclose(c, 6.0, atol=1e-5)
        else:
            # Numpy fallback ignores alpha/beta
            expected = np.matmul(a, b)
            assert np.allclose(c, expected, atol=1e-5)

    def test_sgemm_non_square(self):
        a = np.random.randn(2, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-5)
        assert c.shape == (2, 3)

    def test_sgemm_tall_skinny(self):
        a = np.random.randn(100, 2).astype(np.float32)
        b = np.random.randn(2, 100).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-4)

    def test_sgemm_wide_skinny(self):
        a = np.random.randn(2, 100).astype(np.float32)
        b = np.random.randn(100, 2).astype(np.float32)
        c = sgemm(a, b)
        expected = np.matmul(a, b)
        assert np.allclose(c, expected, atol=1e-4)

    def test_sgemm_result_dtype(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        c = sgemm(a, b)
        assert c.dtype == np.float32

    def test_is_available_returns_bool_type(self):
        result = is_available()
        assert type(result) is bool


# ── LayerNorm Advanced ───────────────────────────────────────────────────────

class TestLayernormAdvanced:
    def test_all_same_value_input(self):
        x = np.full((4, 8), 7.0, dtype=np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        # All same values → zero variance → 0/sqrt(eps) = 0
        assert np.allclose(y, 0.0, atol=1e-5)

    def test_negative_input_values(self):
        x = -np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        means = y.mean(axis=-1)
        assert np.allclose(means, 0.0, atol=1e-5)

    def test_mixed_positive_negative(self):
        x = np.array([[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0]] * 4, dtype=np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)

    def test_weight_amplification(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.full(4, 5.0, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        y = layernorm(x, w, b)
        # Normalized mean should be ~0, variance should be ~25 (5^2)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)

    def test_bias_with_scale(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.full(4, 10.0, dtype=np.float32)
        y = layernorm(x, w, b)
        assert np.allclose(y.mean(axis=-1), 10.0, atol=1e-5)

    def test_very_small_last_dim(self):
        x = np.random.randn(4, 2).astype(np.float32)
        w = np.ones(2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (4, 2)

    def test_many_batches(self):
        x = np.random.randn(128, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        b = np.zeros(16, dtype=np.float32)
        y = layernorm(x, w, b)
        assert y.shape == (128, 16)
        assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)

    def test_output_not_equal_to_input(self):
        x = np.random.randn(4, 8).astype(np.float32) * 10
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        y = layernorm(x, w, b)
        assert not np.allclose(x, y)


# ── RMSNorm Advanced ─────────────────────────────────────────────────────────

class TestRmsnormAdvanced:
    def test_all_same_value(self):
        x = np.full((4, 8), 5.0, dtype=np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        # Constant input: rms = 5.0, y = 5.0/5.0 * 1.0 = 1.0
        assert np.allclose(y, 1.0, atol=1e-5)

    def test_unit_weight(self):
        x = np.random.randn(4, 8).astype(np.float32) * 10
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)

    def test_weight_2x(self):
        x = np.random.randn(4, 8).astype(np.float32) * 10
        w = np.full(8, 2.0, dtype=np.float32)
        y = rmsnorm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 2.0, atol=0.2)

    def test_zero_input_nonzero_weight(self):
        x = np.zeros((4, 8), dtype=np.float32)
        w = np.full(8, 3.0, dtype=np.float32)
        y = rmsnorm(x, w)
        # 0 / sqrt(eps) * 3.0 ≈ 0
        assert np.allclose(y, 0.0, atol=1e-3)

    def test_output_not_equal_to_input(self):
        x = np.random.randn(4, 8).astype(np.float32) * 10
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert not np.allclose(x, y)

    def test_many_batches(self):
        x = np.random.randn(128, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (128, 16)

    def test_per_batch_rms(self):
        x = np.random.randn(4, 8).astype(np.float32) * 10
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        for i in range(4):
            rms = np.sqrt(np.mean(y[i] ** 2))
            assert np.allclose(rms, 1.0, atol=0.1)

    def test_5d_input(self):
        x = np.random.randn(2, 3, 4, 5, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        y = rmsnorm(x, w)
        assert y.shape == (2, 3, 4, 5, 8)
