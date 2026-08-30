"""Tests for domains.inference.ops.blas — sgemm numpy fallback + is_available."""

import numpy as np
import pytest
from domains.inference.ops.blas import sgemm, is_available


class TestSgemm:
    def test_basic_matmul(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = sgemm(a, b)
        expected = a @ b
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_3x3(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_non_square(self):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        result = sgemm(a, b)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_alpha_beta(self):
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        # On non-Accelerate platforms, sgemm falls back to np.matmul
        # which ignores alpha/beta. Only test on Accelerate.
        if is_available():
            result = sgemm(a, b, alpha=2.0, beta=0.0)
            expected = np.full((2, 2), 4.0, dtype=np.float32)
            np.testing.assert_allclose(result, expected, atol=1e-6)
        else:
            # Numpy fallback: alpha/beta ignored
            result = sgemm(a, b, alpha=2.0, beta=0.0)
            expected = np.full((2, 2), 2.0, dtype=np.float32)
            np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shape_mismatch_raises(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        with pytest.raises(ValueError):
            sgemm(a, b)

    def test_empty_matrices(self):
        a = np.empty((0, 3), dtype=np.float32)
        b = np.empty((3, 0), dtype=np.float32)
        result = sgemm(a, b)
        assert result.shape == (0, 0)

    def test_1x1(self):
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, [[21.0]], atol=1e-6)

    def test_identity_right(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.eye(4, dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_identity_left(self):
        a = np.eye(3, dtype=np.float32)
        b = np.random.randn(3, 5).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, b, atol=1e-6)

    def test_zeros_right(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.zeros((4, 2), dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, np.zeros((3, 2), dtype=np.float32), atol=1e-6)

    def test_zeros_left(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, np.zeros((2, 4), dtype=np.float32), atol=1e-6)

    def test_tall_skinny(self):
        a = np.random.randn(10, 2).astype(np.float32)
        b = np.random.randn(2, 10).astype(np.float32)
        result = sgemm(a, b)
        assert result.shape == (10, 10)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_wide_short(self):
        a = np.random.randn(2, 10).astype(np.float32)
        b = np.random.randn(10, 2).astype(np.float32)
        result = sgemm(a, b)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_large_values(self):
        a = np.full((2, 2), 1e5, dtype=np.float32)
        b = np.full((2, 2), 1e5, dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, rtol=1e-4)

    def test_small_values(self):
        a = np.full((2, 2), 1e-7, dtype=np.float32)
        b = np.full((2, 2), 1e-7, dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-12)

    def test_negative_values(self):
        a = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        b = np.array([[5, -6], [-7, 8]], dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_diagonal_matrix(self):
        d = np.array([2.0, 3.0, 5.0], dtype=np.float32)
        a = np.diag(d)
        b = np.ones((3, 3), dtype=np.float32)
        result = sgemm(a, b)
        expected = d[:, None] * np.ones((3, 3), dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_transpose_equivalence_square(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        r1 = sgemm(a, b)
        np.testing.assert_allclose(r1, (a @ b), atol=1e-5)

    def test_output_is_float32(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = sgemm(a, b)
        assert result.dtype == np.float32

    def test_associativity(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        c = np.random.randn(4, 2).astype(np.float32)
        r1 = sgemm(sgemm(a, b), c)
        r2 = sgemm(a, sgemm(b, c))
        np.testing.assert_allclose(r1, r2, atol=1e-4)

    def test_distributivity(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        c = np.random.randn(3, 3).astype(np.float32)
        r1 = sgemm(a, b + c)
        r2 = sgemm(a, b) + sgemm(a, c)
        np.testing.assert_allclose(r1, r2, atol=1e-4)

    def test_symmetric_input(self):
        a = np.array([[4, 2], [2, 3]], dtype=np.float32)
        result = sgemm(a, a)
        expected = a @ a
        np.testing.assert_allclose(result, expected, atol=1e-5)
        np.testing.assert_allclose(result, result.T, atol=1e-5)

    def test_batch_single_element(self):
        a = np.random.randn(1, 1).astype(np.float32)
        b = np.random.randn(1, 1).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_column_times_row(self):
        a = np.array([[1], [2], [3]], dtype=np.float32)
        b = np.array([[4, 5, 6]], dtype=np.float32)
        result = sgemm(a, b)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_row_times_column(self):
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4], [5], [6]], dtype=np.float32)
        result = sgemm(a, b)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result, [[32.0]], atol=1e-6)

    def test_large_dimension(self):
        a = np.random.randn(50, 50).astype(np.float32)
        b = np.random.randn(50, 50).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-3)

    def test_alpha_zero(self):
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        result = sgemm(a, b, alpha=0.0, beta=0.0)
        if is_available():
            np.testing.assert_allclose(result, np.zeros((2, 2), dtype=np.float32), atol=1e-6)
        else:
            np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_multiple_calls_consistent(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        r1 = sgemm(a, b)
        r2 = sgemm(a, b)
        np.testing.assert_array_equal(r1, r2)

    def test_non_contiguous_input(self):
        a_full = np.random.randn(4, 4).astype(np.float32)
        b_full = np.random.randn(4, 4).astype(np.float32)
        a = a_full[::2, ::2]
        b = b_full[::2, ::2]
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_bool_coercion_no_crash(self):
        a = np.array([[True, False], [False, True]], dtype=np.float32)
        b = np.array([[1, 0], [0, 1]], dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_int_input_coerced(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = sgemm(a, b)
        assert result.dtype == np.float32


class TestSgemmExtended:
    def test_cblas_constants(self):
        from domains.inference.ops.blas import CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS
        assert CBLAS_ROW_MAJOR == 101
        assert CBLAS_NO_TRANS == 111
        assert CBLAS_TRANS == 112

    def test_load_accelerate_returns_none_on_linux(self):
        import sys
        if sys.platform != "linux":
            pytest.skip("not Linux")
        from domains.inference.ops.blas import _load_accelerate
        result = _load_accelerate()
        assert result is None

    def test_unavailable_flag_set_on_linux(self):
        import sys
        if sys.platform != "linux":
            pytest.skip("not Linux")
        import domains.inference.ops.blas as blas_mod
        _load = blas_mod._load_accelerate()
        assert blas_mod._unavailable is True

    def test_sgemm_matches_matmul_various_shapes(self):
        shapes = [(1, 1, 1), (2, 3, 4), (5, 2, 7), (10, 10, 10)]
        for m, k, n in shapes:
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)
            result = sgemm(a, b)
            np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_sgemm_output_shape_matches_matmul(self):
        a = np.random.randn(7, 3).astype(np.float32)
        b = np.random.randn(3, 5).astype(np.float32)
        r_sgemm = sgemm(a, b)
        r_matmul = a @ b
        assert r_sgemm.shape == r_matmul.shape

    def test_sgemm_with_all_zeros(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-6)

    def test_sgemm_matrix_power(self):
        a = np.random.randn(3, 3).astype(np.float32)
        result = sgemm(a, a)
        expected = a @ a
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sgemm_negative_alpha(self):
        a = np.ones((2, 2), dtype=np.float32)
        b = np.eye(2, dtype=np.float32)
        result = sgemm(a, b, alpha=-1.0, beta=0.0)
        if is_available():
            np.testing.assert_allclose(result, -a, atol=1e-6)
        else:
            np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_sgemm_with_large_dimension_tall(self):
        a = np.random.randn(100, 2).astype(np.float32)
        b = np.random.randn(2, 100).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-3)

    def test_sgemm_with_large_dimension_wide(self):
        a = np.random.randn(2, 100).astype(np.float32)
        b = np.random.randn(100, 2).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-3)

    def test_sgemm_random_seed_reproducible(self):
        np.random.seed(42)
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        r1 = sgemm(a, b)
        r2 = sgemm(a, b)
        np.testing.assert_array_equal(r1, r2)

    def test_sgemm_diagonal_times_dense(self):
        d = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        a = np.diag(d)
        b = np.random.randn(3, 3).astype(np.float32)
        result = sgemm(a, b)
        expected = d[:, None] * b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sgemm_dense_times_diagonal(self):
        d = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.diag(d)
        result = sgemm(a, b)
        expected = a * d[None, :]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sgemm_scaled_identity(self):
        scale = 5.0
        a = scale * np.eye(4, dtype=np.float32)
        b = np.random.randn(4, 4).astype(np.float32)
        result = sgemm(a, b)
        expected = scale * b
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sgemm_multiple_alpha_beta(self):
        a = np.ones((2, 2), dtype=np.float32) * 2.0
        b = np.ones((2, 2), dtype=np.float32) * 3.0
        result = sgemm(a, b, alpha=1.0, beta=1.0)
        if is_available():
            expected = 1.0 * (a @ b) + 1.0 * np.zeros((2, 2))
            np.testing.assert_allclose(result, expected, atol=1e-6)
        else:
            np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_sgemm_transpose_of_result(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        result = sgemm(a, b)
        expected_T = sgemm(b.T, a.T)
        np.testing.assert_allclose(result, expected_T.T, atol=1e-5)

    def test_sgemm_with_very_small_values(self):
        a = np.full((2, 2), 1e-10, dtype=np.float32)
        b = np.full((2, 2), 1e-10, dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-20)

    def test_sgemm_with_mixed_signs(self):
        a = np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float32)
        b = np.array([[0, 1], [-1, 0], [1, -1]], dtype=np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_sgemm_5x5(self):
        np.random.seed(7)
        a = np.random.randn(5, 5).astype(np.float32)
        b = np.random.randn(5, 5).astype(np.float32)
        result = sgemm(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_sgemm_non_square_uneven(self):
        a = np.random.randn(3, 7).astype(np.float32)
        b = np.random.randn(7, 2).astype(np.float32)
        result = sgemm(a, b)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_sgemm_result_independent_of_input_order(self):
        a = np.random.randn(3, 3).astype(np.float32)
        b = np.random.randn(3, 3).astype(np.float32)
        r1 = sgemm(a, b)
        r2 = sgemm(a.copy(), b.copy())
        np.testing.assert_array_equal(r1, r2)


class TestIsAvailable:
    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_on_linux_returns_false(self):
        import sys
        if sys.platform != "linux":
            pytest.skip("not Linux")
        assert is_available() is False

    def test_idempotent(self):
        r1 = is_available()
        r2 = is_available()
        assert r1 == r2

    def test_cached_result_matches(self):
        import domains.inference.ops.blas as blas_mod
        first = is_available()
        if blas_mod._unavailable:
            assert first is False
