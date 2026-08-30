"""Tests for domains.inference.ops.matmul — matmul dispatch."""

import numpy as np
import pytest
from domains.inference.ops.matmul import matmul


class TestMatmul:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        assert np.allclose(result, a @ b)

    def test_float64(self):
        a = np.array([[1, 2]], dtype=np.float64)
        b = np.array([[3], [4]], dtype=np.float64)
        result = matmul(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(11.0)

    def test_rectangular(self):
        a = np.random.randn(2, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (2, 3)

    def test_1d_vector(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == ()
        assert result == pytest.approx(32.0)

    def test_large_matrix(self):
        a = np.random.randn(100, 64).astype(np.float32)
        b = np.random.randn(64, 100).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (100, 100)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)

    def test_identity_matrix(self):
        identity = np.eye(4, dtype=np.float32)
        a = np.random.randn(4, 4).astype(np.float32)
        result = matmul(a, identity)
        assert np.allclose(result, a, atol=1e-6)

    def test_identity_matrix_right(self):
        identity = np.eye(4, dtype=np.float32)
        a = np.random.randn(4, 4).astype(np.float32)
        result = matmul(identity, a)
        assert np.allclose(result, a, atol=1e-6)

    def test_zero_matrix(self):
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        result = matmul(a, b)
        assert np.allclose(result, np.zeros((3, 5)), atol=1e-6)

    def test_zero_matrix_right(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.zeros((4, 5), dtype=np.float32)
        result = matmul(a, b)
        assert np.allclose(result, np.zeros((3, 5)), atol=1e-6)

    def test_negative_values(self):
        a = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        b = np.array([[-5, 6], [7, -8]], dtype=np.float32)
        result = matmul(a, b)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)

    def test_transposed_view(self):
        a = np.random.randn(3, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        result = matmul(a, b)
        result_t = matmul(a, b.T.T)
        assert np.allclose(result, result_t, atol=1e-5)

    def test_symmetric_matrix(self):
        a = np.array([[2, 1], [1, 3]], dtype=np.float32)
        result = matmul(a, a)
        expected = a @ a
        assert np.allclose(result, expected, atol=1e-5)

    def test_diagonal_matrix(self):
        diag = np.diag([1, 2, 3]).astype(np.float32)
        a = np.random.randn(3, 3).astype(np.float32)
        result = matmul(diag, a)
        expected = diag @ a
        assert np.allclose(result, expected, atol=1e-5)

    def test_single_element(self):
        a = np.array([[5]], dtype=np.float32)
        b = np.array([[3]], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(15.0)

    def test_row_times_column(self):
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4], [5], [6]], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(32.0)

    def test_column_times_row(self):
        a = np.array([[1], [2], [3]], dtype=np.float32)
        b = np.array([[4, 5, 6]], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == (3, 3)
        expected = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)

    def test_very_small_values(self):
        a = np.array([[1e-10, 2e-10], [3e-10, 4e-10]], dtype=np.float32)
        b = np.array([[5e-10, 6e-10], [7e-10, 8e-10]], dtype=np.float32)
        result = matmul(a, b)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-25)

    def test_very_large_values(self):
        a = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=np.float32)
        b = np.array([[5e10, 6e10], [7e10, 8e10]], dtype=np.float32)
        result = matmul(a, b)
        expected = a @ b
        assert np.allclose(result, expected, rtol=1e-3)

    def test_mixed_positive_negative(self):
        a = np.array([[-1, 0, 1], [0, 1, -1]], dtype=np.float32)
        b = np.array([[1, -1], [0, 0], [-1, 1]], dtype=np.float32)
        result = matmul(a, b)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)

    def test_ones_matrix(self):
        a = np.ones((3, 3), dtype=np.float32)
        b = np.ones((3, 3), dtype=np.float32)
        result = matmul(a, b)
        expected = np.full((3, 3), 3.0, dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)

    def test_scale_identity(self):
        scale = 2.5
        a = np.random.randn(4, 4).astype(np.float32)
        b = np.eye(4, dtype=np.float32) * scale
        result = matmul(a, b)
        expected = a * scale
        assert np.allclose(result, expected, atol=1e-5)

    def test_associativity(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = np.random.randn(5, 2).astype(np.float32)
        ab_c = matmul(matmul(a, b), c)
        a_bc = matmul(a, matmul(b, c))
        assert np.allclose(ab_c, a_bc, atol=1e-4)

    def test_distributivity(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        c = np.random.randn(4, 5).astype(np.float32)
        result_left = matmul(a, b + c)
        result_right = matmul(a, b) + matmul(a, c)
        assert np.allclose(result_left, result_right, atol=1e-4)

    def test_float32_dtype_preserved(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        assert result.dtype == np.float32

    def test_float64_dtype_preserved(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = matmul(a, b)
        assert result.dtype == np.float64

    def test_integer_input(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([[5, 6], [7, 8]], dtype=np.int32)
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.int64)
        assert np.allclose(result, expected)

    def test_non_contiguous_array(self):
        a_full = np.random.randn(4, 6).astype(np.float32)
        a = a_full[:, ::2]
        b = np.random.randn(3, 4).astype(np.float32)
        result = matmul(a, b)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-5)

    def test_T_attribute(self):
        a = np.random.randn(3, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        result = matmul(a, b)
        result_t_attr = matmul(a, b)
        assert np.allclose(result, result_t_attr, atol=1e-5)

    def test_deep_copy_result(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        result_copy = result.copy()
        assert np.allclose(result, result_copy)

    def test_large_batch_dimension(self):
        a = np.random.randn(1, 256).astype(np.float32)
        b = np.random.randn(256, 1).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (1, 1)
        expected = a @ b
        assert np.allclose(result, expected, atol=1e-4)

    def test_skinny_tall_matrix(self):
        a = np.random.randn(100, 2).astype(np.float32)
        b = np.random.randn(2, 100).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (100, 100)
        assert np.allclose(result, a @ b, atol=1e-4)

    def test_skinny_wide_matrix(self):
        a = np.random.randn(2, 100).astype(np.float32)
        b = np.random.randn(100, 2).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (2, 2)
        assert np.allclose(result, a @ b, atol=1e-4)

    def test_repeated_multiplication(self):
        a = np.array([[1, 1], [0, 1]], dtype=np.float32)
        result = a.copy()
        for _ in range(10):
            result = matmul(result, a)
        expected = np.linalg.matrix_power(a, 11)
        assert np.allclose(result, expected, atol=1e-4)

    def test_permutation_matrix(self):
        perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        v = np.array([1, 2, 3], dtype=np.float32)
        result = matmul(perm, v)
        assert np.allclose(result, [2, 1, 3], atol=1e-5)

    def test_rotation_matrix(self):
        theta = np.pi / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        v = np.array([1, 0], dtype=np.float32)
        result = matmul(rot, v)
        assert np.allclose(result, [np.cos(theta), np.sin(theta)], atol=1e-5)

    def test_projection_matrix(self):
        v = np.array([1, 1], dtype=np.float32)
        proj = np.outer(v, v) / np.dot(v, v)
        a = np.array([3, 4], dtype=np.float32)
        result = matmul(proj.astype(np.float32), a)
        expected = v * np.dot(v, a) / np.dot(v, v)
        assert np.allclose(result, expected, atol=1e-5)

    def test_orthogonal_matrix(self):
        q = np.array([[0, -1], [1, 0]], dtype=np.float32)
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = matmul(q, a)
        expected = q @ a
        assert np.allclose(result, expected, atol=1e-5)

    def test_upper_triangular(self):
        u = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float32)
        a = np.random.randn(3, 3).astype(np.float32)
        result = matmul(u, a)
        assert np.allclose(result, u @ a, atol=1e-5)

    def test_lower_triangular(self):
        l = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]], dtype=np.float32)
        a = np.random.randn(3, 3).astype(np.float32)
        result = matmul(l, a)
        assert np.allclose(result, l @ a, atol=1e-5)

    def test_sparse_like_structure(self):
        a = np.zeros((5, 5), dtype=np.float32)
        a[0, 0] = 1.0
        a[2, 3] = 2.0
        b = np.eye(5, dtype=np.float32)
        result = matmul(a, b)
        assert np.allclose(result, a, atol=1e-6)

    def test_power_of_two_sizes(self):
        for size in [1, 2, 4, 8, 16, 32, 64]:
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            result = matmul(a, b)
            assert result.shape == (size, size)
            assert np.allclose(result, a @ b, atol=1e-4)

    def test_asymmetric_sizes(self):
        pairs = [(1, 7), (7, 1), (3, 13), (13, 3), (2, 97), (97, 2)]
        for m, n in pairs:
            a = np.random.randn(m, 10).astype(np.float32)
            b = np.random.randn(10, n).astype(np.float32)
            result = matmul(a, b)
            assert result.shape == (m, n)
            assert np.allclose(result, a @ b, atol=1e-4)

    def test_chain_of_three(self):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        c = np.random.randn(4, 5).astype(np.float32)
        result = matmul(matmul(a, b), c)
        expected = a @ b @ c
        assert np.allclose(result, expected, atol=1e-4)

    def test_add_column_like(self):
        a = np.array([[1], [2], [3]], dtype=np.float32)
        b = np.array([[10, 20]], dtype=np.float32)
        result = matmul(a, b)
        assert result.shape == (3, 2)
        expected = np.array([[10, 20], [20, 40], [30, 60]], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)

    def test_bilinear_form(self):
        a = np.array([[1, 2]], dtype=np.float32)
        m = np.array([[1, 0], [0, 2]], dtype=np.float32)
        b = np.array([[3], [4]], dtype=np.float32)
        result = matmul(matmul(a, m), b)
        expected = a @ m @ b
        assert result.shape == (1, 1)
        assert np.allclose(result, expected, atol=1e-5)

    def test_consistency_across_dtypes(self):
        a_f32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b_f32 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        a_f64 = a_f32.astype(np.float64)
        b_f64 = b_f32.astype(np.float64)
        r_f32 = matmul(a_f32, b_f32)
        r_f64 = matmul(a_f64, b_f64)
        assert np.allclose(r_f32, r_f64, atol=1e-5)

    def test_stress_random(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            m = rng.randint(1, 50)
            k = rng.randint(1, 50)
            n = rng.randint(1, 50)
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            result = matmul(a, b)
            assert result.shape == (m, n)
            assert np.allclose(result, a @ b, atol=1e-3)

    def test_eye_equals_self(self):
        a = np.random.randn(5, 5).astype(np.float32)
        result = matmul(np.eye(5, dtype=np.float32), a)
        assert np.allclose(result, a, atol=1e-6)

    def test_dot_product_equivalence(self):
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        result = matmul(a, b)
        expected = np.dot(a, b)
        assert np.allclose(result, expected, atol=1e-5)

    def test_outer_product(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5], dtype=np.float32)
        result = matmul(a.reshape(-1, 1), b.reshape(1, -1))
        expected = np.outer(a, b)
        assert np.allclose(result, expected, atol=1e-5)

    def test_very_tall_thin(self):
        a = np.random.randn(200, 1).astype(np.float32)
        b = np.random.randn(1, 200).astype(np.float32)
        result = matmul(a, b)
        assert result.shape == (200, 200)
        assert np.allclose(result, a @ b, atol=1e-4)

    def test_multiple_random_seeds(self):
        for seed in range(10):
            rng = np.random.RandomState(seed)
            m, k, n = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
            a = rng.randn(m, k).astype(np.float32)
            b = rng.randn(k, n).astype(np.float32)
            assert np.allclose(matmul(a, b), a @ b, atol=1e-4)

    def test_transpose_distributivity(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        result_left = matmul(a, b).T
        result_right = matmul(b.T, a.T)
        assert np.allclose(result_left, result_right, atol=1e-5)
