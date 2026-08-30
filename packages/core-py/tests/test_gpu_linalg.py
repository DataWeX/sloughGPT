"""Tests for domains.training.gpu.accelerator — pure-numpy linear algebra functions."""

import numpy as np
from domains.training.gpu.accelerator import (
    cholesky, solve_triangular, solve_cholesky, dominant_eigen,
)


class TestCholesky:
    def test_identity(self):
        A = np.eye(3)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)

    def test_positive_definite(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)

    def test_lower_triangular(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky(A)
        assert L[0, 1] == 0.0

    def test_1x1(self):
        A = np.array([[9.0]])
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)
        assert L[0, 0] == 3.0

    def test_3x3(self):
        A = np.array([[4, 2, -2], [2, 10, 4], [-2, 4, 10]], dtype=float)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)

    def test_4x4(self):
        n = 4
        A = np.random.randn(n, n)
        A = A @ A.T + n * np.eye(n)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-5)

    def test_diagonal_positive(self):
        d = np.array([1.0, 4.0, 9.0])
        A = np.diag(d)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)
        np.testing.assert_allclose(np.diag(L), np.sqrt(d), atol=1e-10)

    def test_output_is_lower_triangular(self):
        A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
        L = cholesky(A)
        n = A.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert L[i, j] == 0.0

    def test_diagonal_elements_positive(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky(A)
        assert all(np.diag(L) > 0)

    def test_large_matrix(self):
        n = 20
        A = np.random.randn(n, n)
        A = A @ A.T + n * np.eye(n)
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-6)

    def test_scale_invariance(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        scale = 10.0
        L1 = cholesky(A)
        L2 = cholesky(scale * A)
        np.testing.assert_allclose(L2 @ L2.T, scale * A, atol=1e-8)

    def test_reconstruct_symmetric(self):
        A = np.array([[10, 6, 2], [6, 13, 5], [2, 5, 10]], dtype=float)
        L = cholesky(A)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, reconstructed.T, atol=1e-10)

    def test_float32_output(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        L = cholesky(A)
        assert L.dtype == np.float32


class TestSolveTriangular:
    def test_lower(self):
        L = np.array([[2, 0], [1, 3]], dtype=float)
        b = np.array([4, 7], dtype=float)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(L @ x, b, atol=1e-10)

    def test_upper(self):
        U = np.array([[2, 1], [0, 3]], dtype=float)
        b = np.array([5, 6], dtype=float)
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(U @ x, b, atol=1e-10)

    def test_lower_3x3(self):
        L = np.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]], dtype=float)
        b = np.array([1, 8, 32], dtype=float)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(L @ x, b, atol=1e-10)

    def test_upper_3x3(self):
        U = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
        b = np.array([14, 32, 30], dtype=float)
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(U @ x, b, atol=1e-10)

    def test_lower_identity(self):
        L = np.eye(3)
        b = np.array([1, 2, 3], dtype=float)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(x, b, atol=1e-10)

    def test_upper_identity(self):
        U = np.eye(3)
        b = np.array([4, 5, 6], dtype=float)
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(x, b, atol=1e-10)

    def test_lower_diagonal(self):
        L = np.diag([2.0, 3.0, 5.0])
        b = np.array([6, 15, 25], dtype=float)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(x, [3.0, 5.0, 5.0], atol=1e-10)

    def test_upper_diagonal(self):
        U = np.diag([2.0, 3.0, 5.0])
        b = np.array([6, 15, 25], dtype=float)
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(x, [3.0, 5.0, 5.0], atol=1e-10)

    def test_lower_large(self):
        n = 10
        L = np.random.randn(n, n)
        L = np.tril(L) + n * np.eye(n)
        b = np.random.randn(n)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(L @ x, b, atol=1e-6)

    def test_upper_large(self):
        n = 10
        U = np.random.randn(n, n)
        U = np.triu(U) + n * np.eye(n)
        b = np.random.randn(n)
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(U @ x, b, atol=1e-6)

    def test_output_is_float32(self):
        L = np.array([[2, 0], [1, 3]], dtype=np.float32)
        b = np.array([4, 7], dtype=np.float32)
        x = solve_triangular(L, b, lower=True)
        assert x.dtype == np.float32

    def test_lower_1x1(self):
        L = np.array([[5.0]])
        b = np.array([10.0])
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(x, [2.0], atol=1e-10)

    def test_upper_1x1(self):
        U = np.array([[5.0]])
        b = np.array([10.0])
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(x, [2.0], atol=1e-10)

    def test_lower_result_matches_solve(self):
        L = np.array([[2, 0, 0], [1, 3, 0], [2, 1, 4]], dtype=float)
        b = np.array([6, 11, 16], dtype=float)
        x_custom = solve_triangular(L, b, lower=True)
        x_numpy = np.linalg.solve(L, b)
        np.testing.assert_allclose(x_custom, x_numpy, atol=1e-8)

    def test_upper_result_matches_solve(self):
        U = np.array([[2, 1, 3], [0, 4, 1], [0, 0, 5]], dtype=float)
        b = np.array([19, 13, 15], dtype=float)
        x_custom = solve_triangular(U, b, lower=False)
        x_numpy = np.linalg.solve(U, b)
        np.testing.assert_allclose(x_custom, x_numpy, atol=1e-8)

    def test_lower_with_zeros(self):
        L = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        b = np.array([3, 8, 15], dtype=float)
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(x, [3.0, 4.0, 5.0], atol=1e-10)


class TestSolveCholesky:
    def test_basic(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_3x3(self):
        A = np.array([[10, 6, 2], [6, 13, 5], [2, 5, 10]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_1x1(self):
        A = np.array([[4.0]])
        b = np.array([8.0])
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(x, [2.0], atol=1e-10)

    def test_4x4(self):
        A = np.random.randn(4, 4)
        A = A @ A.T + 4 * np.eye(4)
        b = np.random.randn(4)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-5)

    def test_identity(self):
        A = np.eye(3)
        b = np.array([1, 2, 3], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(x, b, atol=1e-10)

    def test_large(self):
        n = 15
        A = np.random.randn(n, n)
        A = A @ A.T + n * np.eye(n)
        b = np.random.randn(n)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-5)

    def test_matches_numpy_solve(self):
        A = np.array([[4, 2, -2], [2, 10, 4], [-2, 4, 10]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        x_custom = solve_cholesky(A, b)
        x_numpy = np.linalg.solve(A, b)
        np.testing.assert_allclose(x_custom, x_numpy, atol=1e-5)

    def test_symmetric_positive_definite(self):
        A = np.array([[6, 2, 1], [2, 5, 3], [1, 3, 4]], dtype=float)
        b = np.array([1, 0, -1], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-5)

    def test_scaled_identity(self):
        scale = 7.0
        A = scale * np.eye(3)
        b = np.array([14, 21, 28], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(x, b / scale, atol=1e-5)

    def test_diagonal_matrix(self):
        A = np.diag([1.0, 4.0, 9.0])
        b = np.array([3, 8, 15], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(x, [3, 2, 15 / 9], atol=1e-10)

    def test_output_is_float32(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        x = solve_cholesky(A, b)
        assert x.dtype == np.float32

    def test_condition_number_2(self):
        A = np.array([[1, 0], [0, 100]], dtype=float)
        b = np.array([1, 100], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(x, [1, 1], atol=1e-8)


class TestDominantEigen:
    def test_identity(self):
        vals, vecs = dominant_eigen(np.eye(3))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-5)

    def test_diagonal(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A)
        np.testing.assert_allclose(vals[0], 5.0, atol=1e-5)

    def test_two_eigen(self):
        A = np.diag([10.0, 5.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=2)
        assert len(vals) == 2
        np.testing.assert_allclose(vals[0], 10.0, atol=1e-5)
        np.testing.assert_allclose(vals[1], 5.0, atol=1e-5)

    def test_symmetric_matrix(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        vals, vecs = dominant_eigen(A)
        assert len(vals) == 1
        assert vals[0] > 0

    def test_eigenvector_normalized(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=1)
        norm = np.linalg.norm(vecs[:, 0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_eigenvector_satisfies_av_equals_lambda_v(self):
        A = np.array([[2, 1], [1, 3]], dtype=float)
        vals, vecs = dominant_eigen(A, n_eigen=1)
        v = vecs[:, 0]
        Av = A @ v
        lambda_v = vals[0] * v
        np.testing.assert_allclose(Av, lambda_v, atol=1e-5)

    def test_all_eigen(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=3)
        assert len(vals) == 3
        np.testing.assert_allclose(sorted(vals, reverse=True), [5, 3, 1], atol=1e-5)

    def test_output_shapes(self):
        n = 4
        A = np.diag([4.0, 3.0, 2.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=2)
        assert vals.shape == (2,)
        assert vecs.shape == (4, 2)

    def test_random_symmetric(self):
        np.random.seed(42)
        n = 5
        A = np.random.randn(n, n)
        A = (A + A.T) / 2
        vals, vecs = dominant_eigen(A, n_eigen=1)
        assert len(vals) == 1
        v = vecs[:, 0]
        Av = A @ v
        np.testing.assert_allclose(Av, vals[0] * v, atol=1e-4)

    def test_idempotent_on_identity(self):
        A = np.eye(3)
        vals1, _ = dominant_eigen(A)
        vals2, _ = dominant_eigen(A)
        np.testing.assert_allclose(vals1, vals2, atol=1e-10)

    def test_large_diagonal(self):
        n = 10
        d = np.arange(n, 0, -1, dtype=float)
        A = np.diag(d)
        vals, vecs = dominant_eigen(A, n_eigen=3)
        np.testing.assert_allclose(vals, [10, 9, 8], atol=1e-3)

    def test_eigenvectors_orthogonal(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=3)
        for i in range(3):
            for j in range(i + 1, 3):
                dot = abs(np.dot(vecs[:, i], vecs[:, j]))
                assert dot < 1e-5

    def test_max_iter_reached(self):
        A = np.eye(3) * 1000
        vals, vecs = dominant_eigen(A, max_iter=1)
        assert len(vals) == 1

    def test_negative_eigenvalues(self):
        A = np.diag([-5.0, -3.0, -1.0])
        vals, vecs = dominant_eigen(A, n_eigen=1)
        np.testing.assert_allclose(vals[0], -5.0, atol=1e-5)

    def test_mixed_sign_eigenvalues(self):
        A = np.diag([5.0, -3.0, 1.0])
        vals, vecs = dominant_eigen(A, n_eigen=1)
        np.testing.assert_allclose(vals[0], 5.0, atol=1e-5)

    def test_tolerance_convergence(self):
        A = np.diag([10.0, 5.0, 1.0])
        vals, _ = dominant_eigen(A, tol=1e-12)
        np.testing.assert_allclose(vals[0], 10.0, atol=1e-10)

    def test_n_eigen_one_default(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A)
        assert len(vals) == 1
        assert vecs.shape == (3, 1)
