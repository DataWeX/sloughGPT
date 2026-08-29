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


class TestSolveCholesky:
    def test_basic(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)


class TestDominantEigen:
    def test_identity(self):
        vals, vecs = dominant_eigen(np.eye(3))
        assert len(vals) == 1
        np.testing.assert_allclose(vals[0], 1.0, atol=1e-5)

    def test_diagonal(self):
        A = np.diag([5.0, 3.0, 1.0])
        vals, vecs = dominant_eigen(A)
        np.testing.assert_allclose(vals[0], 5.0, atol=1e-5)
