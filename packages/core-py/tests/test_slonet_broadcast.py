"""Tests for broadcast gradient correctness in backward pass.

Verifies _mul, _add, and _matmul backward handle mismatched ndim
broadcasts without shape errors or incorrect gradients.
"""

import numpy as np
import pytest
from domains.training.slonet import Tensor, cross_entropy


class TestMulBroadcastBackward:
    """_mul backward must handle extra leading broadcast dimensions."""

    def test_mul_1d_times_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)
        assert np.allclose(b.grad.data, 1.0)

    def test_mul_2d_times_1d(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 1.0)
        assert np.allclose(b.grad.data, 3.0)

    def test_mul_scalar_times_nd(self):
        a = Tensor(np.array(2.0), requires_grad=True)
        b = Tensor(np.ones((2, 3)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 6.0)
        assert np.allclose(b.grad.data, 2.0)

    def test_mul_3d_times_1d(self):
        a = Tensor(np.ones((2, 3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(b.grad.data, 6.0)

    def test_mul_leading_extra_dims(self):
        a = Tensor(np.ones((2, 1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_mul_leading_one_dim_broadcast(self):
        a = Tensor(np.ones((1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a * b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_add_leading_one_dim_broadcast(self):
        a = Tensor(np.ones((1, 4)), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_mul_numeric_correctness(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
        out = a * b
        out.backward()
        assert np.allclose(a.grad.data, [4.0, 5.0])
        assert np.allclose(b.grad.data, [2.0, 3.0])

    def test_mul_chain_through_cross_entropy(self):
        a = Tensor(np.ones(3), requires_grad=True)
        b = Tensor(np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]]), requires_grad=True)
        logits = a * b
        target = Tensor(np.array([0, 1]))
        loss = cross_entropy(logits, target)
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


class TestAddBroadcastBackward:
    """_add backward with extra leading broadcast dims (regression check)."""

    def test_add_1d_plus_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((3, 4)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
        assert np.allclose(a.grad.data, 3.0)

    def test_add_scalar_plus_nd(self):
        a = Tensor(np.array(5.0), requires_grad=True)
        b = Tensor(np.ones((2, 3)), requires_grad=True)
        out = a + b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


class TestMatMulBroadcastBackward:
    """_matmul backward with mixed dimension inputs."""

    def test_matmul_1d_2d(self):
        a = Tensor(np.ones(4), requires_grad=True)
        b = Tensor(np.ones((4, 3)), requires_grad=True)
        out = a @ b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matmul_2d_1d(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones(4), requires_grad=True)
        out = a @ b
        out.backward()
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape
