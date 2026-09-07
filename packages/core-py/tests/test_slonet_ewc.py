"""Tests for training.ewc — Elastic Weight Consolidation."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.ewc import (
    EWCParameters,
    TaskSnapshot,
    DiagonalFisherEstimator,
    _as_array,
    _scalar,
    _zero_grad,
    _batch_size,
    _unpack_batch,
)
from domains.training.slonet import Tensor, SloLinear, SloNet


# ── _as_array / _scalar / _batch_size / _unpack_batch helpers ──────────────


class TestHelperFunctions:

    def test_as_array_numpy(self):
        arr = np.array([1.0, 2.0])
        result = _as_array(arr)
        assert isinstance(result, np.ndarray)

    def test_as_array_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _as_array(t)
        assert isinstance(result, np.ndarray)

    def test_scalar_tensor(self):
        t = Tensor(np.array([5.0]))
        assert _scalar(t) == 5.0

    def test_scalar_numpy(self):
        arr = np.array([3.14])
        assert _scalar(arr) == pytest.approx(3.14)

    def test_batch_size(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert _batch_size(arr) == 2

    def test_batch_size_scalar(self):
        assert _batch_size(np.array(1.0)) == 1

    def test_unpack_batch_tuple(self):
        batch = (np.array([1]), np.array([2]))
        inputs, targets = _unpack_batch(batch)
        assert np.allclose(inputs, [1])
        assert np.allclose(targets, [2])

    def test_unpack_batch_no_targets(self):
        batch = np.array([1])
        inputs, targets = _unpack_batch(batch)
        assert np.allclose(inputs, [1])
        assert targets is None


# ── _zero_grad ──────────────────────────────────────────────────────────────


class TestZeroGrad:

    def test_zero_grad(self):
        net = SloNet(layers=[SloLinear(10, 5)])
        _zero_grad(net)
        for p in net.parameters():
            assert p.grad is None


# ── EWCParameters ──────────────────────────────────────────────────────────


class TestEWCParameters:

    def test_default(self):
        params = EWCParameters()
        assert params.lambda_ewc == 1000.0
        assert params.diagonal_approx is True
        assert params.num_samples == 100

    def test_custom(self):
        params = EWCParameters(lambda_ewc=500.0, num_samples=50)
        assert params.lambda_ewc == 500.0
        assert params.num_samples == 50


# ── TaskSnapshot ───────────────────────────────────────────────────────────


class TestTaskSnapshot:

    def test_init(self):
        snap = TaskSnapshot(
            task_id="t1",
            task_name="test",
            parameters={"w": np.array([1.0])},
            fisher_diagonal={"w": np.array([0.1])},
            optimal_loss=0.5,
            num_samples=100,
        )
        assert snap.task_id == "t1"
        assert snap.optimal_loss == 0.5


# ── DiagonalFisherEstimator ────────────────────────────────────────────────


class TestDiagonalFisherEstimator:

    def test_init(self):
        class MockModel:
            def parameters(self):
                return [Tensor(np.array([1.0, 2.0]))]
            def named_parameters(self):
                return [("w", Tensor(np.array([1.0, 2.0])))]
        estimator = DiagonalFisherEstimator(MockModel())
        assert estimator is not None
