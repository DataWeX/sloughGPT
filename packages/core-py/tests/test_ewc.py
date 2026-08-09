"""Tests for EWC (Elastic Weight Consolidation) — diagonal Fisher, snapshots, penalties.

Covers:
  - EWCParameters defaults
  - TaskSnapshot creation
  - DiagonalFisherEstimator — init, estimate, estimate_from_logits
  - EwcContinualLearner — ewc_loss, multi_task_ewc_loss, forward_and_ewc, forgetting
"""

import numpy as np
import pytest
from domains.training.slonet import Tensor, SloLinear, SloNet
from domains.training.ewc import (
    EWCParameters,
    TaskSnapshot,
    DiagonalFisherEstimator,
    EwcContinualLearner,
    _as_array,
    _scalar,
    _zero_grad,
    _batch_size,
    _unpack_batch,
)


class TestHelpers:
    def test_as_array_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _as_array(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_as_array_ndarray(self):
        arr = np.array([3.0, 4.0])
        result = _as_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_scalar_tensor(self):
        t = Tensor(np.array([5.0]))
        assert _scalar(t) == 5.0

    def test_scalar_float(self):
        assert _scalar(3.14) == 3.14

    def test_zero_grad(self):
        model = SloNet([SloLinear(4, 2)])
        x = Tensor(np.random.randn(1, 4).astype(np.float32))
        model(x)
        for p in model.parameters():
            p.grad = np.ones_like(p.data)
        _zero_grad(model)
        for p in model.parameters():
            assert p.grad is None

    def test_batch_size(self):
        assert _batch_size(np.array([1, 2, 3])) == 3
        assert _batch_size(np.array(5.0)) == 1
        assert _batch_size(np.zeros((4, 8))) == 4

    def test_unpack_batch_list(self):
        inputs, targets = _unpack_batch([np.array([1]), np.array([2])])
        np.testing.assert_array_equal(inputs, [1])
        np.testing.assert_array_equal(targets, [2])

    def test_unpack_batch_single(self):
        inputs, targets = _unpack_batch(np.array([1, 2]))
        np.testing.assert_array_equal(inputs, [1, 2])
        assert targets is None


class TestEWCParameters:
    def test_defaults(self):
        p = EWCParameters()
        assert p.lambda_ewc == 1000.0
        assert p.diagonal_approx is True
        assert p.batch_size == 32
        assert p.num_samples == 100
        assert p.clip_grad_norm == 10.0
        assert p.ema_decay == 0.9

    def test_custom(self):
        p = EWCParameters(lambda_ewc=500.0, num_samples=50)
        assert p.lambda_ewc == 500.0
        assert p.num_samples == 50


class TestTaskSnapshot:
    def test_creation(self):
        snap = TaskSnapshot(
            task_id="t1",
            task_name="Task 1",
            parameters={"w": np.zeros(4)},
            fisher_diagonal={"w": np.ones(4)},
            optimal_loss=1.5,
            num_samples=100,
        )
        assert snap.task_id == "t1"
        assert snap.task_name == "Task 1"
        assert snap.optimal_loss == 1.5
        assert snap.num_samples == 100


class _SimpleModel:
    """Minimal model for testing: single linear layer."""

    def __init__(self, in_dim=4, out_dim=2):
        self.linear = SloLinear(in_dim, out_dim)
        self._training = True

    def parameters(self):
        return self.linear.parameters()

    def named_parameters(self):
        return [("linear.weight", self.linear.weight), ("linear.bias", self.linear.bias)]

    def eval(self):
        self._training = False

    def train(self):
        self._training = True

    def state_dict(self):
        return {name: param.data for name, param in self.named_parameters()}

    def to(self, device):
        return self

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        return self.linear(x)


def _simple_loader(n_batches=5, batch_size=4, in_dim=4):
    for _ in range(n_batches):
        x = np.random.randn(batch_size, in_dim).astype(np.float32)
        y = np.random.randint(0, 2, (batch_size,))
        yield (x, y)


def _simple_loss(outputs, targets):
    if not isinstance(outputs, Tensor):
        outputs = Tensor(np.asarray(outputs, dtype=np.float32))
    if not isinstance(targets, Tensor):
        targets = Tensor(np.asarray(targets, dtype=np.int64))
    # Simple MSE loss for testing
    n = outputs.data.shape[0]
    pred = outputs.data[np.arange(n), targets.data.reshape(-1).astype(int)]
    target_vals = np.ones_like(pred)
    return Tensor(np.array([((pred - target_vals) ** 2).mean()]))


class TestDiagonalFisherEstimator:
    def test_init(self):
        model = _SimpleModel()
        estimator = DiagonalFisherEstimator(model, ema_decay=0.9)
        assert "linear.weight" in estimator.fisher_accum
        assert "linear.bias" in estimator.fisher_accum

    def test_estimate_shapes(self):
        model = _SimpleModel()
        estimator = DiagonalFisherEstimator(model, ema_decay=0.9)
        fisher = estimator.estimate(_simple_loader(3), _simple_loss, num_samples=12)
        assert "linear.weight" in fisher
        assert fisher["linear.weight"].shape == model.linear.weight.data.shape

    def test_estimate_positive(self):
        model = _SimpleModel()
        estimator = DiagonalFisherEstimator(model, ema_decay=0.9)
        fisher = estimator.estimate(_simple_loader(5), _simple_loss, num_samples=20)
        for v in fisher.values():
            assert np.all(v > 0)

    def test_estimate_from_logits(self):
        model = _SimpleModel()
        estimator = DiagonalFisherEstimator(model)
        x = np.random.randn(4, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1])
        fisher = estimator.estimate_from_logits(x, y, num_samples=5)
        assert "linear.weight" in fisher
        assert np.all(fisher["linear.weight"] >= 0)


class TestEwcContinualLearner:
    def test_init(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(lambda_ewc=500.0))
        assert learner.params.lambda_ewc == 500.0
        assert len(learner.task_snapshots) == 0

    def test_save_snapshot(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        snap = learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        assert snap.task_id == "t1"
        assert "t1" in learner.task_snapshots
        assert snap.optimal_loss >= 0

    def test_ewc_loss_no_snapshot(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model)
        loss, stats = learner.ewc_loss("nonexistent")
        assert _scalar(loss) == 0.0
        assert stats["active_tasks"] == 0

    def test_ewc_loss_with_snapshot(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        learner.current_task = "t1"
        loss, stats = learner.ewc_loss()
        assert stats["active_tasks"] == 1
        assert stats["param_count"] > 0
        assert isinstance(stats["ewc_loss"], float)

    def test_multi_task_ewc_loss_empty(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model)
        loss, stats = learner.multi_task_ewc_loss()
        assert stats["active_tasks"] == 0

    def test_multi_task_ewc_loss(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        learner.save_task_snapshot("t2", "Task 2", _simple_loader(3), _simple_loss)
        loss, stats = learner.multi_task_ewc_loss()
        assert stats["active_tasks"] == 2
        assert stats["ewc_loss"] >= 0

    def test_forward_and_ewc(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        learner.current_task = "t1"
        batch = (np.random.randn(4, 4).astype(np.float32), np.array([0, 1, 0, 1]))
        total_loss, stats = learner.forward_and_ewc(batch, _simple_loss)
        assert "task_loss" in stats
        assert "ewc_loss" in stats
        assert stats["active_tasks"] == 1

    def test_forward_and_ewc_no_snapshot(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model)
        batch = (np.random.randn(4, 4).astype(np.float32), np.array([0, 1, 0, 1]))
        total_loss, stats = learner.forward_and_ewc(batch, _simple_loss)
        assert stats["active_tasks"] == 0

    def test_estimate_forgetting_empty(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model)
        assert learner.estimate_forgetting() == {}

    def test_estimate_forgetting(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        forgetting = learner.estimate_forgetting()
        assert "t1" in forgetting
        assert forgetting["t1"] >= 0

    def test_prune_consolidation_empty(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model)
        assert learner.prune_consolidation() == {}

    def test_prune_consolidation(self):
        model = _SimpleModel()
        learner = EwcContinualLearner(model, EWCParameters(num_samples=8))
        learner.save_task_snapshot("t1", "Task 1", _simple_loader(3), _simple_loss)
        important = learner.prune_consolidation(top_k_percent=50.0)
        assert "linear.weight" in important
        assert important["linear.weight"] >= 0
