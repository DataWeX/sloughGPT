"""
Tests for Elastic Weight Consolidation (EWC) — numpy/SloNet
"""

import numpy as np
import pytest
from domains.training.slonet import SloLinear, Tensor
from domains.training.ewc import (
    EWCParameters,
    DiagonalFisherEstimator,
    EwcContinualLearner,
)


class LinearModel:
    """SloNet linear model for testing."""

    def __init__(self):
        self.linear = SloLinear(10, 10)
        self._named = [
            ("linear.weight", self.linear.weight),
            ("linear.bias", self.linear.bias),
        ]

    def named_parameters(self):
        return list(self._named)

    def parameters(self):
        return [p for _, p in self._named]

    def state_dict(self):
        return {name: np.asarray(p).copy() for name, p in self._named}

    def forward(self, x):
        return self.linear(x)

    def __call__(self, x):
        return self.forward(x)

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def to(self, device=None, **kwargs):
        return self


def mse_loss(pred, target):
    """MSE loss as a differentiable SloNet Tensor."""
    target_t = Tensor(np.asarray(target, dtype=np.float32))
    return ((pred - target_t) ** 2).mean()


def random_batches(n=3, batch=4):
    return [
        (np.random.randn(batch, 10).astype(np.float32),
         np.random.randn(batch, 10).astype(np.float32))
        for _ in range(n)
    ]


class TestDiagonalFisherEstimator:
    """Tests for Fisher estimator."""

    def test_init(self):
        """Test Fisher estimator initialization."""
        model = LinearModel()
        estimator = DiagonalFisherEstimator(model)

        assert len(estimator.fisher_accum) == 2

    def test_estimate(self):
        """Test Fisher estimation."""
        model = LinearModel()
        estimator = DiagonalFisherEstimator(model)

        data = [np.random.randn(4, 10).astype(np.float32) for _ in range(5)]

        fisher = estimator.estimate(data, mse_loss, num_samples=10)

        assert len(fisher) > 0
        for name, f in fisher.items():
            assert f.shape == model.state_dict()[name].shape
            assert (f >= 0).all()  # Fisher should be non-negative

    def test_estimate_with_targets(self):
        """Test Fisher estimation with paired batches."""
        model = LinearModel()
        estimator = DiagonalFisherEstimator(model)

        fisher = estimator.estimate(random_batches(3), mse_loss, num_samples=10)

        assert len(fisher) > 0
        for f in fisher.values():
            assert (f > 0).any()  # gradients produced non-zero Fisher

    def test_estimate_from_logits(self):
        """Test Fisher estimation from logits."""
        model = LinearModel()
        estimator = DiagonalFisherEstimator(model)

        inputs = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,)).astype(np.int64)

        fisher = estimator.estimate_from_logits(inputs, targets, num_samples=5)

        assert len(fisher) > 0
        for f in fisher.values():
            assert (f >= 0).all()


class TestEwcContinualLearner:
    """Tests for EWC continual learner."""

    def test_init(self):
        """Test EWC learner initialization."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        assert learner.model is not None
        assert learner.params.lambda_ewc == 1000.0
        assert len(learner.fisher_estimator.fisher_accum) == 2

    def test_save_snapshot(self):
        """Test task snapshot saving."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        data = random_batches(3)
        snapshot = learner.save_task_snapshot("task1", "Addition", data, mse_loss)

        assert snapshot.task_id == "task1"
        assert snapshot.task_name == "Addition"
        assert len(snapshot.parameters) > 0
        assert len(snapshot.fisher_diagonal) > 0
        assert "task1" in learner.task_snapshots

    def test_ewc_loss(self):
        """Test EWC loss calculation."""
        model = LinearModel()
        params = EWCParameters(lambda_ewc=1000.0)
        learner = EwcContinualLearner(model, params=params)

        data = random_batches(3)
        learner.save_task_snapshot("task1", "Task1", data, mse_loss)

        # Params unchanged — penalty should be ~0
        ewc_loss, stats = learner.ewc_loss("task1")

        assert isinstance(ewc_loss.item(), float)
        assert abs(ewc_loss.item()) < 1e-4
        assert stats["active_tasks"] == 1

    def test_ewc_loss_penalizes_drift(self):
        """EWC loss grows after parameters move away from the snapshot."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        data = random_batches(3)
        learner.save_task_snapshot("task1", "Task1", data, mse_loss)
        before, _ = learner.ewc_loss("task1")

        # Perturb the weights away from the snapshot
        for name, param in model.named_parameters():
            param.data = param.data + np.random.randn(*param.data.shape) * 0.5

        after, stats = learner.ewc_loss("task1")

        assert after.item() > before.item()
        assert stats["active_tasks"] == 1

    def test_forward_and_ewc(self):
        """Test forward pass with EWC loss."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        data = random_batches(3)
        learner.save_task_snapshot("task1", "Task1", data, mse_loss)

        batch = (np.random.randn(4, 10).astype(np.float32),
                 np.random.randn(4, 10).astype(np.float32))
        total_loss, stats = learner.forward_and_ewc(batch, mse_loss, "task1")

        assert "total_loss" in stats
        assert "task_loss" in stats
        assert "ewc_loss" in stats
        # Differentiable — backward flows to model params
        total_loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_multi_task_ewc_loss(self):
        """Test combined EWC loss across tasks."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        learner.save_task_snapshot("task1", "T1", random_batches(2), mse_loss)
        learner.save_task_snapshot("task2", "T2", random_batches(2), mse_loss)

        loss, stats = learner.multi_task_ewc_loss()

        assert isinstance(loss.item(), float)
        assert stats["active_tasks"] == 2

    def test_prune_consolidation(self):
        """Test identifying important parameters."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        data = random_batches(3)
        learner.save_task_snapshot("task1", "Task1", data, mse_loss)

        important = learner.prune_consolidation(top_k_percent=20)
        assert isinstance(important, dict)

    def test_estimate_forgetting(self):
        """Test forgetting estimation."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        data = random_batches(3)
        learner.save_task_snapshot("task1", "Task1", data, mse_loss)

        # Perturb weights -> forgetting estimate should be positive
        for name, param in model.named_parameters():
            param.data = param.data + np.random.randn(*param.data.shape) * 0.5

        forgetting = learner.estimate_forgetting()
        assert isinstance(forgetting, dict)
        assert "task1" in forgetting
        assert forgetting["task1"] > 0

    def test_no_snapshots_returns_zero(self):
        """EWC loss with no snapshots returns zero tensor."""
        model = LinearModel()
        learner = EwcContinualLearner(model)

        loss, stats = learner.ewc_loss("nonexistent")
        assert loss.item() == 0.0
        assert stats["active_tasks"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
