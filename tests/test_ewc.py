"""
Tests for Elastic Weight Consolidation (EWC)
"""

import pytest
import torch
import torch.nn as nn
from domains.training.ewc import (
    EWCParameters,
    DiagonalFisherEstimator,
    EwcContinualLearner,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestDiagonalFisherEstimator:
    """Tests for Fisher estimator."""

    def test_init(self):
        """Test Fisher estimator initialization."""
        model = SimpleModel()
        estimator = DiagonalFisherEstimator(model)

        assert len(estimator.fisher_accum) > 0

    def test_estimate(self):
        """Test Fisher estimation."""
        model = SimpleModel()
        estimator = DiagonalFisherEstimator(model)

        # Create dummy data loader
        data = [torch.randn(4, 10) for _ in range(5)]
        loss_fn = nn.MSELoss()

        fisher = estimator.estimate(data, loss_fn, num_samples=10)

        assert len(fisher) > 0
        for name, f in fisher.items():
            assert f.shape == model.state_dict()[name].shape
            assert (f >= 0).all()  # Fisher should be non-negative


class TestEwcContinualLearner:
    """Tests for EWC continual learner."""

    def test_init(self):
        """Test EWC learner initialization."""
        model = SimpleModel()
        learner = EwcContinualLearner(model)

        assert learner.model is not None
        assert learner.params.lambda_ewc == 1000.0

    def test_save_snapshot(self):
        """Test task snapshot saving."""
        model = SimpleModel()
        learner = EwcContinualLearner(model)

        # Dummy data loader
        data = [(torch.randn(4, 10), torch.randn(4, 10)) for _ in range(3)]
        loss_fn = nn.MSELoss()

        snapshot = learner.save_task_snapshot("task1", "Addition", data, loss_fn)

        assert snapshot.task_id == "task1"
        assert snapshot.task_name == "Addition"
        assert len(snapshot.parameters) > 0
        assert len(snapshot.fisher_diagonal) > 0

    def test_ewc_loss(self):
        """Test EWC loss calculation."""
        model = SimpleModel()
        params = EWCParameters(lambda_ewc=1000.0)
        learner = EwcContinualLearner(model, params=params)

        # Save a snapshot first
        data = [(torch.randn(4, 10), torch.randn(4, 10)) for _ in range(3)]
        loss_fn = nn.MSELoss()
        learner.save_task_snapshot("task1", "Task1", data, loss_fn)

        # Calculate EWC loss (should be 0 since params haven't changed)
        ewc_loss, stats = learner.ewc_loss("task1")

        assert isinstance(ewc_loss.item(), float)

    def test_forward_and_ewc(self):
        """Test forward pass with EWC loss."""
        model = SimpleModel()
        learner = EwcContinualLearner(model)

        # Save snapshot
        data = [(torch.randn(4, 10), torch.randn(4, 10)) for _ in range(3)]
        loss_fn = nn.MSELoss()
        learner.save_task_snapshot("task1", "Task1", data, loss_fn)

        # Forward pass
        batch = (torch.randn(4, 10), torch.randn(4, 10))
        total_loss, stats = learner.forward_and_ewc(batch, loss_fn, "task1")

        assert "total_loss" in stats
        assert "task_loss" in stats
        assert "ewc_loss" in stats

    def test_prune_consolidation(self):
        """Test identifying important parameters."""
        model = SimpleModel()
        learner = EwcContinualLearner(model)

        # Save snapshots
        data = [(torch.randn(4, 10), torch.randn(4, 10)) for _ in range(3)]
        loss_fn = nn.MSELoss()
        learner.save_task_snapshot("task1", "Task1", data, loss_fn)

        important = learner.prune_consolidation(top_k_percent=20)
        assert isinstance(important, dict)

    def test_estimate_forgetting(self):
        """Test forgetting estimation."""
        model = SimpleModel()
        learner = EwcContinualLearner(model)

        # Save snapshot
        data = [(torch.randn(4, 10), torch.randn(4, 10)) for _ in range(3)]
        loss_fn = nn.MSELoss()
        learner.save_task_snapshot("task1", "Task1", data, loss_fn)

        forgetting = learner.estimate_forgetting()
        assert isinstance(forgetting, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
