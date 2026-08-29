"""Tests for domains.training.ewc — EWCParameters, TaskSnapshot; domains.training.train_pipeline — TrainerConfig."""

import numpy as np
from domains.training.ewc import EWCParameters, TaskSnapshot
from domains.training.train_pipeline import TrainerConfig


class TestEWCParameters:
    def test_defaults(self):
        ep = EWCParameters()
        assert ep.lambda_ewc == 1000.0
        assert ep.diagonal_approx is True
        assert ep.batch_size == 32
        assert ep.num_samples == 100
        assert ep.clip_grad_norm == 10.0

    def test_custom(self):
        ep = EWCParameters(lambda_ewc=500.0, batch_size=16)
        assert ep.lambda_ewc == 500.0
        assert ep.batch_size == 16


class TestTaskSnapshot:
    def test_fields(self):
        ts = TaskSnapshot(
            task_id="t1",
            task_name="test task",
            parameters={"w": np.array([1.0, 2.0])},
            fisher_diagonal={"w": np.array([0.1, 0.2])},
            optimal_loss=0.5,
            num_samples=100,
        )
        assert ts.task_id == "t1"
        assert ts.task_name == "test task"
        assert ts.optimal_loss == 0.5
        assert ts.num_samples == 100


class TestTrainerConfig:
    def test_defaults(self):
        tc = TrainerConfig()
        assert tc.vocab_size == 256
        assert tc.n_embed == 64
        assert tc.n_layer == 2
        assert tc.batch_size == 32
        assert tc.epochs == 10
        assert tc.device == "cpu"

    def test_custom(self):
        tc = TrainerConfig(vocab_size=512, n_layer=4, epochs=20)
        assert tc.vocab_size == 512
        assert tc.n_layer == 4
        assert tc.epochs == 20

    def test_auto_device(self):
        tc = TrainerConfig(device="auto")
        assert tc.device == "cpu"
