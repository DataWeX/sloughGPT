"""Tests for domains.training.ewc — EWCParameters, TaskSnapshot; domains.training.train_pipeline — TrainerConfig."""

import numpy as np
import pytest
from domains.training.ewc import EWCParameters, TaskSnapshot
from domains.training.train_pipeline import TrainerConfig


# ── EWCParameters ────────────────────────────────────────────────────────


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

    def test_ema_decay_default(self):
        ep = EWCParameters()
        assert ep.ema_decay == 0.9

    def test_ema_decay_custom(self):
        ep = EWCParameters(ema_decay=0.99)
        assert ep.ema_decay == 0.99

    def test_lambda_zero(self):
        ep = EWCParameters(lambda_ewc=0.0)
        assert ep.lambda_ewc == 0.0

    def test_lambda_negative(self):
        ep = EWCParameters(lambda_ewc=-100.0)
        assert ep.lambda_ewc == -100.0

    def test_batch_size_one(self):
        ep = EWCParameters(batch_size=1)
        assert ep.batch_size == 1

    def test_num_samples_zero(self):
        ep = EWCParameters(num_samples=0)
        assert ep.num_samples == 0

    def test_clip_grad_norm_zero(self):
        ep = EWCParameters(clip_grad_norm=0.0)
        assert ep.clip_grad_norm == 0.0

    def test_diagonal_approx_false(self):
        ep = EWCParameters(diagonal_approx=False)
        assert ep.diagonal_approx is False

    def test_all_fields_custom(self):
        ep = EWCParameters(
            lambda_ewc=200.0, diagonal_approx=False, batch_size=64,
            num_samples=500, clip_grad_norm=5.0, ema_decay=0.95,
        )
        assert ep.lambda_ewc == 200.0
        assert ep.diagonal_approx is False
        assert ep.batch_size == 64
        assert ep.num_samples == 500
        assert ep.clip_grad_norm == 5.0
        assert ep.ema_decay == 0.95

    def test_large_values(self):
        ep = EWCParameters(lambda_ewc=1e6, batch_size=1024, num_samples=100000)
        assert ep.lambda_ewc == 1e6
        assert ep.batch_size == 1024

    def test_repr(self):
        ep = EWCParameters()
        r = repr(ep)
        assert "EWCParameters" in r

    def test_equality(self):
        ep1 = EWCParameters(lambda_ewc=100.0, batch_size=16)
        ep2 = EWCParameters(lambda_ewc=100.0, batch_size=16)
        assert ep1 == ep2

    def test_inequality(self):
        ep1 = EWCParameters(lambda_ewc=100.0)
        ep2 = EWCParameters(lambda_ewc=200.0)
        assert ep1 != ep2


# ── TaskSnapshot ─────────────────────────────────────────────────────────


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

    def test_parameters_dict(self):
        params = {"w1": np.array([1.0]), "w2": np.array([2.0, 3.0])}
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters=params,
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        assert len(ts.parameters) == 2
        assert "w1" in ts.parameters
        assert "w2" in ts.parameters

    def test_fisher_diagonal_dict(self):
        fisher = {"w1": np.array([0.1, 0.2, 0.3])}
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal=fisher, optimal_loss=0.0, num_samples=0,
        )
        assert "w1" in ts.fisher_diagonal
        assert len(ts.fisher_diagonal["w1"]) == 3

    def test_empty_parameters(self):
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        assert ts.parameters == {}

    def test_zero_optimal_loss(self):
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        assert ts.optimal_loss == 0.0

    def test_negative_optimal_loss(self):
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=-1.5, num_samples=0,
        )
        assert ts.optimal_loss == -1.5

    def test_large_num_samples(self):
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=1000000,
        )
        assert ts.num_samples == 1000000

    def test_many_parameters(self):
        params = {f"w{i}": np.array([float(i)]) for i in range(50)}
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters=params,
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        assert len(ts.parameters) == 50

    def test_numpy_array_values_preserved(self):
        arr = np.array([1.0, 2.0, 3.0])
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={"w": arr},
            fisher_diagonal={"w": arr * 0.1}, optimal_loss=0.0, num_samples=0,
        )
        np.testing.assert_array_equal(ts.parameters["w"], arr)
        np.testing.assert_array_almost_equal(ts.fisher_diagonal["w"], arr * 0.1)

    def test_repr(self):
        ts = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        r = repr(ts)
        assert "TaskSnapshot" in r

    def test_equality(self):
        ts1 = TaskSnapshot(
            task_id="t1", task_name="n", parameters={"w": np.array([1.0])},
            fisher_diagonal={}, optimal_loss=0.5, num_samples=10,
        )
        ts2 = TaskSnapshot(
            task_id="t1", task_name="n", parameters={"w": np.array([1.0])},
            fisher_diagonal={}, optimal_loss=0.5, num_samples=10,
        )
        assert ts1 == ts2

    def test_inequality_different_id(self):
        ts1 = TaskSnapshot(
            task_id="t1", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        ts2 = TaskSnapshot(
            task_id="t2", task_name="n", parameters={},
            fisher_diagonal={}, optimal_loss=0.0, num_samples=0,
        )
        assert ts1 != ts2


# ── TrainerConfig ────────────────────────────────────────────────────────


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

    def test_n_head_default(self):
        tc = TrainerConfig()
        assert tc.n_head == 4

    def test_block_size_default(self):
        tc = TrainerConfig()
        assert tc.block_size == 64

    def test_dropout_default(self):
        tc = TrainerConfig()
        assert tc.dropout == 0.1

    def test_learning_rate_default(self):
        tc = TrainerConfig()
        assert tc.learning_rate == 1e-3

    def test_weight_decay_default(self):
        tc = TrainerConfig()
        assert tc.weight_decay == 0.01

    def test_max_grad_norm_default(self):
        tc = TrainerConfig()
        assert tc.max_grad_norm == 1.0

    def test_scheduler_type_default(self):
        tc = TrainerConfig()
        assert tc.scheduler_type == "cosine"

    def test_warmup_steps_default(self):
        tc = TrainerConfig()
        assert tc.warmup_steps == 100

    def test_min_lr_default(self):
        tc = TrainerConfig()
        assert tc.min_lr == 1e-5

    def test_checkpoint_dir_default(self):
        tc = TrainerConfig()
        assert tc.checkpoint_dir == "models/auto-training"

    def test_checkpoint_interval_default(self):
        tc = TrainerConfig()
        assert tc.checkpoint_interval == 500

    def test_save_best_only_default(self):
        tc = TrainerConfig()
        assert tc.save_best_only is False

    def test_max_checkpoints_default(self):
        tc = TrainerConfig()
        assert tc.max_checkpoints == 5

    def test_use_lora_default(self):
        tc = TrainerConfig()
        assert tc.use_lora is False

    def test_lora_rank_default(self):
        tc = TrainerConfig()
        assert tc.lora_rank == 8

    def test_lora_alpha_default(self):
        tc = TrainerConfig()
        assert tc.lora_alpha == 16

    def test_log_interval_default(self):
        tc = TrainerConfig()
        assert tc.log_interval == 10

    def test_eval_interval_default(self):
        tc = TrainerConfig()
        assert tc.eval_interval == 100

    def test_early_stopping_patience_default(self):
        tc = TrainerConfig()
        assert tc.early_stopping_patience == 0

    def test_gradient_accumulation_steps_default(self):
        tc = TrainerConfig()
        assert tc.gradient_accumulation_steps == 1

    def test_max_steps_default(self):
        tc = TrainerConfig()
        assert tc.max_steps is None

    def test_all_fields_custom(self):
        tc = TrainerConfig(
            vocab_size=1000, n_embed=128, n_layer=6, n_head=8,
            block_size=128, dropout=0.2, batch_size=64, epochs=50,
            learning_rate=5e-4, weight_decay=0.1, max_grad_norm=2.0,
            scheduler_type="linear", warmup_steps=200, min_lr=1e-6,
            checkpoint_dir="custom/checkpoints", checkpoint_interval=100,
            save_best_only=True, max_checkpoints=3, use_lora=True,
            lora_rank=16, lora_alpha=32, log_interval=5, eval_interval=50,
            early_stopping_patience=5, gradient_accumulation_steps=4,
            max_steps=1000, device="cpu",
        )
        assert tc.vocab_size == 1000
        assert tc.n_embed == 128
        assert tc.n_layer == 6
        assert tc.n_head == 8
        assert tc.block_size == 128
        assert tc.dropout == 0.2
        assert tc.batch_size == 64
        assert tc.epochs == 50
        assert tc.learning_rate == 5e-4
        assert tc.weight_decay == 0.1
        assert tc.max_grad_norm == 2.0
        assert tc.scheduler_type == "linear"
        assert tc.warmup_steps == 200
        assert tc.min_lr == 1e-6
        assert tc.checkpoint_dir == "custom/checkpoints"
        assert tc.checkpoint_interval == 100
        assert tc.save_best_only is True
        assert tc.max_checkpoints == 3
        assert tc.use_lora is True
        assert tc.lora_rank == 16
        assert tc.lora_alpha == 32
        assert tc.log_interval == 5
        assert tc.eval_interval == 50
        assert tc.early_stopping_patience == 5
        assert tc.gradient_accumulation_steps == 4
        assert tc.max_steps == 1000

    def test_repr(self):
        tc = TrainerConfig()
        r = repr(tc)
        assert "TrainerConfig" in r

    def test_equality(self):
        tc1 = TrainerConfig(vocab_size=256, n_layer=2)
        tc2 = TrainerConfig(vocab_size=256, n_layer=2)
        assert tc1 == tc2

    def test_inequality(self):
        tc1 = TrainerConfig(vocab_size=256)
        tc2 = TrainerConfig(vocab_size=512)
        assert tc1 != tc2
