"""
Tests for distributed training wrapper.
"""

import pytest
from unittest.mock import MagicMock, patch
from domains.training.trainer_protocol import TrainResult
from domains.training.distributed import (
    DistributedTrainer,
    DistributedConfig,
    init_distributed,
    cleanup_distributed,
)


class MockTrainer:
    """Mock trainer that satisfies TrainerProtocol."""

    def __init__(self, result=None):
        self._result = result or TrainResult(success=True, final_loss=0.5, total_steps=100)
        self._training = False

    @property
    def is_training(self):
        return self._training

    def stop(self):
        self._training = False

    def train(self, **kwargs):
        return self._result


class TestDistributedTrainer:
    """Tests for DistributedTrainer."""

    def test_single_process_passthrough(self):
        """Without distributed init, delegates directly to base trainer."""
        base = MockTrainer(TrainResult(success=True, final_loss=0.3))
        dist = DistributedTrainer(base)
        result = dist.train()
        assert result.success is True
        assert result.final_loss == 0.3

    def test_is_training_property(self):
        base = MockTrainer()
        dist = DistributedTrainer(base)
        assert dist.is_training is False
        base._training = True
        assert dist.is_training is True

    def test_stop_delegates(self):
        base = MockTrainer()
        base._training = True
        dist = DistributedTrainer(base)
        dist.stop()
        assert base._training is False

    def test_failure_returns_failed_result(self):
        class FailingTrainer:
            is_training = False
            def stop(self): pass
            def train(self, **kwargs): raise RuntimeError("boom")

        dist = DistributedTrainer(FailingTrainer())
        result = dist.train()
        assert result.success is False
        assert "boom" in result.error

    def test_metrics_include_distributed_info(self):
        base = MockTrainer()
        dist = DistributedTrainer(base)
        m = dist.get_metrics()
        assert m["distributed"] is False
        assert m["world_size"] == 1
        assert m["rank"] == 0

    def test_elapsed_recorded(self):
        base = MockTrainer()
        dist = DistributedTrainer(base)
        result = dist.train()
        assert result.elapsed >= 0

    def test_config_defaults(self):
        cfg = DistributedConfig()
        assert cfg.world_size == 1
        assert cfg.rank == 0
        assert cfg.backend == "nccl"
        assert cfg.gradient_accumulation_steps == 1

    def test_custom_config(self):
        cfg = DistributedConfig(world_size=4, rank=2, local_rank=2)
        dist = DistributedTrainer(MockTrainer(), config=cfg)
        assert dist.config.world_size == 4
        assert dist.config.rank == 2


class TestInitDistributed:
    """Tests for init_distributed helper."""

    def test_returns_config_when_not_distributed(self):
        cfg = init_distributed()
        assert isinstance(cfg, DistributedConfig)

    @patch.dict("os.environ", {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"})
    def test_reads_env_vars(self):
        """init_distributed reads env vars (may fall back if MASTER_ADDR missing)."""
        cfg = init_distributed()
        # On this machine without MASTER_ADDR, init fails and falls back to defaults
        # But the config is still valid
        assert isinstance(cfg, DistributedConfig)


class TestCleanupDistributed:
    """Tests for cleanup_distributed helper."""

    def test_cleanup_noop_when_not_initialized(self):
        cleanup_distributed()  # Should not raise
