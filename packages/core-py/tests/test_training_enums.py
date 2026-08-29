"""Tests for domains.training — RLHFMetric, RLHFConfig, TrackerBackend, TrackingConfig, LoRAType, LoRAConfig, DataFormat."""

from domains.training.rlhf import RLHFMetric, RLHFConfig
from domains.training.tracking import TrackerBackend, TrackingConfig
from domains.training.lora import LoRAType, LoRAConfig
from domains.training import DataFormat


class TestRLHFMetric:
    def test_all_members(self):
        assert len(RLHFMetric) == 6

    def test_values(self):
        assert RLHFMetric.REWARD.value == "reward"
        assert RLHFMetric.KL_DIVERGENCE.value == "kl_divergence"


class TestRLHFConfig:
    def test_defaults(self):
        cfg = RLHFConfig()
        assert cfg.ppo_epochs == 4
        assert cfg.clip_epsilon == 0.2
        assert cfg.gamma == 1.0
        assert cfg.lam == 0.95
        assert cfg.use_ref_model is True


class TestTrackerBackend:
    def test_all_members(self):
        assert len(TrackerBackend) >= 3

    def test_values(self):
        assert TrackerBackend.MLFLOW.value == "mlflow"
        assert TrackerBackend.WANDB.value == "wandb"


class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.backend == TrackerBackend.NONE


class TestLoRAType:
    def test_all_members(self):
        assert len(LoRAType) >= 2

    def test_values(self):
        assert LoRAType.LORA.value == "lora"
        assert LoRAType.IA3.value == "ia3"


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.dropout == 0.05

    def test_custom(self):
        cfg = LoRAConfig(rank=16, alpha=32.0)
        assert cfg.rank == 16
        assert cfg.alpha == 32.0


class TestDataFormat:
    def test_all_members(self):
        assert len(DataFormat) == 3

    def test_values(self):
        assert DataFormat.JSON.value == "json"
        assert DataFormat.JSONL.value == "jsonl"
        assert DataFormat.CSV.value == "csv"
