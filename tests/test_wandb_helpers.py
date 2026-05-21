"""Tests for W&B helper flattening and env flags."""

from config_loader import Config, load_config
from domains.training.wandb_helpers import (
    flatten_for_wandb_config,
    wandb_training_enabled_from_env,
)


def test_flatten_for_wandb_config_nested_dataclass():
    cfg = Config()
    flat = flatten_for_wandb_config(cfg)
    assert "model.name" in flat or "training.epochs" in flat
    assert isinstance(flat.get("training.epochs", 0), int)


def test_wandb_training_env_flag(monkeypatch):
    monkeypatch.delenv("SLOUGHGPT_WANDB_TRAINING", raising=False)
    assert wandb_training_enabled_from_env() is False
    monkeypatch.setenv("SLOUGHGPT_WANDB_TRAINING", "1")
    assert wandb_training_enabled_from_env() is True


def test_load_config_tracking_defaults():
    c = load_config("__does_not_exist__.yaml")
    assert c.tracking.enabled is False
    assert c.tracking.backend == "wandb"
