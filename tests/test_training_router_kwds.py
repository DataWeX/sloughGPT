"""Unit tests for ``training.router._sloughgpt_trainer_kwds`` (HTTP → SloughGPTTrainer)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def test_maps_learning_rate_scheduler_and_device() -> None:
    from training.router import _sloughgpt_trainer_kwds

    snap = {
        "n_embed": 64,
        "n_layer": 2,
        "n_head": 2,
        "block_size": 32,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.02,
        "max_steps": 10,
        "log_interval": 5,
        "eval_interval": 20,
        "scheduler": "linear",
        "device": "cpu",
        "dropout": 0.2,
        "weight_decay": 0.05,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 2.0,
        "use_mixed_precision": False,
        "mixed_precision_dtype": "fp16",
        "warmup_steps": 0,
        "min_lr": 1e-6,
        "use_lora": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "checkpoint_dir": "ck",
        "checkpoint_interval": 99,
        "save_best_only": True,
        "max_checkpoints": 3,
    }
    k = _sloughgpt_trainer_kwds(snap)
    assert k["lr"] == 0.02
    assert k["scheduler_type"] == "linear"
    assert k["device"] == "cpu"
    assert k["dropout"] == 0.2
    assert k["weight_decay"] == 0.05
    assert k["gradient_accumulation_steps"] == 2
    assert k["max_grad_norm"] == 2.0
    assert k["use_mixed_precision"] is False
    assert k["mixed_precision_dtype"] == "fp16"
    assert k["warmup_steps"] == 0
    assert k["min_lr"] == 1e-6
    assert k["use_lora"] is True
    assert k["lora_rank"] == 16
    assert k["lora_alpha"] == 32
    assert k["checkpoint_dir"] == "ck"
    assert k["checkpoint_interval"] == 99
    assert k["save_best_only"] is True
    assert k["max_checkpoints"] == 3


def test_device_none_when_missing() -> None:
    from training.router import _sloughgpt_trainer_kwds

    k = _sloughgpt_trainer_kwds({"learning_rate": 1e-3})
    assert k["device"] is None
