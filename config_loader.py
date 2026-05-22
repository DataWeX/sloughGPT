#!/usr/bin/env python3
"""Configuration loader for SloughGPT."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "sloughgpt"
    #: Optional Soul / job display name; defaults to ``name`` for trainer exports and ``train --api``.
    soul_name: Optional[str] = None
    vocab_size: Optional[int] = None
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128
    dropout: float = 0.1


@dataclass
class DataConfig:
    dataset: str = "shakespeare"
    data_path: str = "datasets/shakespeare/input.txt"
    val_split: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    min_lr: float = 1e-5
    warmup_steps: int = 500
    scheduler: str = "cosine"
    #: Steps between progress logs (CLI / HTTP ``TrainingRequest`` / trainer).
    log_interval: int = 10
    #: Steps between eval passes.
    eval_interval: int = 100
    #: Optional step cap from CLI merge; ``None`` means use epochs only.
    max_steps: Optional[int] = None
    use_mixed_precision: bool = True
    #: ``fp16`` or ``bf16`` (``SloughGPTTrainer``).
    mixed_precision_dtype: str = "bf16"


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])


@dataclass
class QuantizationConfig:
    enabled: bool = False
    bits: int = 8


@dataclass
class CheckpointConfig:
    save_every: int = 5
    keep_last: int = 3
    save_dir: str = "models"
    #: Set by CLI ``--resume`` merge when present; not required in YAML.
    resume: Optional[str] = None
    #: ``SloughGPTTrainer`` periodic ``step_*.pt`` directory (``--checkpoint-dir``).
    trainer_dir: str = "checkpoints"
    #: Steps between full trainer checkpoints (``--checkpoint-interval``).
    trainer_interval: int = 1000
    save_best_only: bool = False
    max_checkpoints: int = 5
    #: Post-training artifact format for ``cli.py train`` (``--save-format``).
    export_format: str = "sou"


@dataclass
class TrackingConfig:
    enabled: bool = False
    backend: str = "wandb"  # wandb, mlflow, none
    project: str = "sloughgpt"
    entity: Optional[str] = None
    log_every: int = 10


@dataclass
class DeviceConfig:
    type: str = "auto"  # auto, cpu, mps, cuda


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        return Config()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return Config()
    
    return Config(
        model=ModelConfig(**data.get('model', {})),
        data=DataConfig(**data.get('data', {})),
        training=TrainingConfig(**data.get('training', {})),
        lora=LoRAConfig(**data.get('lora', {})),
        quantization=QuantizationConfig(**data.get('quantization', {})),
        checkpoint=CheckpointConfig(**data.get('checkpoint', {})),
        tracking=TrackingConfig(**data.get('tracking', {})),
        device=DeviceConfig(**data.get('device', {})),
    )


def get_device(config: DeviceConfig) -> str:
    """Get device string based on config."""
    if config.type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return config.type


def merge_args_with_config(config: Config, args) -> Config:
    """Merge CLI args with config."""
    import argparse
    
    if hasattr(args, 'dataset') and args.dataset:
        config.data.dataset = args.dataset
        config.data.data_path = f"datasets/{args.dataset}/input.txt"
    
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
    
    if hasattr(args, 'lr') and args.lr:
        config.training.learning_rate = args.lr

    if hasattr(args, "dropout") and getattr(args, "dropout", None) is not None:
        config.model.dropout = float(args.dropout)

    if hasattr(args, "scheduler") and getattr(args, "scheduler", None):
        config.training.scheduler = str(args.scheduler)

    if hasattr(args, "warmup_steps") and getattr(args, "warmup_steps", None) is not None:
        config.training.warmup_steps = int(args.warmup_steps)

    if hasattr(args, "weight_decay") and getattr(args, "weight_decay", None) is not None:
        config.training.weight_decay = float(args.weight_decay)
    
    if hasattr(args, 'use_lora') and args.use_lora:
        config.lora.enabled = True

    if hasattr(args, "lora_rank") and getattr(args, "lora_rank", None) is not None:
        config.lora.rank = int(args.lora_rank)

    if hasattr(args, "lora_alpha") and getattr(args, "lora_alpha", None) is not None:
        config.lora.alpha = int(args.lora_alpha)
    
    if hasattr(args, 'resume') and args.resume:
        config.checkpoint.resume = args.resume

    if hasattr(args, "checkpoint_dir") and getattr(args, "checkpoint_dir", None):
        config.checkpoint.trainer_dir = str(args.checkpoint_dir)

    if hasattr(args, "checkpoint_interval") and getattr(args, "checkpoint_interval", None) is not None:
        config.checkpoint.trainer_interval = int(args.checkpoint_interval)

    if hasattr(args, "save_best_only") and getattr(args, "save_best_only", False):
        config.checkpoint.save_best_only = True

    if hasattr(args, "max_checkpoints") and getattr(args, "max_checkpoints", None) is not None:
        config.checkpoint.max_checkpoints = int(args.max_checkpoints)
    
    if hasattr(args, "max_steps") and getattr(args, "max_steps", None) is not None:
        config.training.max_steps = int(args.max_steps)

    if getattr(args, "log_interval", None) is not None:
        config.training.log_interval = int(args.log_interval)
    if getattr(args, "eval_interval", None) is not None:
        config.training.eval_interval = int(args.eval_interval)

    if hasattr(args, "min_lr") and getattr(args, "min_lr", None) is not None:
        config.training.min_lr = float(args.min_lr)

    if hasattr(args, "max_grad_norm") and getattr(args, "max_grad_norm", None) is not None:
        config.training.gradient_clip = float(args.max_grad_norm)

    if (
        hasattr(args, "gradient_accumulation_steps")
        and getattr(args, "gradient_accumulation_steps", None) is not None
    ):
        config.training.gradient_accumulation_steps = int(args.gradient_accumulation_steps)

    if hasattr(args, "use_mixed_precision"):
        config.training.use_mixed_precision = bool(args.use_mixed_precision)

    if hasattr(args, "precision") and getattr(args, "precision", None):
        config.training.mixed_precision_dtype = str(args.precision)

    if hasattr(args, "soul_name") and getattr(args, "soul_name", None):
        _sn = str(args.soul_name).strip()
        if _sn:
            config.model.soul_name = _sn

    if hasattr(args, "save_format") and getattr(args, "save_format", None):
        config.checkpoint.export_format = str(args.save_format)

    if hasattr(args, "train_device") and getattr(args, "train_device", None):
        config.device.type = str(args.train_device)

    return config


import torch
