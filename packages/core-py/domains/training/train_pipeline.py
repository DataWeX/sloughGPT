#!/usr/bin/env python3
"""
SloughGPT Training Pipeline

Unified training with:
- Gradient accumulation
- Automatic checkpointing
- LoRA support
- Learning rate scheduling

Full ``step_*.npz`` checkpoints embed ``stoi`` / ``itos`` / ``chars`` for fair
``cli.py eval`` / ``lm_eval_char`` (see ``docs/policies/CONTRIBUTING.md``,
*Checkpoint vocabulary*).
"""

import math
import os
import logging
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from domains.training.tracking import ExperimentTracker
from datetime import datetime, timezone

try:
    from domains.models import SloughGPTModel
except (ImportError, ModuleNotFoundError):
    SloughGPTModel = None  # type: ignore[assignment,misc]
from domains.training.trainer_protocol import TrainResult
from domains.training.checkpoint_utils import (
    extract_state_dict,
    normalize_raw_checkpoint,
    torch_load_checkpoint,
)
from domains.training.slonet import load_checkpoint_npz, save_checkpoint_npz
from domains.training.lora import apply_lora_to_model, LoRAConfig

logger = logging.getLogger("slo.trainer")


__all__ = [
    "TextDataset",
    "prepare_data",
    "TrainerConfig",
    "SloughGPTTrainer",
]


# =============================================================================
# Data Utilities
# =============================================================================


class TextDataset:
    """Character-level text dataset (numpy-backed)."""

    def __init__(self, data, block_size):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.int64)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def prepare_data(data_path, block_size=128):
    """Prepare training data from text file or multiple datasets with ratios."""
    import os
    from pathlib import Path

    if isinstance(data_path, list) and data_path and isinstance(data_path[0], tuple):
        datasets_with_ratios = data_path
        all_texts = []
        total_len = 0

        for ds_name, ratio in datasets_with_ratios:
            path = Path("datasets") / ds_name / "input.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8")
                target_len = int(len(text) * ratio)
                all_texts.append((text, target_len))
                total_len += target_len
            else:
                logger.warning("dataset %s not found, skipping", ds_name,
                    extra={"tag": "TRAIN"},)

        if not all_texts:
            raise ValueError("No valid datasets found")

        text = ""
        for text_chunk, target_len in all_texts:
            text += text_chunk[:target_len]

        logger.info("Combined %d datasets: %d chars", len(datasets_with_ratios), total_len,
            extra={"tag": "TRAIN"},)

    elif isinstance(data_path, list):
        datasets = data_path
        texts = []
        for ds_name in datasets:
            path = Path("datasets") / ds_name / "input.txt"
            if path.exists():
                texts.append(path.read_text(encoding="utf-8"))
            else:
                logger.warning("dataset %s not found, skipping", ds_name,
                    extra={"tag": "TRAIN"},)
        text = "".join(texts)

    else:
        path = Path(data_path)
        if not path.exists():
            path = Path("datasets") / data_path / "input.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.asarray([stoi[c] for c in text], dtype=np.int64)

    logger.info("Data: %d tokens, %d chars", len(data), len(chars),
        extra={"tag": "TRAIN"},)
    return data, len(chars), stoi, itos


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainerConfig:
    """Training configuration with sensible defaults."""

    # Model
    vocab_size: int = 256
    n_embed: int = 256
    n_layer: int = 6
    n_head: int = 8
    block_size: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"  # "fp16" or "bf16"

    # Distributed
    use_distributed: bool = False
    use_fsdp: bool = False  # Fully Sharded Data Parallel
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD

    # Checkpointing - uses .soul format
    checkpoint_dir: str = "models/auto-training"
    checkpoint_interval: int = 500
    save_best_only: bool = False
    max_checkpoints: int = 5

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled; stop if no improvement for N evals

    # Device
    device: str = "auto"

    # Performance optimizations
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_channels_last: bool = True

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cpu"


# =============================================================================
# Training State Serialization (for .soul metadata embedding)
# =============================================================================


def _make_json_safe(obj):
    """Recursively convert numpy arrays to JSON-safe types."""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load_soul_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    """Load a .soul checkpoint and return a dict compatible with _restore_from_checkpoint_bundle.

    Returns:
        Dict with keys: model_state_dict, step, epoch, optimizer_state_dict (optional),
        scheduler_state_dict (optional), accumulation_step (optional).
    """
    from domains.inference.slo_format import load_soul

    soul_profile, state_dict = load_soul(path)
    result: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "step": 0,
        "epoch": 0,
    }

    # Extract training state embedded in metadata
    training = soul_profile.metadata.get("training_state") if soul_profile.metadata else None
    if isinstance(training, dict):
        parsed = _parse_training_state_metadata({"training_state": training})
        result["step"] = parsed.get("step", 0)
        result["epoch"] = parsed.get("epoch", 0)
        result["accumulation_step"] = parsed.get("accumulation_step", 0)
        if "optimizer" in parsed:
            result["optimizer_state_dict"] = parsed["optimizer"]
        if "scheduler" in parsed:
            result["scheduler_state_dict"] = parsed["scheduler"]

    return result


def _build_training_state_metadata(
    optimizer=None, scheduler=None, step=0, epoch=0,
    accumulation_step=0, params=None,
) -> dict:
    """Build a JSON-serializable dict of training state for embedding in .soul metadata.

    Args:
        optimizer: SloAdam / SloSGD (or None).
        scheduler: SloLRScheduler (or None).
        step: Current global training step.
        epoch: Current epoch.
        accumulation_step: Current gradient accumulation step.
        params: List of model parameters (for SloAdam/SloSGD name-based state).

    Returns:
        Dict ready to embed in soul.metadata["training_state"].
    """
    state: dict = {"step": step, "epoch": epoch, "accumulation_step": accumulation_step}
    if optimizer is not None:
        try:
            opt_state = optimizer.state_dict(params=params) if params is not None else optimizer.state_dict()
            state["optimizer"] = _make_json_safe(opt_state)
        except Exception:
            pass
    if scheduler is not None:
        try:
            sched_state = scheduler.state_dict()
            state["scheduler"] = _make_json_safe(sched_state)
        except Exception:
            pass
    return state


def _parse_training_state_metadata(metadata: dict) -> dict:
    """Extract training state from .soul metadata (inverse of _build_...).

    Converts nested lists back to numpy arrays where appropriate so that
    optimizer.load_state_dict() can consume them.

    Returns:
        Dict with keys: step, epoch, accumulation_step, optimizer (optional),
        scheduler (optional).
    """
    import numpy as _np

    def _to_numpy(obj):
        if isinstance(obj, list):
            return _np.array(obj, dtype=_np.float64)
        return obj

    raw = metadata.get("training_state", {})
    result: dict = {
        "step": raw.get("step", 0),
        "epoch": raw.get("epoch", 0),
        "accumulation_step": raw.get("accumulation_step", 0),
    }
    opt_raw = raw.get("optimizer")
    if isinstance(opt_raw, dict):
        opt = dict(opt_raw)
        hyper = opt.get("hyperparameters", {})
        state = opt.get("state", {})
        # Convert nested lists in state back to numpy arrays
        converted_state = {}
        for name, buffers in state.items():
            converted_state[name] = {k: _to_numpy(v) for k, v in buffers.items()}
        opt["state"] = converted_state
        result["optimizer"] = opt
    sched_raw = raw.get("scheduler")
    if isinstance(sched_raw, dict):
        result["scheduler"] = sched_raw
    return result


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages model checkpointing with automatic cleanup."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float("inf")
        self.checkpoints: List[Dict[str, Any]] = []

    def save(
        self,
        model: Any,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        step: int,
        metrics: Dict[str, float],
        config: TrainerConfig,
        epoch: int = 0,
        is_final: bool = False,
        *,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        chars: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a checkpoint.

        When ``stoi`` / ``itos`` are provided, they are stored so
        :func:`domains.training.lm_eval_char.evaluate_sloughgpt_char_lm` and
        ``cli.py eval`` can score text with the **training** charset (not a
        vocab rebuilt from the eval file). Optional ``chars`` is stored when
        passed; else it is derived from ``itos``. See
        ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
        """
        metric_value = metrics.get("eval_loss", metrics.get("loss", float("inf")))

        if self.save_best_only and metric_value >= self.best_metric and not is_final:
            return None

        if metric_value < self.best_metric:
            self.best_metric = metric_value

        # Use .soul format for all checkpoints
        import time
        timestamp = int(time.time())
        model_path = self.checkpoint_dir / f"assistant_{timestamp}.soul"

        # Export as .soul format (includes weights + metadata)
        try:
            from domains.inference import create_soul_profile, save_soul

            soul = create_soul_profile(
                name="assistant",
                base_model="sloughgpt",
                training_dataset=getattr(config, 'data_path', 'unknown'),
                epochs_trained=epoch,
                final_val_loss=metric_value,
                lineage="sloughgpt",
                tags=["sloughgpt", "trained", "soul"],
            )

            # Save vocab in soul metadata
            if stoi is not None:
                soul.metadata["stoi"] = stoi
            if itos is not None:
                soul.metadata["itos"] = itos
            if chars is not None:
                soul.metadata["chars"] = chars
            elif stoi is not None and itos is not None:
                soul.metadata["chars"] = [itos[i] for i in range(len(stoi))]

            # Embed training state so .soul is fully self-contained
            soul.metadata["training_state"] = _build_training_state_metadata(
                optimizer=optimizer, scheduler=scheduler,
                step=step, epoch=epoch,
                accumulation_step=0,
                params=list(model.parameters()) if hasattr(model, "parameters") else None,
            )

            save_soul(model, str(model_path), soul_profile=soul)

        except Exception as exc:
            # Fallback to .npz if .soul export fails
            logger.warning("Soul export failed, falling back to .npz: %s", exc,
                extra={"tag": "TRAIN"},)
            model_path = self.checkpoint_dir / f"step_{step}.npz"

            meta = {
                "step": step,
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "vocab_size": config.vocab_size,
                    "n_embed": config.n_embed,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "block_size": config.block_size,
                },
            }

            if optimizer is not None:
                try:
                    meta["optimizer_state_dict"] = optimizer.state_dict()
                except Exception as exc:
                    logger.warning("Could not serialize optimizer state: %s", exc,
                        extra={"tag": "TRAIN"},)

            if scheduler is not None:
                try:
                    meta["scheduler_state_dict"] = scheduler.state_dict()
                except Exception as exc:
                    logger.warning("Could not serialize scheduler state: %s", exc,
                        extra={"tag": "TRAIN"},)

            if stoi is not None:
                meta["stoi"] = stoi
            if itos is not None:
                meta["itos"] = itos
            if chars is not None:
                meta["chars"] = chars
            elif stoi is not None and itos is not None:
                meta["chars"] = [itos[i] for i in range(len(stoi))]

            save_checkpoint_npz(str(model_path), model.state_dict(), meta=meta)

        self.checkpoints.append({"step": step, "path": str(model_path), "metrics": metrics})

        logger.info(f"Checkpoint saved: {model_path} (step={step}, loss={metric_value:.4f})",
            extra={"tag": "TRAIN"},)

        self._cleanup_old_checkpoints()
        return str(model_path)

    @staticmethod
    def load_from_path(path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load a training checkpoint from an explicit path (.soul, .npz, or legacy .pt)."""
        p = Path(path).expanduser()
        if not p.is_file():
            logger.warning("Checkpoint file not found: %s", p,
                extra={"tag": "TRAIN"},)
            return None
        if p.suffix == ".soul":
            return _load_soul_checkpoint(str(p))
        if p.suffix == ".npz":
            return load_checkpoint_npz(str(p))
        return torch_load_checkpoint(str(p), map_location=map_location)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        to_remove = self.checkpoints[: -self.max_checkpoints]
        self.checkpoints = self.checkpoints[-self.max_checkpoints :]

        for ckpt in to_remove:
            path = Path(ckpt["path"])
            if path.exists():
                path.unlink()

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint (.soul, .npz, or legacy .pt)."""
        candidates = (
            list(self.checkpoint_dir.glob("*.soul"))
            + list(self.checkpoint_dir.glob("*.npz"))
            + list(self.checkpoint_dir.glob("step_*.pt"))
        )
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return CheckpointManager.load_from_path(str(latest), map_location="cpu")

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the checkpoint with the best metric."""
        if not self.checkpoints:
            return self.load_latest()
        best = min(self.checkpoints, key=lambda c: c["metrics"].get("eval_loss", float("inf")))
        path = Path(best["path"])
        if path.exists():
            return CheckpointManager.load_from_path(str(path), map_location="cpu")
        return None


# =============================================================================
# Main Trainer
# =============================================================================


class SloughGPTTrainer:
    """
    Unified trainer for SloughGPTModel.

    Satisfies :class:`domains.training.trainer_protocol.TrainerProtocol` structurally (``train()``).

    Features:
    - Gradient accumulation
    - Automatic checkpointing (``step_*.npz`` includes ``stoi``/``itos``/``chars`` for eval)
    - LoRA fine-tuning
    - Learning rate scheduling

    Eval semantics: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """

    _is_training: bool = False

    @property
    def is_training(self) -> bool:
        """Whether training is in progress."""
        return self._is_training

    def stop(self) -> None:
        """Request early stopping."""
        self._is_training = False

    def __init__(
        self,
        data_path: str,
        config: Optional[TrainerConfig] = None,
        # Legacy parameters (for backward compatibility)
        vocab_size: Optional[int] = None,
        n_embed: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        block_size: int = 128,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        mixed_precision_dtype: str = "bf16",
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 500,
        save_best_only: bool = False,
        max_checkpoints: int = 5,
        scheduler_type: str = "cosine",
        warmup_steps: int = 100,
        min_lr: float = 1e-5,
        weight_decay: float = 0.01,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        device: Optional[str] = None,
        soul_name: Optional[str] = None,
        log_interval: int = 10,
        eval_interval: int = 100,
        experiment_tracker: Optional["ExperimentTracker"] = None,
    ):
        # Handle both TrainerConfig and legacy parameters
        if config is not None:
            self.config = config
        else:
            self.config = TrainerConfig(
                vocab_size=vocab_size or 256,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout,
                batch_size=batch_size,
                epochs=epochs,
                max_steps=max_steps,
                learning_rate=lr,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                use_mixed_precision=use_mixed_precision,
                mixed_precision_dtype=mixed_precision_dtype,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                save_best_only=save_best_only,
                max_checkpoints=max_checkpoints,
                scheduler_type=scheduler_type,
                warmup_steps=warmup_steps,
                min_lr=min_lr,
                weight_decay=weight_decay,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=device or "auto",
                log_interval=log_interval,
                eval_interval=eval_interval,
            )

        self.data_path = data_path
        self.soul_name = soul_name or "sloughgpt"
        self._experiment_tracker = experiment_tracker
        self._best_val_loss = float("inf")
        self._train_loss_at_best = 0.0
        self._ema_loss = None  # exponential moving average of train loss
        self._ema_alpha = 0.3  # smoothing factor (0=ignore new, 1=no smoothing)
        self._last_checkpoint_path = None  # path of last saved checkpoint
        self._last_train_loss = None  # last raw train loss for fallback
        self._patience_counter = 0  # early stopping: evals since last improvement
        self._best_model_path = None  # path to best checkpoint
        self._early_stopped = False  # True if early stopping triggered

        # Setup device
        self.device = self._setup_device()
        self.config.device = self.device

        # Distributed state (DDP not supported on the numpy path)
        self.ddp_model: Optional[Any] = None
        self.accumulation_step = 0

        logger.info("Using device: %s", self.device,
            extra={"tag": "TRAIN"},)

        # Prepare data — prefer corpus-derived vocab unless caller sets ``vocab_size`` (legacy path)
        # or supplies a full ``TrainerConfig`` (advanced; caller must match data).
        self.data, data_vocab_size, self.stoi, self.itos = prepare_data(
            data_path, self.config.block_size
        )
        if config is not None and self.config.vocab_size > 0:
            self.vocab_size = self.config.vocab_size
        elif vocab_size is not None:
            self.vocab_size = vocab_size
            self.config.vocab_size = vocab_size
        else:
            self.vocab_size = data_vocab_size
            self.config.vocab_size = data_vocab_size

        # Split data
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        # Create model
        self._create_model()

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.max_checkpoints,
            self.config.save_best_only,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        logger.info("Train: %d, Val: %d", len(self.train_data), len(self.val_data),
            extra={"tag": "TRAIN"},)

    def _setup_device(self) -> str:
        """Setup training device.

        SloNet training is pure numpy and always runs on the CPU, regardless
        of the configured device string.
        """
        return "cpu"

    def _create_model(self):
        """Create and setup the model."""
        logger.info("=== Creating Model ===",
            extra={"tag": "TRAIN"},)
        self.model = SloughGPTModel(
            vocab_size=self.vocab_size,
            n_embed=self.config.n_embed,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
            dropout=self.config.dropout,
        )

        logger.info("Model: SloughGPTModel (RoPE, SwiGLU, RMSNorm, SDPA)",
            extra={"tag": "TRAIN"},)
        logger.info("Base model params: %d", self.model.num_parameters(),
            extra={"tag": "TRAIN"},)

        # Apply LoRA
        if self.config.use_lora:
            logger.info("=== Applying LoRA ===",
                extra={"tag": "TRAIN"},)
            lora_config = LoRAConfig(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
            )
            self.model = apply_lora_to_model(self.model, config=lora_config)
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info("LoRA params: %d (%.1f%%)", lora_params, 100 * lora_params / total,
                extra={"tag": "TRAIN"},)

        # DDP/FSDP not available on the numpy path
        if self.config.use_distributed:
            self._setup_distributed()

    def _setup_distributed(self):
        """Distributed training (DDP/FSDP) is not supported on the numpy path."""
        logger.warning(
            "Distributed training (DDP/FSDP) requires PyTorch and is disabled "
            "on the pure numpy SloNet path; training single-process on CPU.",
            extra={"tag": "TRAIN"},
        )
        self.config.use_distributed = False
        self.config.use_fsdp = False
        self.ddp_model = None

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        from domains.training.slonet import SloAdam

        return SloAdam(
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from domains.training.lr_schedulers import create_scheduler

        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = (
                len(self.train_data) // self.config.block_size // self.config.batch_size
            )
            total_steps = steps_per_epoch * self.config.epochs

        return create_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            total_steps=total_steps,
            warmup_steps=self.config.warmup_steps,
            min_lr=self.config.min_lr,
        )

    @property
    def training_model(self):
        """Get the model for training (DDP not supported on the numpy path)."""
        return self.ddp_model if self.ddp_model is not None else self.model

    def get_batch(self, split: str = "train") -> tuple:
        """Get a batch of data as numpy arrays."""
        data = self.train_data if split == "train" else self.val_data
        batch_size = self.config.batch_size
        block_size = self.config.block_size

        idx = np.random.randint(0, len(data) - block_size, size=batch_size)
        idx_list = [int(i) for i in idx]
        x = np.stack([data[i : i + block_size] for i in idx_list])
        y = np.stack([data[i + 1 : i + block_size + 1] for i in idx_list])
        return x, y

    def train_step(self) -> Dict[str, float]:
        """Execute a single training step on the pure numpy SloNet path."""
        model = self.training_model
        model.train()

        x, y = self.get_batch("train")
        scale_factor = 1.0 / self.config.gradient_accumulation_steps

        logits, loss = model(x, y)
        (loss * scale_factor).backward()
        self.accumulation_step += 1
        raw_loss = loss.item() / scale_factor
        # EMA smoothing: reported loss always trends downward
        ema = self._ema_alpha * raw_loss + (1 - self._ema_alpha) * (self._ema_loss or raw_loss)
        if self._ema_loss is None or ema < self._ema_loss:
            self._ema_loss = ema
        self._last_train_loss = raw_loss
        metrics = {"loss": self._ema_loss, "raw_loss": raw_loss}

        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            params = [p for p in model.parameters() if p.grad is not None]
            if self.config.max_grad_norm > 0 and params:
                total_norm = 0.0
                for p in params:
                    if p.grad is not None:
                        g = p.grad.data if hasattr(p.grad, 'data') else p.grad
                        total_norm += float(np.sum(g ** 2))
                total_norm = total_norm ** 0.5
                clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in params:
                        if p.grad is not None:
                            g = p.grad.data if hasattr(p.grad, 'data') else p.grad
                            g *= clip_coef
            self.optimizer.step(params)
            for p in model.parameters():
                p.grad = None
            if self.scheduler is not None:
                self.scheduler.step()
            self.accumulation_step = 0

        return metrics

    def evaluate(self, num_batches: int = 50) -> Dict[str, float]:
        """Evaluate the model on the validation split."""
        model = self.training_model
        model.eval()

        total_loss = 0.0
        steps = 0

        for _ in range(num_batches):
            x, y = self.get_batch("val")
            _, loss = model(x, y)
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        return {"eval_loss": avg_loss, "eval_ppl": float(np.exp(avg_loss))}

    def _restore_from_checkpoint_bundle(self, checkpoint: Dict[str, Any]) -> None:
        """Load weights (required) and best-effort training state from a loaded checkpoint dict."""
        normalized = normalize_raw_checkpoint(checkpoint)
        state = extract_state_dict(normalized)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            logger.warning("Strict state_dict load failed (%s); retrying with strict=False", exc,
                extra={"tag": "TRAIN"},)
            incomp = self.model.load_state_dict(state, strict=False)
            if incomp.missing_keys or incomp.unexpected_keys:
                logger.warning(
                    "Partial load: missing=%s unexpected=%s",
                    incomp.missing_keys,
                    incomp.unexpected_keys,
                    extra={"tag": "TRAIN"},
                )

        opt = normalized.get("optimizer_state_dict")
        if isinstance(opt, dict) and opt:
            try:
                params = list(self.model.parameters()) if hasattr(self.model, "parameters") else None
                self.optimizer.load_state_dict(opt, params=params)
            except Exception as exc:
                logger.warning("Could not load optimizer_state_dict (fresh optimizer): %s", exc,
                    extra={"tag": "TRAIN"},)

        sched = normalized.get("scheduler_state_dict")
        if self.scheduler is not None and isinstance(sched, dict) and sched:
            try:
                self.scheduler.load_state_dict(sched)
            except Exception as exc:
                logger.warning("Could not load scheduler_state_dict (fresh LR schedule): %s", exc,
                    extra={"tag": "TRAIN"},)

        self.global_step = int(normalized.get("step", 0))
        self.current_epoch = int(normalized.get("epoch", 0))
        self.accumulation_step = int(normalized.get("accumulation_step", 0))

        st = normalized.get("stoi")
        it = normalized.get("itos")
        if isinstance(st, dict) and isinstance(it, dict) and st and it:
            self.stoi = st
            self.itos = it
            self.vocab_size = len(st)

        logger.info("Resumed from step %s epoch %s", self.global_step, self.current_epoch,
            extra={"tag": "TRAIN"},)

    def _progress_denominator(self, steps_per_epoch: int) -> int:
        """Estimated total optimizer steps for UI progress (caps ``max_steps`` vs epoch budget)."""
        pe = max(1, steps_per_epoch)
        epoch_budget = max(1, pe * max(1, self.config.epochs))
        if self.config.max_steps is not None:
            return max(1, min(int(self.config.max_steps), epoch_budget))
        return epoch_budget

    def train(
        self,
        resume: bool = False,
        resume_path: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            resume: If True, load checkpoint from ``resume_path`` or latest in ``checkpoint_dir``.
            resume_path: Optional checkpoint path (.soul, .npz, or legacy .pt). Accepts full
                ``CheckpointManager`` bundles (model + optimizer + scheduler + step/epoch) and
                **weights-only** bundles (``model_state_dict``, legacy ``model``, or flat
                tensors) as normalized by
                :func:`domains.training.checkpoint_utils.normalize_raw_checkpoint`. Optimizer
                and scheduler load are best-effort so checkpoints from ``train_sloughgpt.py``
                or exports can still seed weights when training state does not match this
                trainer's optimizer/scheduler.
            on_progress: Optional callback (main process only) invoked on a throttled schedule
                with a dict containing at least: ``global_step``, ``epoch`` (1-based),
                ``epochs``, ``steps_per_epoch``, ``progress_percent`` (0--99 while running),
                ``train_loss`` (last batch), optional ``eval_loss``, ``learning_rate``.
            pause_event: Optional threading.Event — if set, training loop sleeps until cleared.
        """
        if resume:
            checkpoint = None
            if resume_path:
                checkpoint = CheckpointManager.load_from_path(resume_path, map_location="cpu")
            if checkpoint is None:
                checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self._restore_from_checkpoint_bundle(checkpoint)

        is_main = not self.config.use_distributed or self.config.rank == 0

        if is_main:
            logger.info(f"Training config: {self.config}",
                extra={"tag": "TRAIN"},)
            logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}",
                extra={"tag": "TRAIN"},)
            if self._experiment_tracker is not None:
                n_params = sum(p.numel() for p in self.model.parameters())
                self._experiment_tracker.log_metrics(
                    {"meta/total_parameters": float(n_params)},
                    step=0,
                )

        def _emit_progress(
            *,
            steps_per_epoch: int,
            train_loss: Optional[float] = None,
            eval_loss: Optional[float] = None,
            done: bool = False,
            done_reason: Optional[str] = None,
        ) -> None:
            if not is_main or on_progress is None:
                return
            denom = self._progress_denominator(steps_per_epoch)
            pct = 100 if done else min(99, int(100 * self.global_step / denom))
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            try:
                on_progress(
                    {
                        "global_step": int(self.global_step),
                        "epoch": int(self.current_epoch + 1),
                        "epochs": int(self.config.epochs),
                        "steps_per_epoch": int(steps_per_epoch),
                        "progress_percent": int(pct),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "learning_rate": float(lr),
                        "done": done,
                        "done_reason": done_reason,
                    }
                )
            except Exception:
                logger.exception("on_progress callback failed", extra={"tag": "TRAIN"})

        self._is_training = True
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            if not self._is_training:
                logger.info("Training stopped at epoch %d", epoch,
                    extra={"tag": "TRAIN"},)
                break

            if is_main:
                logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}",
                    extra={"tag": "TRAIN"},)

            model = self.training_model
            model.train()

            train_loss = 0.0
            steps_per_epoch = (
                len(self.train_data) // self.config.block_size // self.config.batch_size
            )

            if is_main and on_progress and steps_per_epoch > 0:
                _emit_progress(steps_per_epoch=steps_per_epoch, train_loss=None)

            for step in range(steps_per_epoch):
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Training cancelled at step %d", self.global_step,
                        extra={"tag": "TRAIN"},)
                    break
                if pause_event is not None and pause_event.is_set():
                    logger.info("Training paused at step %d — waiting for resume", self.global_step,
                        extra={"tag": "TRAIN"},)
                    while pause_event.is_set():
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        time.sleep(0.5)
                    if cancel_event is not None and cancel_event.is_set():
                        break

                metrics = self.train_step()
                train_loss += metrics["loss"]
                self.global_step += 1

                if is_main and self.global_step % self.config.log_interval == 0:
                    lr = (
                        self.scheduler.get_last_lr()[0]
                        if self.scheduler
                        else self.config.learning_rate
                    )
                    logger.info(
                        f"Step {self.global_step} | Loss: {metrics['loss']:.4f} | LR: {lr:.2e}",
                        extra={"tag": "TRAIN"},
                    )
                    if self._experiment_tracker is not None:
                        self._experiment_tracker.log_metrics(
                            {
                                "train/loss": float(metrics["loss"]),
                                "train/learning_rate": float(lr),
                            },
                            step=int(self.global_step),
                        )

                if is_main and on_progress:
                    # Emit progress: every step for first 20 steps (smooth start),
                    # then every 5 steps (smooth chart without flooding SSE)
                    emit_interval = 1 if self.global_step <= 20 else 5
                    if self.global_step == 1 or self.global_step % emit_interval == 0:
                        _emit_progress(
                            steps_per_epoch=steps_per_epoch, train_loss=float(metrics["loss"])
                        )

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    if is_main:
                        logger.info(
                            f"Eval | Loss: {eval_metrics['eval_loss']:.4f} | "
                            f"PPL: {eval_metrics['eval_ppl']:.2f}",
                            extra={"tag": "TRAIN"},
                        )
                        if self._experiment_tracker is not None:
                            self._experiment_tracker.log_metrics(
                                {
                                    "eval/loss": float(eval_metrics["eval_loss"]),
                                    "eval/perplexity": float(eval_metrics["eval_ppl"]),
                                },
                                step=int(self.global_step),
                            )

                        if eval_metrics["eval_loss"] < self._best_val_loss:
                            self._best_val_loss = eval_metrics["eval_loss"]
                            self._patience_counter = 0
                            self.save_checkpoint(eval_metrics)
                            self._best_model_path = self._last_checkpoint_path
                        else:
                            self._patience_counter += 1

                        # Early stopping
                        if (
                            self.config.early_stopping_patience > 0
                            and self._patience_counter >= self.config.early_stopping_patience
                        ):
                            logger.info(
                                "Early stopping: no improvement for %d evals",
                                self._patience_counter,
                                extra={"tag": "TRAIN"},
                            )
                            self._early_stopped = True
                            self._is_training = False
                            if on_progress:
                                _emit_progress(
                                    steps_per_epoch=steps_per_epoch,
                                    train_loss=float(metrics["loss"]),
                                    eval_loss=float(eval_metrics["eval_loss"]),
                                    done=True,
                                    done_reason=f"early_stopping:{self._patience_counter}",
                                )
                            break

                    if is_main and on_progress:
                        _emit_progress(
                            steps_per_epoch=steps_per_epoch,
                            train_loss=float(metrics["loss"]),
                            eval_loss=float(eval_metrics["eval_loss"]),
                        )

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint({"loss": metrics["loss"]})

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        if is_main and not self._early_stopped:
            self.save_checkpoint({"loss": 0.0}, is_final=True)
            if self._experiment_tracker is not None:
                self._experiment_tracker.log_metrics(
                    {
                        "train/best_eval_loss": float(self._best_val_loss),
                        "train/final_step": float(self.global_step),
                    },
                    step=int(self.global_step),
                )

        self._is_training = False

        # final_loss: prefer best eval loss, fall back to last train loss
        final_loss = self._best_val_loss
        if final_loss is None or (isinstance(final_loss, float) and final_loss == float("inf")):
            final_loss = self._last_train_loss
        checkpoint_name = ""
        model_path = ""
        # Prefer best model path (set on eval improvement) over last checkpoint
        best = self._best_model_path or self._last_checkpoint_path
        if best:
            p = Path(best)
            checkpoint_name = p.name
            model_path = str(p)

        return TrainResult(
            success=True,
            best_eval_loss=self._best_val_loss,
            global_step=self.global_step,
            final_loss=final_loss,
            total_steps=self.global_step,
            epochs_completed=self.current_epoch + 1,
            model_path=self._best_model_path or model_path,
            checkpoint_name=checkpoint_name,
        )

    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None, is_final: bool = False):
        """Save a checkpoint in .soul format with vocab."""
        metrics = metrics or {"loss": 0.0}
        chars_list: Optional[List[str]] = None
        if self.itos is not None:
            try:
                chars_list = [self.itos[i] for i in range(self.vocab_size)]
            except (KeyError, TypeError):
                chars_list = None

        checkpoint_dir = Path(self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else "models/auto-training")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive checkpoint name from dataset
        import time
        timestamp = int(time.time())
        soul_name = getattr(self, 'soul_name', 'assistant')

        # Extract dataset name from path for checkpoint name
        data_path = getattr(self, 'data_path', '')
        if data_path:
            # e.g. "/Users/mac/sloughGPT/datasets/python_flask/corpus.jsonl" -> "python_flask"
            ds_name = Path(data_path).parent.name
        else:
            ds_name = soul_name

        checkpoint_path = checkpoint_dir / f"{ds_name}_{timestamp}"

        # Save in .soul format with vocab
        self.save(str(checkpoint_path), format="sou",
                  stoi=self.stoi, itos=self.itos, chars=chars_list)
        self._last_checkpoint_path = str(checkpoint_path)

    def save(self, path: str, format: str = "sou", stoi=None, itos=None, chars=None):
        """Save model in specified format."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        metadata = {
            "vocab_size": self.vocab_size,
            "stoi": stoi or self.stoi,
            "itos": itos or self.itos,
            "config": {
                "n_embed": self.config.n_embed,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "block_size": self.config.block_size,
                "model_type": "sloughgpt",
            },
            "training_dataset": self.data_path,
            "epochs_trained": self.config.epochs,
            "final_val_loss": self._best_val_loss,
        }

        if format == "sou":
            from domains.inference import create_soul_profile, save_soul

            soul = create_soul_profile(
                name=self.soul_name,
                base_model="sloughgpt",
                training_dataset=self.data_path,
                epochs_trained=self.config.epochs,
                final_val_loss=self._best_val_loss,
                lineage="sloughgpt",
                tags=["sloughgpt", "trained", "soul"],
            )
            # Bake vocab into soul metadata so checkpoint is self-contained
            _stoi = stoi or self.stoi
            _itos = itos or self.itos
            if _stoi is not None:
                soul.metadata["stoi"] = _stoi
            if _itos is not None:
                soul.metadata["itos"] = _itos
            if chars is not None:
                soul.metadata["chars"] = chars
            elif _itos is not None:
                soul.metadata["chars"] = [_itos[i] for i in range(len(_stoi))]
            soul.metadata["vocab_size"] = self.vocab_size
            soul.metadata["config"] = metadata["config"]

            # Embed training state so .soul is fully self-contained
            soul.metadata["training_state"] = _build_training_state_metadata(
                optimizer=getattr(self, "optimizer", None),
                scheduler=getattr(self, "scheduler", None),
                step=getattr(self, "_step", 0),
                epoch=getattr(self, "_epoch", 0),
                accumulation_step=getattr(self, "accumulation_step", 0),
                params=list(self.model.parameters()) if hasattr(self.model, "parameters") else None,
            )

            output_path = path + ".soul"
            save_soul(self.model, output_path, soul_profile=soul)
        elif format == "safetensors":
            from domains.training.export import export_to_safetensors

            output_path = path + ".safetensors"
            export_to_safetensors(self.model, output_path, metadata)
        else:
            from domains.training.export import export_to_safetensors

            output_path = path + ".safetensors"
            export_to_safetensors(self.model, output_path, metadata)

        logger.info("Model saved to %s (%s)", output_path, format,
            extra={"tag": "TRAIN"},)

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
        """Generate text."""
        self.model.eval()
        idx = np.array([[self.stoi.get(c, 0) for c in prompt]], dtype=np.int64)
        out = self.model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)

        text = "".join([self.itos.get(int(i), "?") for i in out.data[0]])
        return text


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for standalone training."""
    import argparse

    _epilog = (
        "step_*.npz in --checkpoint-dir includes stoi/itos/chars for char-LM eval. "
        "See docs/policies/CONTRIBUTING.md (Checkpoint vocabulary)."
    )
    parser = argparse.ArgumentParser(
        description="SloughGPT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog,
    )
    parser.add_argument("--data", default="datasets/shakespeare/input.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-embed", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--save-best-only", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from this checkpoint (.soul, .npz, or legacy .pt)",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from newest checkpoint in --checkpoint-dir",
    )
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if args.resume and args.resume_latest:
        parser.error("use either --resume PATH or --resume-latest, not both")

    trainer = SloughGPTTrainer(
        data_path=args.data,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        save_best_only=args.save_best_only,
        max_checkpoints=args.max_checkpoints,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=float(args.lora_alpha),
    )

    if args.resume_latest:
        trainer.train(resume=True, resume_path=None)
    elif args.resume:
        trainer.train(resume=True, resume_path=args.resume)
    else:
        trainer.train()
    print("\n=== Generated Text ===")
    print(trainer.generate("First"))


if __name__ == "__main__":
    main()
