#!/usr/bin/env python3
"""
SloughGPT Training Pipeline (SloNet-native)

Trains :class:`domains.models.SloughGPTModel` on pure NumPy via SloNet. There is
no external framework dependency anywhere in the training path: the optimizer is
``SloAdamW`` (decoupled weight decay), scheduling is
``domains.training.lr_schedulers``, and checkpoints are ``.soul``
(self-contained weights + vocab + training state) with a ``.npz`` fallback.

Features:
- Gradient accumulation
- EMA-smoothed loss reporting (honest curve, tracks raw loss both ways)
- Automatic ``.soul`` checkpointing with rotation
- LoRA fine-tuning (optional, flag-gated)
- Learning rate scheduling
- Resume from ``.soul`` / ``.npz``
- Cancel / pause events + progress callback (API / CLI integration)

Eval semantics: ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from domains.training.tracking import ExperimentTracker

try:
    from domains.models import SloughGPTModel
except (ImportError, ModuleNotFoundError):  # pragma: no cover (domains.models always importable)
    SloughGPTModel = None  # type: ignore[assignment,misc]
from domains.training.checkpoint_utils import extract_state_dict, normalize_raw_checkpoint
from domains.training.lora import LoRAConfig, apply_lora_to_model
from domains.training.quality_scorer import compute_data_quality
from domains.training.slonet import load_checkpoint_npz
from domains.training.trainer_protocol import TrainResult

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


def prepare_data(data_path, block_size=128, tokenizer=None):
    """Prepare training data from a text file or multiple datasets with ratios.

    Args:
        data_path: path to a text file, a dataset name (resolved against
            ``datasets/<name>/input.txt``), a list of dataset names, or a list
            of ``(name, ratio)`` tuples.
        block_size: context window (used for logging only).
        tokenizer: optional SloBPE-compatible tokenizer (e.g. a trained
            TokenTree). When provided, the corpus is tokenized with it and the
            returned vocabulary is the tokenizer's, not the raw char set.

    Returns:
        (data, vocab_size, stoi, itos) — ``data`` is ``np.int64`` token ids.
    """
    if isinstance(data_path, list) and data_path and isinstance(data_path[0], tuple):
        datasets_with_ratios = data_path
        all_texts = []
        total_len = 0

        for ds_name, ratio in datasets_with_ratios:
            path = Path("data") / ds_name / "input.txt"
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
            path = Path("data") / ds_name / "input.txt"
            if path.exists():
                texts.append(path.read_text(encoding="utf-8"))
            else:
                logger.warning("dataset %s not found, skipping", ds_name,
                    extra={"tag": "TRAIN"},)
        text = "".join(texts)

    else:
        if data_path is None:
            raise FileNotFoundError(
                "No data_path provided and no default dataset found"
            )
        path = Path(data_path)
        if not path.is_file():
            # Try as a dataset name under data/
            alt = Path("data") / data_path / "input.txt"
            if alt.is_file():
                path = alt
            else:
                raise FileNotFoundError(
                    f"Data file not found: '{data_path}' (tried '{path}' and '{alt}')"
                )
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    if tokenizer is not None:
        data = np.asarray(tokenizer.encode(text), dtype=np.int64)
        vocab_size = tokenizer.vocab_size
        stoi = dict(tokenizer.stoi)
        itos = dict(tokenizer.itos)
        logger.info("Data: %d tokens, vocab %d (tokenized)", len(data), vocab_size,
            extra={"tag": "TRAIN"},)
        return data, vocab_size, stoi, itos

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
    n_embed: int = 64
    n_layer: int = 2
    n_head: int = 4
    block_size: int = 64
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # Checkpointing - uses .soul format
    checkpoint_dir: str = "models/auto-training"
    checkpoint_interval: int = 500
    save_best_only: bool = False
    max_checkpoints: int = 5

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Logging / eval
    log_interval: int = 10
    eval_interval: int = 100

    # Early stopping (0 = disabled; stop if no improvement for N evals)
    early_stopping_patience: int = 5

    # Device — SloNet training is pure numpy and always runs on the CPU.
    device: str = "cpu"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cpu"

        # Validate hyperparameters
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_embed < 1:
            raise ValueError(f"n_embed must be >= 1, got {self.n_embed}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}")
        if self.block_size < 8:
            raise ValueError(f"block_size must be >= 8, got {self.block_size}")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be 0-1, got {self.dropout}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")
        if self.n_head > self.n_embed:
            raise ValueError(f"n_head ({self.n_head}) must be <= n_embed ({self.n_embed})")
        if self.block_size > self.n_embed * 4:
            import warnings
            warnings.warn(
                f"block_size ({self.block_size}) > 4*n_embed ({self.n_embed*4}) may cause instability",
                UserWarning,
                stacklevel=2,
            )


# =============================================================================
# Training State Serialization (for .soul metadata embedding)
# =============================================================================


def _make_json_safe(obj):
    """Recursively convert numpy arrays to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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
        result["completed_epochs"] = parsed.get("completed_epochs")
        if "optimizer" in parsed:
            result["optimizer_state_dict"] = parsed["optimizer"]
        if "scheduler" in parsed:
            result["scheduler_state_dict"] = parsed["scheduler"]

    return result


def _build_training_state_metadata(
    optimizer=None, scheduler=None, step=0, epoch=0, completed_epochs=0,
    accumulation_step=0, params=None, include_optimizer_state=True,
    initial_lr=None,
) -> dict:
    """Build a JSON-serializable dict of training state for embedding in .soul metadata.

    Args:
        optimizer: SloAdamW / SloAdam / SloSGD (or None).
        scheduler: SloLRScheduler (or None).
        step: Current global training step.
        epoch: Current epoch.
        completed_epochs: Number of fully completed epochs so far (honest
            epochs_trained claim; a mid-epoch save reports the completed count).
        accumulation_step: Current gradient accumulation step.
        params: List of model parameters (for SloAdamW/SloAdam/SloSGD name-based
            state).
        include_optimizer_state: If False, the bulky per-parameter momentum
            buffers (``optimizer["state"]``) are dropped; optimizer
            hyperparameters are still embedded so a resume can rebuild a
            fresh-momentum optimizer. Keeps checkpoint metadata small.
        initial_lr: The learning rate from config (before scheduler decay).
            When provided, this is written into ``optimizer.hyperparameters.lr``
            instead of the scheduler-decayed value — the decayed lr is useless
            for metadata since it reflects the final (often near-zero) value,
            not the lr used during training.

    Returns:
        Dict ready to embed in soul.metadata["training_state"].
    """
    state: dict = {
        "step": step, "epoch": epoch, "completed_epochs": completed_epochs,
        "accumulation_step": accumulation_step,
    }
    if optimizer is not None:
        try:
            opt_state = optimizer.state_dict(params=params) if params is not None else optimizer.state_dict()
            if not include_optimizer_state and isinstance(opt_state, dict):
                # Momentum buffers dwarf the weights themselves; hyperparameters
                # alone let resume recreate a working (fresh-momentum) optimizer.
                opt_state = dict(opt_state)
                opt_state.pop("state", None)
            # Override scheduler-decayed lr with the config lr so metadata
            # records the lr used during training, not the near-zero tail.
            if initial_lr is not None and isinstance(opt_state, dict):
                hyper = opt_state.get("hyperparameters")
                if isinstance(hyper, dict):
                    hyper["lr"] = initial_lr
            state["optimizer"] = _make_json_safe(opt_state)
        except Exception as e:
            logger.warning("train_pipeline: optimizer state serialization failed", extra={
                "error": str(e),
            })
    if scheduler is not None:
        try:
            sched_state = scheduler.state_dict()
            # Persist initial_lr in scheduler state so resume can rebuild
            # the warmup schedule from the correct starting point.
            if initial_lr is not None and isinstance(sched_state, dict):
                sched_state["initial_lr"] = initial_lr
            state["scheduler"] = _make_json_safe(sched_state)
        except Exception as e:
            logger.warning("train_pipeline: scheduler state serialization failed", extra={
                "error": str(e),
            })
    return state


def _parse_training_state_metadata(metadata: dict) -> dict:
    """Extract training state from .soul metadata (inverse of _build_...).

    Converts nested lists back to numpy arrays where appropriate so that
    optimizer.load_state_dict() can consume them.

    Returns:
        Dict with keys: step, epoch, completed_epochs (optional),
        accumulation_step, optimizer (optional), scheduler (optional).
    """

    def _to_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj, dtype=np.float64)
        return obj

    raw = metadata.get("training_state", {})
    result: dict = {
        "step": raw.get("step", 0),
        "epoch": raw.get("epoch", 0),
        "accumulation_step": raw.get("accumulation_step", 0),
    }
    if "completed_epochs" in raw:
        result["completed_epochs"] = raw.get("completed_epochs")
    opt_raw = raw.get("optimizer")
    if isinstance(opt_raw, dict):
        opt = dict(opt_raw)
        opt.get("hyperparameters", {})
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
    """Reads checkpoints written by :class:`SloughGPTTrainer`.

    A resume-only manager: ``SloughGPTTrainer.save`` / ``save_checkpoint`` are
    the single writers (periodic checkpoints embed optimizer momentum for exact
    resume; final checkpoints strip it for a compact artifact), while this class
    locates and loads the most recent checkpoint for crash-resume.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_from_path(path: str) -> Optional[Dict[str, Any]]:
        """Load a training checkpoint from an explicit path (.soul or .npz).

        Args:
            path: Absolute or relative path to a ``.soul`` or ``.npz`` checkpoint.

        Returns:
            Loaded checkpoint bundle (``model_state_dict`` plus metadata) or
            ``None`` when the file is missing or uses an unsupported format.

        Side effects:
            - logs a warning when the file is missing or unsupported
        """
        p = Path(path).expanduser()
        if not p.is_file():
            logger.warning("Checkpoint file not found: %s", p,
                extra={"tag": "TRAIN"},)
            return None
        if p.suffix == ".soul":
            return _load_soul_checkpoint(str(p))
        if p.suffix == ".npz":
            return load_checkpoint_npz(str(p))
        logger.warning("Unsupported checkpoint format: %s (use .soul or .npz)", p,
            extra={"tag": "TRAIN"},)
        return None

    @staticmethod
    def is_resumable(path: str) -> bool:
        """Cheap, parse-free check that ``path`` is a resumable checkpoint.

        Unlike :meth:`load_from_path`, this never reads the file contents — it
        only verifies the path exists and uses a supported extension, so it can
        be used to resolve a resume path without paying a full load.

        Returns:
            True when ``path`` is an existing ``.soul`` or ``.npz`` file,
            False otherwise.
        """
        p = Path(path).expanduser()
        return p.is_file() and p.suffix in (".soul", ".npz")

    def _candidates_newest_first(self) -> List[Path]:
        """Checkpoint files under ``checkpoint_dir``, newest modification first.

        In-progress temp artifacts (``*.tmp`` / ``*.tmp.npz``) written during an
        atomic save are excluded, so an orphaned temp (from a crash between the
        write and the rename) never surfaces as a resume candidate.
        """
        return sorted(
            [
                p
                for p in list(self.checkpoint_dir.glob("*.soul"))
                + list(self.checkpoint_dir.glob("*.npz"))
                if not (p.name.endswith(".tmp") or p.name.endswith(".tmp.npz"))
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def latest_path(self) -> Optional[str]:
        """Return the path of the most recently modified checkpoint, if any.

        This is a pure path lookup — it does not validate the file's contents.
        Use :meth:`latest_valid_path` to skip unreadable (e.g. partially
        written) checkpoints.

        Returns:
            Absolute path to the newest ``.soul`` or ``.npz`` file under
            ``checkpoint_dir``, or ``None`` when the directory holds no
            checkpoints.
        """
        candidates = self._candidates_newest_first()
        return str(candidates[0]) if candidates else None

    def load_latest_with_path(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Load the newest readable checkpoint and return its path and bundle.

        Single-load primitive: iterates checkpoints newest-first and returns
        the first ``(path, bundle)`` that parses successfully, skipping
        unreadable (e.g. partially written) files. Prefer this over calling
        :meth:`latest_valid_path` then :meth:`load_latest`, which would load
        the same checkpoint twice.

        Returns:
            ``(path, bundle)`` for the newest loadable checkpoint, or
            ``(None, None)`` when no checkpoint in the directory loads.

        Side effects:
            - logs a warning per skipped unreadable checkpoint
        """
        for p in self._candidates_newest_first():
            try:
                bundle = CheckpointManager.load_from_path(str(p))
            except Exception as exc:
                logger.warning("Skipping unreadable checkpoint %s: %s", p, exc,
                    extra={"tag": "TRAIN"},)
                continue
            if bundle is not None:
                return str(p), bundle
        return None, None

    def latest_valid_path(self) -> Optional[str]:
        """Path of the most recent checkpoint that actually loads.

        Iterates checkpoints newest-first and returns the first one that
        parses successfully, skipping corrupt or partially-written files (each
        skipped file is logged). A crash mid-write can leave a ``.soul`` as the
        newest file; this method falls back to the previous good checkpoint
        instead of failing.

        Returns:
            Path of the newest loadable ``.soul``/``.npz`` under
            ``checkpoint_dir``, or ``None`` when no checkpoint loads.

        Side effects:
            - logs a warning per skipped unreadable checkpoint
        """
        return self.load_latest_with_path()[0]

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent readable ``.soul`` or ``.npz`` checkpoint.

        Unlike :meth:`latest_path`, this skips unreadable checkpoints and
        returns the newest one that parses (crash-resilient resume).

        Returns:
            The loaded checkpoint bundle, or ``None`` when no checkpoint in
            the directory loads.

        Side effects:
            - logs a warning per skipped unreadable checkpoint
        """
        return self.load_latest_with_path()[1]


# =============================================================================
# Main Trainer
# =============================================================================


class SloughGPTTrainer:
    """
    Unified trainer for SloughGPTModel (pure NumPy / SloNet).

    Satisfies :class:`domains.training.trainer_protocol.TrainerProtocol` structurally (``train()``).

    Features:
    - Gradient accumulation
    - Automatic checkpointing (``.soul`` includes ``stoi``/``itos``/``chars`` for eval)
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
        n_embed: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        block_size: int = 64,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 3e-4,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
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
        personality: Optional[dict] = None,
        log_interval: int = 10,
        eval_interval: int = 100,
        experiment_tracker: Optional["ExperimentTracker"] = None,
        tokenizer: Optional[Any] = None,
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
        self._personality = personality
        self.tokenizer = tokenizer
        self._experiment_tracker = experiment_tracker
        self._best_val_loss = float("inf")
        self._train_loss_at_best = 0.0
        self._ema_loss = None  # exponential moving average of train loss
        self._ema_alpha = 0.3  # smoothing factor (0=ignore new, 1=no smoothing)
        self._last_checkpoint_path = None  # path of last saved checkpoint
        self._last_train_loss = None  # last raw train loss for fallback
        self._patience_counter = 0  # early stopping: evals since last improvement
        self._best_model_path = None  # path to best checkpoint
        self._best_checkpoint_loss = float("inf")  # best train loss for save_best_only
        self._early_stopped = False  # True if early stopping triggered
        self._quality_scores: List[float] = []  # rolling quality scores of training data
        self._avg_quality: Optional[float] = None  # running average quality

        self.device = self._setup_device()
        self.config.device = self.device

        logger.info("Using device: %s", self.device,
            extra={"tag": "TRAIN"},)

        # Prepare data — prefer corpus-derived vocab unless caller sets ``vocab_size`` (legacy path)
        # or supplies a full ``TrainerConfig`` (advanced; caller must match data).
        self.data, data_vocab_size, self.stoi, self.itos = prepare_data(
            data_path, self.config.block_size, tokenizer=tokenizer
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

        # Compute data quality metrics
        try:
            raw_text = "".join(self.itos.get(int(i), "") for i in self.data[:min(50000, len(self.data))])
            self._data_quality = compute_data_quality(raw_text)
            self._avg_quality = self._data_quality.get("avg_quality")
            logger.info("Data quality: avg=%.2f repetition=%.2f diversity=%.2f language=%.2f",
                self._data_quality["avg_quality"], self._data_quality["repetition_rate"],
                self._data_quality["diversity"], self._data_quality["language_quality"],
                extra={"tag": "TRAIN"})
        except Exception as e:
            logger.warning("Data quality computation failed, using defaults: %s", e)
            self._data_quality = {"avg_quality": 0.0, "repetition_rate": 0.0, "diversity": 0.0, "language_quality": 0.0}

        # Create model
        self._create_model()

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup checkpointing (resume-only manager; save_checkpoint is the writer)
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.accumulation_step = 0

        logger.info("Train: %d, Val: %d", len(self.train_data), len(self.val_data),
            extra={"tag": "TRAIN"},)

    def _setup_device(self) -> str:
        """SloNet training is pure numpy and always runs on the CPU."""
        return "cpu"

    def _create_model(self):
        """Create and setup the model (optionally wrapped with LoRA)."""
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

    def _create_optimizer(self):
        """Create SloAdamW optimizer with decoupled weight decay."""
        from domains.training.slonet import SloAdamW

        return SloAdamW(
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with adaptive defaults."""
        from domains.training.lr_schedulers import create_scheduler

        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = (
                len(self.train_data) // self.config.block_size // self.config.batch_size
            )
            total_steps = steps_per_epoch * self.config.epochs

        # Adaptive learning rate based on model size
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count < 1_000_000:  # < 1M params: small model
            max_lr = self.config.learning_rate
            min_lr = self.config.min_lr
        elif param_count < 10_000_000:  # 1M-10M params: medium model
            max_lr = self.config.learning_rate * 0.5
            min_lr = self.config.min_lr * 0.5
        else:  # > 10M params: large model
            max_lr = self.config.learning_rate * 0.25
            min_lr = self.config.min_lr * 0.25

        return create_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            total_steps=total_steps,
            warmup_steps=self.config.warmup_steps,
            min_lr=min_lr,
            max_lr=max_lr,
        )

    @property
    def training_model(self):
        """Get the model for training (single-process CPU; always the raw model)."""
        return self.model

    def get_batch(self, split: str = "train") -> tuple:
        """Get a batch of data as numpy arrays.

        Uses vectorized advanced indexing instead of Python-level loops
        for O(1) batch construction regardless of batch_size.
        """
        data = self.train_data if split == "train" else self.val_data
        batch_size = self.config.batch_size
        block_size = self.config.block_size

        idx = np.random.randint(0, len(data) - block_size, size=batch_size)
        offsets = np.arange(block_size)
        x = data[idx[:, None] + offsets]
        y = data[idx[:, None] + offsets + 1]
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
        # EMA smoothing: true exponential moving average that may rise with
        # batch noise, so the reported curve tracks the raw loss honestly
        # instead of freezing at a one-way floor.
        if self._ema_loss is None:
            ema = raw_loss
        else:
            ema = self._ema_alpha * raw_loss + (1 - self._ema_alpha) * self._ema_loss
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

    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """Evaluate the model on the validation split.

        Uses 10 batches — sufficient for loss estimation while keeping the
        per-eval cost low.
        """
        import time as _time
        eval_start = _time.monotonic()

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
        elapsed_ms = (_time.monotonic() - eval_start) * 1000
        logger.info("train_pipeline: evaluate complete", extra={
            "eval_loss": round(avg_loss, 4),
            "eval_ppl": round(float(np.exp(avg_loss)), 2),
            "batches": num_batches,
            "elapsed_ms": round(elapsed_ms, 1),
        })
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
            if incomp is not None and (incomp.missing_keys or incomp.unexpected_keys):
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

    def _steps_per_epoch(self) -> int:
        """Optimizer steps in one full pass over the training data (0 when no data)."""
        if getattr(self, "train_data", None) is None:
            return 0
        return len(self.train_data) // self.config.block_size // self.config.batch_size

    @property
    def _completed_epochs(self) -> int:
        """Fully completed epochs of training, derived from the absolute step count.

        Computed as ``global_step // steps_per_epoch`` so the value stays honest
        across mid-epoch checkpoints and resume boundaries (the trainer restarts
        an epoch's inner loop at the resumed step, so per-process epoch counting
        would drift; the absolute step count does not).
        """
        spe = self._steps_per_epoch()
        return self.global_step // spe if spe else 0

    def _training_elapsed(self) -> float:
        """Seconds since training start (0 before the loop begins)."""
        start = getattr(self, "_training_start_time", None)
        if start is None:
            return 0.0
        return max(0.0, time.time() - start)

    def _steps_per_sec(self) -> float:
        """Rolling optimizer steps per second based on whole-run elapsed time."""
        elapsed = self._training_elapsed()
        if elapsed <= 0 or self.global_step <= 0:
            return 0.0
        return self.global_step / elapsed

    def _eta_seconds(self, steps_per_epoch: int) -> Optional[float]:
        """Estimated seconds until the final step, or None when speed is 0."""
        total = self._progress_denominator(steps_per_epoch)
        remaining = max(0, total - self.global_step)
        sps = self._steps_per_sec()
        if sps <= 0 or remaining <= 0:
            return None
        return remaining / sps

    @staticmethod
    def _format_eta(seconds: Optional[float]) -> str:
        """Render a seconds value as a human ETA string (``--`` when unknown)."""
        if seconds is None or seconds < 0:
            return "--"
        total_s = int(seconds)
        m, s = divmod(total_s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def train(
        self,
        resume: bool = False,
        resume_path: Optional[str] = None,
        resume_checkpoint: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            resume: If True, load a checkpoint and continue training. The
                checkpoint comes from ``resume_checkpoint`` when provided,
                otherwise ``resume_path``, otherwise the latest readable
                checkpoint under ``checkpoint_dir``. An explicit ``resume_path``
                that cannot be loaded raises :class:`ValueError`; an implicit
                resume (no path) with no checkpoint logs a warning and starts
                fresh.
            resume_path: Optional checkpoint path (.soul or .npz). Accepts full
                ``CheckpointManager`` bundles (model + optimizer + scheduler + step/epoch) and
                **weights-only** bundles (``model_state_dict`` or flat tensors) as normalized
                by :func:`domains.training.checkpoint_utils.normalize_raw_checkpoint`. Optimizer
                and scheduler load are best-effort.
            resume_checkpoint: Optional pre-loaded checkpoint bundle to restore
                from, bypassing all disk I/O. Takes precedence over
                ``resume_path`` and ``latest``. Callers that already resolved
                and loaded a checkpoint (e.g. crash recovery, which uses
                :meth:`CheckpointManager.load_latest_with_path`) hand it here so
                the checkpoint is loaded exactly once.
            on_progress: Optional callback invoked on a throttled schedule with a dict
                containing at least: ``global_step``, ``epoch`` (1-based), ``epochs``,
                ``steps_per_epoch``, ``progress_percent`` (0--99 while running),
                ``train_loss`` (last batch), optional ``eval_loss``, ``learning_rate``.
            cancel_event: Optional threading.Event — if set, training stops cooperatively.
            pause_event: Optional threading.Event — if set, training loop sleeps until cleared.

        Raises:
            ValueError: When ``resume`` is True and an explicit ``resume_path``
                is missing, uses an unsupported format, or is unreadable
                (corrupt/truncated). The path is never silently replaced by a
                different checkpoint. Also raised when ``resume_checkpoint`` is
                provided without ``resume=True`` — a pre-loaded bundle must
                never be silently discarded.
        """
        if resume_checkpoint is not None and not isinstance(resume_checkpoint, dict):
            raise ValueError(
                "resume_checkpoint must be a checkpoint bundle dict (as returned by "
                "load_from_path / load_latest_with_path), got "
                f"{type(resume_checkpoint).__name__}"
            )
        if resume_checkpoint is not None and not resume:
            raise ValueError(
                "resume_checkpoint requires resume=True: a pre-loaded bundle is "
                "only consumed when resuming training"
            )
        if resume:
            if resume_checkpoint is not None:
                checkpoint = resume_checkpoint
            elif resume_path:
                try:
                    checkpoint = CheckpointManager.load_from_path(resume_path)
                except Exception as exc:
                    raise ValueError(
                        f"Cannot resume from '{resume_path}': checkpoint is unreadable ({exc})"
                    ) from exc
                if checkpoint is None:
                    raise ValueError(
                        f"Cannot resume from '{resume_path}': checkpoint missing or unsupported "
                        "(use a .soul or .npz file)"
                    )
            else:
                checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self._restore_from_checkpoint_bundle(checkpoint)
            else:
                logger.warning(
                    "Resume requested but no checkpoint found under %s — starting fresh",
                    self.checkpoint_manager.checkpoint_dir,
                    extra={"tag": "TRAIN"},
                )

        logger.info("Training config: %s", self.config,
            extra={"tag": "TRAIN"},)
        logger.info("Total parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}",
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
            if on_progress is None:
                return
            denom = self._progress_denominator(steps_per_epoch)
            pct = 100 if done else min(99, int(100 * self.global_step / denom))
            lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
            sps = self._steps_per_sec()
            eta = None if sps <= 0 else max(0, int((denom - self.global_step) / sps))
            try:
                on_progress(
                    {
                        "global_step": int(self.global_step),
                        "epoch": int(self.current_epoch + 1),
                        "epochs": int(self.config.epochs),
                        "steps_per_epoch": int(steps_per_epoch),
                        "total_steps": int(denom),
                        "progress_percent": int(pct),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "learning_rate": float(lr),
                        "steps_per_sec": round(sps, 2),
                        "eta_s": eta,
                        "elapsed_s": round(self._training_elapsed(), 1),
                        "done": done,
                        "done_reason": done_reason,
                        "avg_quality": self._avg_quality,
                    }
                )
            except Exception:
                logger.exception("on_progress callback failed", extra={"tag": "TRAIN"})

        self._is_training = True
        self._training_start_time = time.time()

        # Record dashboard event
        try:
            from domains.infrastructure.event_buffer import get_event_buffer
            epochs = self.config.epochs
            max_steps = self.config.max_steps or "unlimited"
            get_event_buffer().record("TRAIN", f"started epochs={epochs} max_steps={max_steps}")
        except Exception as exc:
            logger.debug("Failed to record training start event: %s", exc)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            if not self._is_training:
                logger.info("Training stopped at epoch %d", epoch,
                    extra={"tag": "TRAIN"},)
                break

            logger.info("Epoch %d/%d", epoch + 1, self.config.epochs,
                extra={"tag": "TRAIN"},)

            model = self.training_model
            model.train()

            train_loss = 0.0
            steps_per_epoch = (
                len(self.train_data) // self.config.block_size // self.config.batch_size
            )

            if steps_per_epoch == 0:
                raise ValueError(
                    f"Training data too small for block_size={self.config.block_size} "
                    f"and batch_size={self.config.batch_size}: only {len(self.train_data)} "
                    f"samples available, need at least {self.config.block_size * self.config.batch_size}"
                )

            if on_progress and steps_per_epoch > 0:
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

                if self.global_step % self.config.log_interval == 0:
                    lr = (
                        self.scheduler.get_last_lr()[0]
                        if self.scheduler
                        else self.config.learning_rate
                    )
                    denom = self._progress_denominator(steps_per_epoch)
                    pct = min(100, int(100 * self.global_step / denom))
                    sps = self._steps_per_sec()
                    eta = self._eta_seconds(steps_per_epoch)
                    logger.info(
                        "Step %d/%d | Loss: %.4f | LR: %.2e | %d%% | %.1f steps/s | ETA %s",
                        self.global_step, denom, metrics['loss'], lr, pct, sps, self._format_eta(eta),
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

                if on_progress:
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
                    logger.info(
                        "Eval | Loss: %.4f | PPL: %.2f",
                        eval_metrics['eval_loss'], eval_metrics['eval_ppl'],
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

                    # Train loss plateau detection (independent of eval loss)
                    if hasattr(self, '_recent_train_losses'):
                        self._recent_train_losses.append(float(metrics["loss"]))
                        if len(self._recent_train_losses) > 10:
                            self._recent_train_losses.pop(0)
                        # Check if train loss has plateaued (std dev < 0.001 over last 10 evals)
                        if len(self._recent_train_losses) >= 5:
                            import numpy as _np
                            _std = float(_np.std(self._recent_train_losses[-10:]))
                            if _std < 0.001:
                                logger.info(
                                    "Train loss plateau detected (std=%.6f over last %d evals)",
                                    _std, min(len(self._recent_train_losses), 10),
                                    extra={"tag": "TRAIN"},
                                )
                    else:
                        self._recent_train_losses = [float(metrics["loss"])]

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

                    if on_progress:
                        _emit_progress(
                            steps_per_epoch=steps_per_epoch,
                            train_loss=float(metrics["loss"]),
                            eval_loss=float(eval_metrics["eval_loss"]),
                        )

                # Checkpoint. With save_best_only, periodic (train-loss) saves
                # are skipped unless the loss is a new best; eval-improvement
                # and final saves are never gated. checkpoint_interval=0
                # disables periodic checkpoints entirely (final save remains).
                if self.config.checkpoint_interval and self.global_step % self.config.checkpoint_interval == 0:
                    if (
                        self.config.save_best_only
                        and metrics["loss"] >= self._best_checkpoint_loss
                    ):
                        logger.info(
                            "Skipping periodic checkpoint (save_best_only; loss %.4f not a new best)",
                            float(metrics["loss"]),
                            extra={"tag": "TRAIN"},
                        )
                    else:
                        self._best_checkpoint_loss = metrics["loss"]
                        self.save_checkpoint({"loss": metrics["loss"]})

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        if not self._early_stopped:
            # Report the last observed loss, never a fabricated value. The soul
            # profile's final_train_loss is re-derived from _last_train_loss.
            self.save_checkpoint({"loss": self._last_train_loss}, is_final=True)
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

        # Record dashboard event
        try:
            from domains.infrastructure.event_buffer import get_event_buffer
            final_loss_str = f"{final_loss:.4f}" if final_loss is not None else "n/a"
            get_event_buffer().record("TRAIN", f"completed step={self.global_step} loss={final_loss_str}")
        except Exception as exc:
            logger.debug("Failed to record training completion event: %s", exc)
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
            epochs_completed=self._completed_epochs,
            model_path=self._best_model_path or model_path,
            checkpoint_name=checkpoint_name,
            avg_quality=self._avg_quality,
            data_quality=getattr(self, '_data_quality', None),
        )

    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None, is_final: bool = False):
        """Save a checkpoint in ``.soul`` format with vocab.

        Periodic checkpoints (``is_final=False``) embed full optimizer state so
        a crashed run resumes exactly; the final checkpoint (``is_final=True``)
        drops momentum buffers — keeping the delivered artifact small, since it
        is intended for inference, not resume.

        Args:
            metrics: Optional training metrics to record (informational only;
                serialization reads tracked ``_last_train_loss`` /
                ``_best_val_loss``, never a fabricated placeholder).
            is_final: True on the last save of a run; prunes older checkpoints
                and strips optimizer momentum buffers from the artifact.

        Side effects:
            - writes ``<checkpoint_dir>/<dataset>_<timestamp>.soul``
            - prunes stale checkpoints via ``_prune_stale_checkpoints``

        Returns:
            None.
        """
        chars_list: Optional[List[str]] = None
        if self.itos is not None:
            try:
                chars_list = [self.itos[i] for i in range(self.vocab_size)]
            except (KeyError, TypeError):
                chars_list = None

        checkpoint_dir = Path(self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else "models/auto-training")  # pragma: no cover (TrainerConfig always has checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive checkpoint name from dataset
        timestamp = int(time.time())
        soul_name = getattr(self, 'soul_name', 'assistant')

        # Extract dataset name from path for checkpoint name
        data_path = getattr(self, 'data_path', '')
        if data_path:
            # e.g. "/Users/mac/sloughGPT/datasets/python_flask/corpus.jsonl" -> "python_flask"
            ds_name = Path(data_path).parent.name
        else:  # pragma: no cover (trainer always has data_path)
            ds_name = soul_name

        checkpoint_path = checkpoint_dir / f"{ds_name}_{timestamp}"

        # Compute training duration
        start_t = getattr(self, '_training_start_time', None)
        training_duration = round(time.time() - start_t, 1) if start_t else None

        # Save in .soul format with vocab; periodic checkpoints keep optimizer
        # state (accurate resume), final artifact strips momentum buffers.
        self.save(str(checkpoint_path),
                  stoi=self.stoi, itos=self.itos, chars=chars_list,
                  training_duration=training_duration,
                  include_optimizer_state=not is_final,
                  avg_quality=self._avg_quality)
        self._last_checkpoint_path = str(checkpoint_path) + ".soul"
        self._prune_stale_checkpoints(keep_final=is_final)

    def _prune_stale_checkpoints(self, keep_final: bool = False) -> None:
        """Delete stale ``.soul`` checkpoints so the directory never accumulates files.

        During training only the ``max_checkpoints`` newest files are kept, so a
        crashed run can still be resumed (``--resume-latest`` scans by mtime). On
        the final save all older checkpoints are removed, leaving a single model
        file.

        Args:
            keep_final: If True, keep only the newest checkpoint (the just-written
                final one); otherwise keep the ``max_checkpoints`` newest files.

        Side effects:
            - unlinks stale ``.soul`` files under ``config.checkpoint_dir``
            - rewires ``_best_model_path`` so it never points at a deleted file
              (on final saves it points at the final checkpoint; otherwise it is
              reset to None when the best checkpoint was pruned by the window)

        Returns:
            None.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            return
        try:
            files = sorted(
                checkpoint_dir.glob("*.soul"),
                key=lambda p: (p.stat().st_mtime, p.name),
                reverse=True,
            )
        except OSError:
            return

        keep_count = 1 if keep_final else max(
            1, int(getattr(self.config, "max_checkpoints", 5))
        )
        keep = {str(p) for p in files[:keep_count]}
        for stale in files[keep_count:]:
            try:
                stale.unlink()
            except OSError:
                pass
            meta = Path(str(stale) + ".meta.json")
            try:
                if meta.exists():
                    meta.unlink()
            except OSError:
                pass

        if keep_final and self._last_checkpoint_path:
            # The just-written final checkpoint is the newest file (kept above);
            # rewire the best reference in case it pointed at a pruned checkpoint.
            self._best_model_path = self._last_checkpoint_path
        elif self._best_model_path and self._best_model_path not in keep:
            self._best_model_path = None

    def save(self, path: str, format: Optional[str] = None, stoi=None, itos=None, chars=None,
             training_duration=None, include_optimizer_state: bool = True, avg_quality: Optional[float] = None):
        """Save the model in ``.soul`` format (the only SloNet checkpoint format).

        Args:
            path: Output path without the extension; ``.soul`` is appended.
            format: DEPRECATED and ignored. Retained for backward compatibility;
                ``SloughGPTTrainer.save()`` always writes ``.soul`` regardless of
                this value. Passing a non-None value emits a
                ``DeprecationWarning``.
            stoi: String-to-index vocab map embedded in soul metadata so the
                checkpoint is self-contained for char-level inference.
            itos: Index-to-string vocab map embedded in soul metadata.
            chars: Ordered character list used to rebuild ``itos`` when
                ``itos`` alone is insufficient; derived from ``itos`` when not
                provided.
            training_duration: Training duration in seconds, embedded into
                metadata when provided.
            include_optimizer_state: If True, embed the full optimizer state
                (step, hyperparameters, and per-parameter momentum buffers) so
                a mid-training run resumes exactly. If False, only step and
                hyperparameters are embedded — momentum buffers are dropped,
                keeping the checkpoint small; a resume rebuilds the optimizer
                with fresh momentum. Pass False for final delivered artifacts
                that only need inference weights.

        Returns:
            None.

        Side effects:
            - Writes ``<path>.soul`` and its ``<path>.soul.meta.json``.
        """
        if format is not None:
            warnings.warn(
                f"The `format` parameter is deprecated and ignored — "
                f"SloughGPTTrainer.save() always writes .soul (got {format!r})",
                DeprecationWarning,
                stacklevel=2,
            )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        from domains.inference import create_soul_profile, save_soul
        from domains.inference.slo_format import PersonalityCore

        # Honest metadata: only claim a loss that was actually observed. A save
        # before any training step has neither a train loss nor an eval loss,
        # so both serialize as null rather than fabricated 0.0 values.
        final_train_loss = self._last_train_loss
        final_val_loss = (
            None if self._best_val_loss == float("inf") else self._best_val_loss
        )

        # Honest metadata: only claim what actually happened. A save before any
        # training step has no observed loss, so epochs serialize as 0 rather
        # than the config's target epoch count. Fully completed epochs come
        # from the absolute step count (global_step // steps_per_epoch), so a
        # mid-epoch stop reports the completed epochs, not the entered count.
        epochs_trained = self._completed_epochs

        soul = create_soul_profile(
            name=self.soul_name,
            base_model="sloughgpt",
            training_dataset=self.data_path,
            epochs_trained=epochs_trained,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            personality=PersonalityCore(**self._personality) if self._personality else None,
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
            try:
                soul.metadata["chars"] = [_itos[i] for i in range(len(_stoi))]
            except (KeyError, TypeError):
                pass
        soul.metadata["vocab_size"] = self.vocab_size
        soul.metadata["config"] = {
            "n_embed": self.config.n_embed,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "block_size": self.config.block_size,
            "model_type": "sloughgpt",
        }
        if training_duration is not None:
            soul.metadata["training_duration_s"] = training_duration
        if avg_quality is not None:
            soul.metadata["avg_quality"] = avg_quality

        # Embed the tokenizer (e.g. a trained TokenTree) so the .soul is fully
        # self-contained and inference can reproduce BPE-level encoding.
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "to_dict"):
            soul.metadata["tokenizer"] = {
                "type": "token_tree",
                "tree": tokenizer.to_dict(),
            }

        # Embed training state so the .soul is self-contained. Momentum buffers
        # are omitted when include_optimizer_state=False (final artifacts) —
        # they dwarf the weights and are not needed for inference; resume
        # rebuilds a fresh-momentum optimizer from the embedded hyperparameters.
        soul.metadata["training_state"] = _build_training_state_metadata(
            optimizer=getattr(self, "optimizer", None),
            scheduler=getattr(self, "scheduler", None),
            step=getattr(self, "global_step", getattr(self, "_step", 0)),
            epoch=getattr(self, "current_epoch", getattr(self, "_epoch", 0)),
            completed_epochs=getattr(self, "_completed_epochs", 0),
            accumulation_step=getattr(self, "accumulation_step", 0),
            params=list(self.model.parameters()) if hasattr(self.model, "parameters") else None,
            include_optimizer_state=include_optimizer_state,
            initial_lr=self.config.learning_rate,
        )

        output_path = path + ".soul"
        save_soul(self.model, output_path, soul_profile=soul)

        logger.info("Model saved to %s", output_path, extra={"tag": "TRAIN"})

        # Auto-compress checkpoint into pugqeep Points for efficient inference
        try:
            from domains.training.executor import compress_checkpoint
            result = compress_checkpoint(output_path, n_clusters=16)
            if result:
                logger.info(
                    "Compressed %s: %.1fx ratio (%d points)",
                    output_path,
                    result.get("compression_ratio", 0),
                    result.get("point_count", 0),
                    extra={"tag": "TRAIN"},
                )
        except Exception as exc:
            logger.debug("Pugqeep compression skipped: %s", exc, extra={"tag": "TRAIN"})

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
        """Generate text."""
        self.model.eval()
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(prompt)
            out = self.model.generate(
                np.array([ids], dtype=np.int64),
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            out = np.asarray(out)
            return self.tokenizer.decode(out[0].tolist())
        idx = np.array([[self.stoi.get(c, 0) for c in prompt]], dtype=np.int64)
        out = self.model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        out = np.asarray(out)

        text = "".join([self.itos.get(int(i), "?") for i in out[0]])
        return text


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for standalone training."""
    import argparse

    _epilog = (
        "checkpoints in --checkpoint-dir are .soul files and include "
        "stoi/itos/chars for char-LM eval. "
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
        help="Resume from this checkpoint (.soul or .npz)",
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

    try:
        if args.resume_latest:
            trainer.train(resume=True, resume_path=None)
        elif args.resume:
            trainer.train(resume=True, resume_path=args.resume)
        else:
            trainer.train()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    logger.info("=== Generated Text ===")
    logger.info("%s", trainer.generate("First"))


if __name__ == "__main__":  # pragma: no cover (entry-point guard)
    main()
