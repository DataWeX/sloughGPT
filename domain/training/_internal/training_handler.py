"""Training handler — single composable training engine.

Everything training-related lives here:
    - Protocols: ``BatchSampler``, ``GradientHandler``, ``LossTracker``, ``CheckpointSaver``
    - Implementations: samplers, gradient handlers, loss trackers, checkpoint savers
    - Utilities: ``clip_gradients``, ``zero_grads``, ``build_char_vocab``

No other training files.  This is it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("slo.training.handler")


# =============================================================================
# Utilities
# =============================================================================


def clip_gradients(params: list, max_norm: float) -> float:
    """Compute total gradient norm and clip in-place.

    Returns the unclipped total norm (before scaling).
    """
    if max_norm <= 0 or not params:
        return 0.0

    total_norm = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            g = _as_ndarray(p.grad)
            grads.append(g)
            total_norm += float(np.sum(g**2))
    total_norm = total_norm**0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale

    return total_norm


def _as_ndarray(grad) -> np.ndarray:
    """Extract the raw numpy array from a gradient, however wrapped."""
    if isinstance(grad, np.ndarray):
        return grad
    if hasattr(grad, "data") and isinstance(grad.data, np.ndarray):
        return grad.data
    return np.asarray(grad)


def zero_grads(params: list) -> None:
    """Set all gradients to None (lightweight, equivalent to zeroing)."""
    for p in params:
        p.grad = None


def build_char_vocab(
    texts: list[str],
    special_tokens: dict[str, int] | None = None,
    offset: int = 0,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-level stoi/itos from a list of texts.

    Args:
        texts: Strings to extract characters from.
        special_tokens: Pre-assigned token->id mappings (e.g. ``{"<pad>": 0}``).
        offset: Starting index for character tokens.

    Returns:
        (stoi, itos)
    """
    chars: set[str] = set()
    for text in texts:
        chars.update(text)
    sorted_chars = sorted(chars)

    if special_tokens is not None:
        stoi = dict(special_tokens)
        start = max(stoi.values()) + 1 if stoi else offset
    else:
        stoi = {}
        start = offset

    for i, ch in enumerate(sorted_chars):
        stoi[ch] = start + i

    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class BatchSampler(Protocol):
    """Provides (x, y) batches for training."""

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class GradientHandler(Protocol):
    """Handles backward pass, gradient clipping, and optimizer step."""

    def step(
        self, model: Any, loss: Any, optimizer: Any, scheduler: Any = None
    ) -> dict[str, float]: ...
    def zero_grads(self, model: Any) -> None: ...


@runtime_checkable
class LossTracker(Protocol):
    """Tracks and optionally smooths loss values."""

    def update(self, loss_val: float) -> float: ...

    @property
    def current(self) -> float: ...


@runtime_checkable
class CheckpointSaver(Protocol):
    """Saves and loads model checkpoints."""

    def save(self, model: Any, name: str, **metadata: Any) -> str: ...
    def load(self, path: str, model: Any) -> bool: ...
    def latest_path(self) -> str | None: ...


# =============================================================================
# BatchSampler implementations
# =============================================================================


class RandomBlockSampler:
    """Random block sampling from a flat token array (train_pipeline style)."""

    def __init__(self, data: np.ndarray, block_size: int) -> None:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.int64)
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self)
        if n <= 0:
            raise ValueError("Dataset too small for block_size")
        idx = np.random.randint(0, n, size=batch_size)
        offsets = np.arange(self.block_size)
        x = self.data[idx[:, None] + offsets]
        y = self.data[idx[:, None] + offsets + 1]
        return x.astype(np.int32), y.astype(np.int32)


class PermutationSampler:
    """Epoch-shuffle sampler (consciousness style)."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = np.asarray(x_data, dtype=np.int32)
        self.y_data = np.asarray(y_data, dtype=np.int32)
        self.n_samples = x_data.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.n_samples, size=batch_size)
        return self.x_data[idx], self.y_data[idx]


class ChatPairSampler:
    """Chat-pair sampler (chat_trainer style)."""

    def __init__(
        self,
        pairs: list[dict[str, str]],
        stoi: dict[str, int],
        block_size: int,
        seed: int = 42,
    ) -> None:
        self.block_size = block_size
        self.stoi = stoi
        self._rng = np.random.default_rng(seed)
        text = "".join(f"User: {p['user_msg']}\nAssistant: {p['assistant_msg']}\n\n" for p in pairs)
        self.ids = [stoi.get(c, 0) for c in text]
        self.n_samples = max(1, len(self.ids) - block_size - 1)

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self._rng.integers(0, self.n_samples, size=batch_size)
        offsets = np.arange(self.block_size)
        ids = np.asarray(self.ids, dtype=np.int32)
        pos = indices[:, None] + offsets
        x = ids[pos]
        y = ids[pos + 1]
        return x, y


# =============================================================================
# GradientHandler implementations
# =============================================================================


class DirectGradientHandler:
    """Single backward -> clip -> optimizer step -> zero grads."""

    def __init__(self, grad_clip: float = 1.0) -> None:
        self.grad_clip = grad_clip

    def step(
        self, model: Any, loss: Any, optimizer: Any, scheduler: Any = None
    ) -> dict[str, float]:
        loss.backward()
        params = [p for p in model.parameters() if p.grad is not None]
        total_norm = clip_gradients(params, self.grad_clip)
        optimizer.step(params)
        if scheduler is not None:
            scheduler.step()
        return {"loss": float(loss.item()), "grad_norm": total_norm}

    def zero_grads(self, model: Any) -> None:
        zero_grads(list(model.parameters()))


class AccumulationGradientHandler:
    """Gradient accumulation -- clips and steps every N micro-steps."""

    def __init__(self, grad_clip: float = 1.0, accum_steps: int = 1) -> None:
        self.grad_clip = grad_clip
        self.accum_steps = max(1, accum_steps)
        self._step_count = 0

    def step(
        self, model: Any, loss: Any, optimizer: Any, scheduler: Any = None
    ) -> dict[str, float]:
        scale = 1.0 / self.accum_steps
        (loss * scale).backward()
        self._step_count += 1

        metrics: dict[str, float] = {"loss": float(loss.item()) / scale}

        if self._step_count >= self.accum_steps:
            params = [p for p in model.parameters() if p.grad is not None]
            total_norm = clip_gradients(params, self.grad_clip)
            optimizer.step(params)
            zero_grads(list(model.parameters()))
            if scheduler is not None:
                scheduler.step()
            self._step_count = 0
            metrics["grad_norm"] = total_norm

        return metrics

    def zero_grads(self, model: Any) -> None:
        zero_grads(list(model.parameters()))


# =============================================================================
# LossTracker implementations
# =============================================================================


class RawLossTracker:
    """Simple running average of loss values."""

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0

    def update(self, loss_val: float) -> float:
        self._total += loss_val
        self._count += 1
        return self.current

    @property
    def current(self) -> float:
        return self._total / max(1, self._count)


class EMALossTracker:
    """Exponential moving average of loss values."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self._value: float | None = None

    def update(self, loss_val: float) -> float:
        if self._value is None:
            self._value = loss_val
        else:
            self._value = self.alpha * loss_val + (1 - self.alpha) * self._value
        return self.current

    @property
    def current(self) -> float:
        return self._value if self._value is not None else 0.0


# =============================================================================
# CheckpointSaver implementations
# =============================================================================


class SoulCheckpointSaver:
    """Saves model checkpoints in .soul format with rotation."""

    def __init__(self, checkpoint_dir: str = "checkpoints", max_keep: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save(self, model: Any, name: str, **metadata: Any) -> str:
        from domain.training._internal.slonet import export_to_sou

        path = self.checkpoint_dir / f"{name}.soul"
        export_to_sou(model, str(path), metadata=metadata)
        self._rotate()
        try:
            from domain.infrastructure._internal.artifact_registry import try_register

            try_register("checkpoint", path, name=name)
        except Exception:
            logger.debug("Checkpoint registry update skipped", exc_info=True)
        logger.info("Saved checkpoint: %s", path)
        return str(path)

    def load(self, path: str, model: Any) -> bool:
        from domain.training._internal.slonet import import_from_sou

        p = Path(path)
        if not p.is_file():
            logger.warning("Checkpoint not found: %s", p)
            return False
        try:
            loaded = import_from_sou(str(p))
            for name, param in model.named_parameters():
                if hasattr(loaded, "parameters"):
                    for lp in loaded.parameters():
                        if hasattr(lp, "name") and lp.name == name:
                            param.data = lp.data
                            break
            logger.info("Loaded checkpoint: %s", p)
            return True
        except Exception as e:
            logger.warning("Failed to load checkpoint %s: %s", p, e)
            return False

    def latest_path(self) -> str | None:
        souls = sorted(
            self.checkpoint_dir.glob("*.soul"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(souls[0]) if souls else None

    def _rotate(self) -> None:
        if self.max_keep <= 0:
            return
        souls = sorted(
            self.checkpoint_dir.glob("*.soul"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in souls[self.max_keep :]:
            try:
                stale.unlink()
            except OSError:
                pass


class NpzCheckpointSaver:
    """Saves filtered tensor weights as .npz (e.g. LoRA-only weights)."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        filter_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filter_fn = filter_fn or (lambda name: True)

    def save(self, model: Any, name: str, **metadata: Any) -> str:
        path = self.checkpoint_dir / f"{name}.npz"
        state = {}
        for pname, param in model.named_parameters():
            if self.filter_fn(pname):
                state[pname] = param.data if hasattr(param, "data") else param
        if state:
            np.savez(str(path), **state)
            try:
                from domain.infrastructure._internal.artifact_registry import try_register

                try_register("checkpoint", path, name=name)
            except Exception:
                logger.debug("Checkpoint registry update skipped", exc_info=True)
            logger.info("Saved checkpoint: %s (%d tensors)", path, len(state))
        return str(path)

    def load(self, path: str, model: Any) -> bool:
        p = Path(path)
        if not p.is_file():
            logger.warning("Checkpoint not found: %s", p)
            return False
        try:
            data = np.load(str(p))
            for name, param in model.named_parameters():
                if name in data:
                    param.data = data[name].astype(np.float32)
            logger.info("Loaded checkpoint: %s", p)
            return True
        except Exception as e:
            logger.warning("Failed to load checkpoint %s: %s", p, e)
            return False

    def latest_path(self) -> str | None:
        npzs = sorted(
            self.checkpoint_dir.glob("*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(npzs[0]) if npzs else None
