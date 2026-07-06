"""
SloNet-based activity classifier for 6-axis phone sensor data.

Architecture (dual-path):
  Path A — global statistics (mean, std, min, max per channel)
  Path B — temporal conv (2× Conv2D with residual → avg pool)
  Both paths concatenated → Linear → classes

  This captures both per-axis statistics (MLP path) and local
  temporal patterns (conv path) simultaneously.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, List

logger = logging.getLogger("man.activity")

_ACTIVITY_MODEL: Optional["ActivityClassifier"] = None


class ActivityClassifier:
    """Hybrid activity classifier: per-channel statistics + temporal conv features.

    Computes 4 statistics per axis (mean, std, min, max) for a global view,
    plus 2 conv layers for local temporal patterns. Both paths concatenated
    into a joint representation → Linear → classes.

    Args:
        num_classes: number of activity classes
        stats_dim: number of per-axis statistics × 6 channels (default 4×6=24)
        conv_dim: conv output channels at each layer
    """

    def __init__(
        self,
        num_classes: int = 6,
        stats_dim: int = 24,
        conv_dim: int = 32,
    ):
        self.num_classes = num_classes

        from domains.training.slonet import SloConv2D, SloLinear

        self.conv1 = SloConv2D(6, conv_dim, kernel_size=(1, 7), padding=(0, 3))
        self.conv2 = SloConv2D(conv_dim, conv_dim, kernel_size=(1, 5), padding=(0, 2))
        self.fc = SloLinear(stats_dim + conv_dim, num_classes)
        self._params: Optional[List["Tensor"]] = None

    def parameters(self) -> List["Tensor"]:
        if self._params is not None:
            return self._params
        self._params = (
            self.conv1.parameters()
            + self.conv2.parameters()
            + self.fc.parameters()
        )
        return self._params

    def save(self, path: str):
        """Serialize all weight arrays to .npz."""
        weights = {}
        for i, p in enumerate(self.parameters()):
            weights[f"arr_{i}"] = p.data
        np.savez_compressed(path, **weights)

    @classmethod
    def load(cls, path: str, num_classes: int = 6) -> "ActivityClassifier":
        """Deserialize weights from .npz, return a fresh model with those weights.

        If the saved fc weight's first dimension differs from num_classes,
        num_classes is auto-detected from the weight shape.
        """
        data = np.load(path)
        keys = sorted([k for k in data.keys() if k.startswith("arr_")])
        # The fc weight is second-to-last parameter (fc.weight before fc.bias)
        fc_weight = data[keys[-2]]
        detected = fc_weight.shape[0]
        if detected != num_classes:
            num_classes = detected
        model = cls(num_classes=num_classes)
        for p, k in zip(model.parameters(), keys):
            p.data[:] = data[k]
        return model

    def forward(self, x: "Tensor") -> "Tensor":
        from domains.training.slonet import Tensor, relu

        # x: (batch, T, 6)

        # Path A — temporal conv (with gradient flow)
        x_t = Tensor(x.data.transpose(0, 2, 1), requires_grad=True)
        x_t = Tensor(x_t.data[:, :, np.newaxis, :], requires_grad=True)

        h = relu(self.conv1.forward(x_t))
        h = relu(self.conv2.forward(h))            # (batch, conv_dim, 1, T)

        # Global avg pooling over time (with gradient)
        pooled = _global_mean(h, axis=3)           # (batch, conv_dim, 1, 1)
        sq = Tensor(pooled.data[:, :, 0, 0], requires_grad=True, _children=(pooled,))
        def _bk_sq(g):
            if pooled.requires_grad:
                pooled.grad = Tensor(g[:, :, np.newaxis, np.newaxis], requires_grad=False)
        sq._backward_fn = _bk_sq

        # Path B — per-axis statistics (numpy, no gradient)
        raw = x.data
        stats = np.concatenate([
            raw.mean(axis=1), raw.std(axis=1),
            raw.min(axis=1), raw.max(axis=1),
        ], axis=1).astype(np.float32)               # (batch, 24)
        stats_t = Tensor(stats, requires_grad=False)

        # Concatenate: conv features (requires_grad) + stats (detached)
        # Use numpy concat then wrap as Tensor with grad from sq path
        joint_data = np.concatenate([sq.data, stats], axis=1).astype(np.float32)
        joint = Tensor(joint_data, requires_grad=True, _children=(sq,))
        conv_dim = self.conv2.out_ch
        def _bk_joint(g):
            if sq.requires_grad:
                sq.grad = Tensor(g[:, :conv_dim], requires_grad=False)
        joint._backward_fn = _bk_joint

        return self.fc.forward(joint)


def _global_mean(t: "Tensor", axis: int) -> "Tensor":
    """Mean over one spatial axis with proper backward pass.

    For input (N, C, H, W) and axis=3, computes mean over W
    → (N, C, H, 1) — gradient is 1/N_W per element.
    """
    from domains.training.slonet import Tensor
    n = t.data.shape[axis]
    out_data = t.data.mean(axis=axis, keepdims=True)

    if not t.requires_grad:
        return Tensor(out_data, requires_grad=False)

    out = Tensor(out_data, requires_grad=True, _children=(t,))
    def bk(g):
        if t.requires_grad:
            t.grad = Tensor(np.broadcast_to(g / n, t.data.shape), requires_grad=False)
    out._backward_fn = bk
    return out


def _augment_batch(X: np.ndarray) -> np.ndarray:
    """Online data augmentation for 6-axis sensor windows.

    Applies a random composition of:
      - Gaussian noise (p=0.6) scaled to per-channel std
      - Amplitude scaling (p=0.5) per-channel uniform factor
      - Time shift  (p=0.3)  ±5 timestep circular
      - Channel dropout (p=0.15) zero out 1-2 channels

    Args:
        X: (batch, time_steps, 6) sensor data

    Returns:
        augmented copy of X (same shape, float32)
    """
    aug = X.copy()
    n, t, c = aug.shape

    # Per-channel std for noise scaling
    channel_std = aug.std(axis=(0, 1), keepdims=True) + 1e-6

    # 1. Gaussian noise (p=0.6)
    if np.random.random() < 0.6:
        noise_factor = np.random.uniform(0.03, 0.10)
        aug += np.random.randn(n, t, c).astype(np.float32) * channel_std * noise_factor

    # 2. Amplitude scaling (p=0.5)
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.85, 1.15, size=(1, 1, c)).astype(np.float32)
        aug *= scale

    # 3. Time shift (p=0.3)
    if np.random.random() < 0.3:
        shift = np.random.randint(-5, 6)
        if shift > 0:
            aug[:, :-shift, :] = aug[:, shift:, :]
            aug[:, -shift:, :] = 0.0
        elif shift < 0:
            aug[:, -shift:, :] = aug[:, :shift, :]
            aug[:, :-shift, :] = 0.0

    # 4. Channel dropout (p=0.15)
    if np.random.random() < 0.15:
        drop = np.random.choice(c, size=np.random.randint(1, min(3, c)), replace=False)
        aug[:, :, drop] = 0.0

    return aug


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 0.005,
    batch_size: int = 16,
    val_split: float = 0.2,
    augment: bool = True,
    verbose: bool = True,
    on_epoch: callable = None,
) -> ActivityClassifier:
    """Train activity classifier on collected sensor data.

    Args:
        X: (num_samples, time_steps, 6) sensor data
        y: (num_samples,) integer labels
        epochs: training epochs
        lr: learning rate
        batch_size: samples per batch
        val_split: fraction for validation
        augment: apply online data augmentation per batch
        verbose: print progress

    Returns:
        trained ActivityClassifier
    """
    from domains.training.slonet import (
        Tensor, cross_entropy, SloAdam, SloReduceLROnPlateau, clip_grad_norm_,
    )

    num_classes = len(np.unique(y))
    model = ActivityClassifier(num_classes=num_classes)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    opt = SloAdam(lr=lr, max_grad_norm=1.0, weight_decay=1e-4)
    scheduler = SloReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-6)
    current_lr = lr

    best_val_loss = float("inf")
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        losses = []
        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            x_batch = X_train[batch_idx]
            if augment:
                x_batch = _augment_batch(x_batch)
            xb = Tensor(x_batch, requires_grad=False)
            yb = Tensor(y_train[batch_idx], requires_grad=False)

            logits = model.forward(xb)
            loss = cross_entropy(logits, yb)
            losses.append(float(loss.data))

            loss.backward()
            opt.step(model.parameters())
            for p in model.parameters():
                p.grad = None

        xv = Tensor(X_val, requires_grad=False)
        yv = Tensor(y_val, requires_grad=False)
        val_logits = model.forward(xv)
        val_loss = float(cross_entropy(val_logits, yv).data)

        scheduler.step(val_loss)
        new_lr = opt.lr
        lr_note = f"  lr={new_lr:.2e}" if new_lr != current_lr else ""
        current_lr = new_lr

        val_acc = _accuracy(val_logits, y_val)
        avg_loss = np.mean(losses)
        best_val_loss = min(best_val_loss, val_loss)

        if verbose:
            print(f"  epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}{lr_note}")

        if on_epoch is not None:
            on_epoch(epoch=epoch + 1, epochs=epochs, loss=float(avg_loss),
                     val_loss=float(val_loss), val_accuracy=float(val_acc),
                     lr=float(current_lr))

    if verbose:
        print(f"  Done — best val_loss={best_val_loss:.4f}")

    # Persist to disk
    model_path = Path(__file__).resolve().parent / "model.npz"
    model.save(str(model_path))
    logger.info(f"Saved model to {model_path}")
    return model


def predict_activity(
    model: ActivityClassifier,
    sensor_data: np.ndarray,
) -> Tuple[int, str, np.ndarray]:
    """Predict activity from a single sensor reading.

    Args:
        model: trained ActivityClassifier
        sensor_data: (time_steps, 6) or (1, time_steps, 6) array

    Returns:
        (class_id, class_name, class_probabilities)
    """
    from domains.training.slonet import Tensor, softmax
    from .dataset import ACTIVITIES

    if sensor_data.ndim == 2:
        sensor_data = sensor_data[np.newaxis, :, :]

    x = Tensor(sensor_data.astype(np.float32), requires_grad=False)
    logits = model.forward(x)
    probs = softmax(logits)
    probs_np = probs.data[0]
    pred = int(np.argmax(probs_np))
    return pred, ACTIVITIES[pred], probs_np


def _accuracy(logits: "Tensor", targets: np.ndarray) -> float:
    if len(targets) == 0:
        return 0.0
    preds = np.argmax(logits.data, axis=1)
    return float((preds == targets).mean())
