"""
RenderNeuralDevice — neural processor for Cycles renderer state tensors.

Bridges the path tracer output into the neural processing pipeline.
Takes state tensors (image, depth, normal, albedo, emission, mask) and
runs a lightweight numpy CNN to produce:
  - Scene feature embeddings (fixed-size vector)
  - Per-pixel semantic labels
  - Scene descriptor (summary statistics)
  - Material classification

All computation is pure numpy — no model downloads required.

Device bus protocol:
    DEV_OPEN   R0, render_neural
    DEV_CALL   R1, R0, process          # R1 = dict of neural outputs
    DEV_CALL   R1, R0, embed            # R1 = scene embedding vector
    DEV_CALL   R1, R0, classify         # R1 = per-pixel material classes
    DEV_CALL   R1, R0, descriptor       # R1 = scene summary dict
"""

from __future__ import annotations

import numpy as np

from .vm import Device, DeviceFault
from .cycles_device import CyclesDevice


class RenderNeuralDevice(Device):
    """VM device that processes rendered state tensors through a neural pipeline."""

    def __init__(self, cycles_device: CyclesDevice | None = None,
                 embed_dim: int = 64, num_classes: int = 8):
        self._cycles = cycles_device
        self._embed_dim = embed_dim
        self._num_classes = num_classes
        self._last_inputs: dict[str, np.ndarray] | None = None
        self._last_outputs: dict[str, np.ndarray] | None = None

        # Learnable conv weights (Xavier init)
        rng = np.random.RandomState(42)
        self._conv1_w = rng.randn(16, 6, 3, 3).astype(np.float32) * np.sqrt(2.0 / (6 * 9))
        self._conv1_b = np.zeros(16, dtype=np.float32)
        self._conv2_w = rng.randn(32, 16, 3, 3).astype(np.float32) * np.sqrt(2.0 / (16 * 9))
        self._conv2_b = np.zeros(32, dtype=np.float32)
        self._proj_w = rng.randn(32, embed_dim).astype(np.float32) * np.sqrt(2.0 / 32)
        self._proj_b = np.zeros(embed_dim, dtype=np.float32)
        self._classify_w = rng.randn(32, num_classes).astype(np.float32) * np.sqrt(2.0 / 32)
        self._classify_b = np.zeros(num_classes, dtype=np.float32)

        self._ops = {
            "process": self._process,
            "embed": self._embed,
            "classify": self._classify,
            "descriptor": self._descriptor,
            "info": self.info,
            "set_source": self._set_source,
            "forward": self._forward_raw,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"RenderNeuralDevice: unknown op: {method}")
        return fn(*args)

    def info(self):
        return {
            "type": "render_neural",
            "ops": list(self._ops.keys()),
            "embed_dim": self._embed_dim,
            "num_classes": self._num_classes,
            "has_source": self._cycles is not None,
            "input_channels": 6,
        }

    def _ensure_source(self) -> dict[str, np.ndarray]:
        if self._cycles is None:
            raise DeviceFault("RenderNeuralDevice: no CyclesDevice source set")
        return self._cycles.call("state_tensors")

    def _stack_channels(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        """Stack state tensors into (1, 6, H, W) — 6 channels:
        image RGB (3) + depth (1) + normal-Y (1) + albedo-R (1)."""
        chans = []
        H, W = 0, 0
        for key in ("image", "depth", "normal", "albedo"):
            t = tensors.get(key)
            if t is None:
                continue
            if t.ndim == 3:
                H, W = max(H, t.shape[0]), max(W, t.shape[1])
        if H == 0:
            raise DeviceFault("RenderNeuralDevice: no valid state tensors")

        img = tensors.get("image")
        if img is not None:
            chans.extend([img[..., i] for i in range(min(3, img.shape[-1]))])
        else:
            chans.extend([np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))])

        depth = tensors.get("depth")
        if depth is not None:
            if depth.ndim == 2:
                chans.append(depth)
            else:
                # For 3D+, reduce to (H, W) by taking first element along trailing dims
                result = depth
                while result.ndim > 2:
                    result = result[..., 0]
                chans.append(result)
        else:
            chans.append(np.zeros((H, W)))

        normal = tensors.get("normal")
        if normal is not None and normal.ndim == 3 and normal.shape[-1] >= 2:
            chans.append(normal[..., 1])  # Y component
        else:
            chans.append(np.zeros((H, W)))

        albedo = tensors.get("albedo")
        if albedo is not None and albedo.ndim == 3:
            chans.append(albedo[..., 0])  # R component
        else:
            chans.append(np.zeros((H, W)))

        stacked = np.stack(chans, axis=0)  # (6, H, W)
        return stacked[np.newaxis].astype(np.float32)  # (1, 6, H, W)

    def _conv2d_relu(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully vectorized 2D conv with ReLU. x: (N,C,H,W), w: (C_out,C_in,k,k)."""
        N, C_in, H, W = x.shape
        C_out, _, k, _ = w.shape
        pad = k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

        # Build col indices once
        if not hasattr(self, '_col_indices') or self._col_shape != (C_in, k, H, W):
            idx_h = np.arange(k)[:, None] + np.arange(H)[None, :]
            idx_w = np.arange(k)[:, None] + np.arange(W)[None, :]
            ih, iw = np.meshgrid(idx_h.ravel(), idx_w.ravel(), indexing='ij')
            self._col_indices = (ih, iw)
            self._col_shape = (C_in, k, H, W)

        ih, iw = self._col_indices
        cols = xp[:, :, ih, iw]  # (N, C_in, k*k, H*W)
        cols = cols.reshape(N, C_in * k * k, H * W)  # (N, C_in*k*k, H*W)

        w_mat = w.reshape(C_out, -1)  # (C_out, C_in*k*k)
        cols_t = cols.transpose(0, 2, 1)  # (N, H*W, C_in*k*k)
        out = cols_t @ w_mat.T + b  # (N, H*W, C_out)
        out = np.maximum(out, 0.0)
        return out.reshape(N, H, W, C_out).transpose(0, 3, 1, 2)  # (N, C_out, H, W)

    def _adaptive_avg_pool(self, x: np.ndarray, size: int = 1) -> np.ndarray:
        """Global average pool: (N, C, H, W) → (N, C)."""
        return x.mean(axis=(2, 3))

    def _forward(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Forward pass through the neural pipeline."""
        h1 = self._conv2d_relu(x, self._conv1_w, self._conv1_b)  # (1, 16, H, W)
        h2 = self._conv2d_relu(h1, self._conv2_w, self._conv2_b)  # (1, 32, H, W)
        pooled = self._adaptive_avg_pool(h2)  # (1, 32)

        # Embedding
        embed = pooled @ self._proj_w + self._proj_b  # (1, embed_dim)
        embed = embed / (np.linalg.norm(embed, axis=-1, keepdims=True) + 1e-8)

        # Classification (per-pixel via global feature broadcast)
        logits = pooled @ self._classify_w + self._classify_b  # (1, num_classes)
        probs = _softmax(logits, axis=-1)

        return {
            "embedding": embed.squeeze(0),
            "logits": logits.squeeze(0),
            "probabilities": probs.squeeze(0),
            "features": pooled.squeeze(0),
        }

    def _process(self):
        """Full pipeline: render → process → outputs."""
        tensors = self._ensure_source()
        self._last_inputs = tensors
        x = self._stack_channels(tensors)
        self._last_outputs = self._forward(x)
        return self._last_outputs

    def _embed(self):
        """Return scene embedding vector only."""
        tensors = self._ensure_source()
        x = self._stack_channels(tensors)
        out = self._forward(x)
        return out["embedding"]

    def _classify(self):
        """Return per-pixel material classification."""
        tensors = self._ensure_source()
        x = self._stack_channels(tensors)
        out = self._forward(x)

        H, W = tensors.get("image", np.zeros((1, 1, 3))).shape[:2]
        logits_2d = out["probabilities"].repeat(H * W).reshape(H, W, -1)
        labels = np.argmax(logits_2d, axis=-1)
        return {"labels": labels, "probabilities": logits_2d}

    def _descriptor(self):
        """Scene summary statistics from state tensors."""
        tensors = self._ensure_source()
        desc = {}
        for key, t in tensors.items():
            if isinstance(t, np.ndarray) and t.size > 0:
                desc[key] = {
                    "mean": float(np.mean(t)),
                    "std": float(np.std(t)),
                    "min": float(np.min(t)),
                    "max": float(np.max(t)),
                    "shape": list(t.shape),
                }
        # Run neural features
        x = self._stack_channels(tensors)
        out = self._forward(x)
        desc["neural_embedding_norm"] = float(np.linalg.norm(out["embedding"]))
        desc["neural_entropy"] = float(-np.sum(out["probabilities"] * np.log(out["probabilities"] + 1e-10)))
        desc["dominant_class"] = int(np.argmax(out["probabilities"]))
        return desc

    def _set_source(self, cycles_device):
        """Set or change the CyclesDevice source."""
        self._cycles = cycles_device
        self._last_inputs = None
        self._last_outputs = None

    def _forward_raw(self, *args):
        """Raw forward pass for integration with NeuralEngineDevice ioctl."""
        if args and isinstance(args[0], dict):
            x = args[0].get("state_tensors")
            if x is not None:
                stacked = self._stack_channels(x)
                return self._forward(stacked)
        if self._last_inputs is not None:
            x = self._stack_channels(self._last_inputs)
            return self._forward(x)
        raise DeviceFault("RenderNeuralDevice: no inputs available")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)
